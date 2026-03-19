"""
segmentation_metrics.py
=======================
Advanced segmental evaluation metrics for surgical phase segmentation.

Metrics
-------
1. Boundary F1-Score  – transition-point detection with a tolerance window τ
2. Edit Score         – workflow-level correctness via Levenshtein distance on
                        the run-length-compressed label sequence
3. Segmental F1@k     – IoU-thresholded segment-level TP/FP/FN for k ∈ {0.10, 0.25, 0.50}
4. Over-segmentation Error – relative excess of predicted segments vs ground truth

Usage
-----
    evaluator = SegmentationEvaluator()
    results   = evaluator.evaluate(y_true, y_pred, tau=15)
    evaluator.print_report(results)
"""

from __future__ import annotations

import numpy as np
import editdistance
from dataclasses import dataclass, field
from typing import List, Tuple, Dict


# ─────────────────────────────────────────────────────────────────────────────
# Data containers
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class BoundaryF1Result:
    precision:  float
    recall:     float
    f1:         float
    tau:        int
    n_gt_boundaries:   int
    n_pred_boundaries: int
    n_true_positives:  int


@dataclass
class EditScoreResult:
    score:              float   # 0-100
    levenshtein_dist:   int
    len_compressed_true: int
    len_compressed_pred: int


@dataclass
class SegmentalF1Result:
    f1_at_k:    Dict[float, float]  # threshold → F1
    precision_at_k: Dict[float, float]
    recall_at_k:    Dict[float, float]
    thresholds: List[float]


@dataclass
class EvaluationResult:
    boundary_f1:          BoundaryF1Result
    edit_score:           EditScoreResult
    segmental_f1:         SegmentalF1Result
    oversegmentation_err: float
    frame_accuracy:       float          # included for reference


# ─────────────────────────────────────────────────────────────────────────────
# Main evaluator
# ─────────────────────────────────────────────────────────────────────────────

class SegmentationEvaluator:
    """
    Compute advanced segmental metrics between a ground-truth and predicted
    label sequence for temporal action / surgical phase segmentation.

    All public methods accept 1-D array-like inputs of integer labels.
    Frames labelled -1 are treated as unlabelled and excluded from every
    metric before any computation begins.
    """

    # ── public API ────────────────────────────────────────────────────────────

    def evaluate(
        self,
        y_true: np.ndarray | list,
        y_pred: np.ndarray | list,
        tau: int = 15,
        segmental_thresholds: List[float] = (0.10, 0.25, 0.50),
    ) -> EvaluationResult:
        """
        Run all four metrics in one call.

        Parameters
        ----------
        y_true : array-like of int
            Ground-truth frame labels.
        y_pred : array-like of int
            Predicted frame labels.
        tau : int
            Tolerance window in frames for boundary matching (default 15).
        segmental_thresholds : sequence of float
            IoU thresholds for segmental F1 (default [0.10, 0.25, 0.50]).

        Returns
        -------
        EvaluationResult
        """
        y_true, y_pred = self._prepare(y_true, y_pred)

        return EvaluationResult(
            boundary_f1=self.boundary_f1(y_true, y_pred, tau=tau),
            edit_score=self.edit_score(y_true, y_pred),
            segmental_f1=self.segmental_f1(y_true, y_pred,
                                            thresholds=list(segmental_thresholds)),
            oversegmentation_err=self.oversegmentation_error(y_true, y_pred),
            frame_accuracy=float(np.mean(y_true == y_pred)),
        )

    def boundary_f1(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        tau: int = 15,
    ) -> BoundaryF1Result:
        """
        Boundary F1-Score with tolerance window τ.

        A predicted boundary is a True Positive (TP) when it lies within τ
        frames of *any unmatched* ground-truth boundary.  Each ground-truth
        boundary can only absorb one prediction (greedy nearest-first
        matching), preventing one GT boundary from being counted multiple
        times.

        Parameters
        ----------
        tau : int
            Maximum allowed frame distance for a match.
        """
        y_true, y_pred = self._prepare(y_true, y_pred)

        gt_bounds   = self._get_boundaries(y_true)
        pred_bounds = self._get_boundaries(y_pred)

        if len(gt_bounds) == 0 and len(pred_bounds) == 0:
            return BoundaryF1Result(1.0, 1.0, 1.0, tau, 0, 0, 0)

        matched_gt = set()
        tp = 0
        # Sort pred boundaries; for each, find the closest unmatched GT boundary
        for pb in sorted(pred_bounds):
            best_dist, best_gt = float('inf'), None
            for gb in gt_bounds:
                if gb in matched_gt:
                    continue
                d = abs(pb - gb)
                if d <= tau and d < best_dist:
                    best_dist, best_gt = d, gb
            if best_gt is not None:
                matched_gt.add(best_gt)
                tp += 1

        precision = tp / len(pred_bounds) if pred_bounds else 0.0
        recall    = tp / len(gt_bounds)   if gt_bounds   else 0.0
        f1        = (2 * precision * recall / (precision + recall)
                     if (precision + recall) > 0 else 0.0)

        return BoundaryF1Result(
            precision=precision, recall=recall, f1=f1, tau=tau,
            n_gt_boundaries=len(gt_bounds),
            n_pred_boundaries=len(pred_bounds),
            n_true_positives=tp,
        )

    def edit_score(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
    ) -> EditScoreResult:
        """
        Edit Score – workflow-level correctness.

        Both sequences are first run-length compressed so that consecutive
        identical labels merge into one symbol (e.g. [1,1,2,2,2,3] → [1,2,3]).
        The normalised Levenshtein distance between the two compressed
        sequences is then converted to a 0-100 score.

        Score = (1 − edit_dist / max(|compressed_true|, |compressed_pred|)) × 100
        """
        y_true, y_pred = self._prepare(y_true, y_pred)

        ct = self._compress(y_true)
        cp = self._compress(y_pred)

        dist  = editdistance.eval(ct, cp)
        denom = max(len(ct), len(cp))
        score = (1.0 - dist / denom) * 100.0 if denom > 0 else 100.0

        return EditScoreResult(
            score=score,
            levenshtein_dist=dist,
            len_compressed_true=len(ct),
            len_compressed_pred=len(cp),
        )

    def segmental_f1(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        thresholds: List[float] = (0.10, 0.25, 0.50),
    ) -> SegmentalF1Result:
        """
        Segmental F1@k  (also written F1@{10,25,50}).

        For each threshold k:
          - Extract continuous segments from y_true and y_pred.
          - A predicted segment is a TP if there exists an unmatched GT segment
            of the *same label* whose IoU with the prediction is ≥ k.
          - Precision = TP / len(pred_segments)
          - Recall    = TP / len(gt_segments)
          - F1        = harmonic mean

        Parameters
        ----------
        thresholds : list of float
            IoU thresholds, e.g. [0.10, 0.25, 0.50].
        """
        y_true, y_pred = self._prepare(y_true, y_pred)

        gt_segs   = self._get_segments(y_true)
        pred_segs = self._get_segments(y_pred)

        f1_at_k, prec_at_k, rec_at_k = {}, {}, {}

        for k in thresholds:
            tp = 0
            matched_gt = set()

            for ps in pred_segs:
                best_iou, best_idx = 0.0, -1
                for gi, gs in enumerate(gt_segs):
                    if gi in matched_gt:
                        continue
                    if gs[2] != ps[2]:          # different label → skip
                        continue
                    iou = self._segment_iou(ps, gs)
                    if iou > best_iou:
                        best_iou, best_idx = iou, gi
                if best_idx >= 0 and best_iou >= k:
                    matched_gt.add(best_idx)
                    tp += 1

            prec = tp / len(pred_segs) if pred_segs else 0.0
            rec  = tp / len(gt_segs)   if gt_segs   else 0.0
            f1   = (2 * prec * rec / (prec + rec)
                    if (prec + rec) > 0 else 0.0)

            f1_at_k[k]   = f1
            prec_at_k[k] = prec
            rec_at_k[k]  = rec

        return SegmentalF1Result(
            f1_at_k=f1_at_k,
            precision_at_k=prec_at_k,
            recall_at_k=rec_at_k,
            thresholds=list(thresholds),
        )

    def oversegmentation_error(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
    ) -> float:
        """
        Over-segmentation Error.

        Measures the relative difference in the total number of segments:

            OSE = |n_segments(y_pred) − n_segments(y_true)| / n_segments(y_true)

        Returns 0.0 when both have the same number of segments.
        A value > 1 means the predictor produces more than twice as many
        segments as the ground truth (severe chattering).
        """
        y_true, y_pred = self._prepare(y_true, y_pred)
        n_true = len(self._get_segments(y_true))
        n_pred = len(self._get_segments(y_pred))
        if n_true == 0:
            return 0.0
        return abs(n_pred - n_true) / n_true

    # ── pretty-print ──────────────────────────────────────────────────────────

    @staticmethod
    def print_report(result: EvaluationResult, title: str = "Segmentation Evaluation Report") -> None:
        """Print a formatted summary of all metrics."""
        sep = "─" * 55
        print(f"\n{'═' * 55}")
        print(f"  {title}")
        print(f"{'═' * 55}")

        print(f"\n  Frame Accuracy          : {result.frame_accuracy * 100:6.2f} %")

        bf = result.boundary_f1
        print(f"\n  Boundary F1  (τ={bf.tau:>2d} frames)")
        print(f"  {sep}")
        print(f"    GT boundaries         : {bf.n_gt_boundaries}")
        print(f"    Pred boundaries       : {bf.n_pred_boundaries}")
        print(f"    True Positives        : {bf.n_true_positives}")
        print(f"    Precision             : {bf.precision * 100:6.2f} %")
        print(f"    Recall                : {bf.recall * 100:6.2f} %")
        print(f"    F1                    : {bf.f1 * 100:6.2f} %")

        es = result.edit_score
        print(f"\n  Edit Score")
        print(f"  {sep}")
        print(f"    Compressed GT length  : {es.len_compressed_true}")
        print(f"    Compressed Pred length: {es.len_compressed_pred}")
        print(f"    Levenshtein distance  : {es.levenshtein_dist}")
        print(f"    Edit Score            : {es.score:6.2f} / 100")

        sf = result.segmental_f1
        print(f"\n  Segmental F1@k")
        print(f"  {sep}")
        print(f"    {'Threshold (k)':>14}  {'Precision':>10}  {'Recall':>8}  {'F1':>8}")
        for k in sf.thresholds:
            print(f"    {k:>14.2f}  "
                  f"{sf.precision_at_k[k] * 100:>9.2f}%  "
                  f"{sf.recall_at_k[k] * 100:>7.2f}%  "
                  f"{sf.f1_at_k[k] * 100:>7.2f}%")

        print(f"\n  Over-segmentation Error : {result.oversegmentation_err:.4f}")
        print(f"{'═' * 55}\n")

    # ── private helpers ───────────────────────────────────────────────────────

    @staticmethod
    def _prepare(
        y_true: np.ndarray | list,
        y_pred: np.ndarray | list,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Convert to numpy int arrays and remove unlabelled frames (label == -1)."""
        y_true = np.asarray(y_true, dtype=int)
        y_pred = np.asarray(y_pred, dtype=int)
        assert y_true.shape == y_pred.shape, (
            f"Shape mismatch: y_true {y_true.shape} vs y_pred {y_pred.shape}"
        )
        valid = y_true != -1
        return y_true[valid], y_pred[valid]

    @staticmethod
    def _get_boundaries(seq: np.ndarray) -> List[int]:
        """
        Return indices where the label changes.
        Boundary at index i means seq[i] != seq[i-1].
        """
        if len(seq) < 2:
            return []
        changes = np.where(seq[1:] != seq[:-1])[0] + 1
        return changes.tolist()

    @staticmethod
    def _compress(seq: np.ndarray) -> List[int]:
        """Run-length encode: keep only the first element of each run."""
        if len(seq) == 0:
            return []
        compressed = [seq[0]]
        for v in seq[1:]:
            if v != compressed[-1]:
                compressed.append(int(v))
        return compressed

    @staticmethod
    def _get_segments(seq: np.ndarray) -> List[Tuple[int, int, int]]:
        """
        Extract continuous segments as (start, end_inclusive, label) tuples.

        Example: [0,0,1,1,1,2] → [(0,1,0), (2,4,1), (5,5,2)]
        """
        if len(seq) == 0:
            return []
        segments = []
        start = 0
        for i in range(1, len(seq)):
            if seq[i] != seq[i - 1]:
                segments.append((start, i - 1, int(seq[start])))
                start = i
        segments.append((start, len(seq) - 1, int(seq[start])))
        return segments

    @staticmethod
    def _segment_iou(
        seg_a: Tuple[int, int, int],
        seg_b: Tuple[int, int, int],
    ) -> float:
        """
        Intersection over Union for two 1-D temporal segments.

        Segments are represented as (start, end_inclusive, label).
        Labels are assumed to match before calling this helper.
        """
        inter_start = max(seg_a[0], seg_b[0])
        inter_end   = min(seg_a[1], seg_b[1])
        intersection = max(0, inter_end - inter_start + 1)
        if intersection == 0:
            return 0.0
        union = (seg_a[1] - seg_a[0] + 1) + (seg_b[1] - seg_b[0] + 1) - intersection
        return intersection / union


# ─────────────────────────────────────────────────────────────────────────────
# Demo
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    np.random.seed(42)

    # ── synthetic ground truth: clean 6-phase surgical procedure ──────────────
    y_true = np.array(
        [0] * 60 +    # P0 Right Move
        [1] * 40 +    # P1 Pick Needle
        [2] * 55 +    # P2 Right Move2
        [3] * 35 +    # P3 Pass Needle
        [4] * 50 +    # P4 Left Move
        [5] * 30      # P5 Left Pick
    )

    # ── imperfect prediction: correct phases but with boundary offsets,
    #    some chattering, and one over-segmented region ─────────────────────────
    # Total must equal len(y_true) = 270
    y_pred = np.array(
        [0] * 55 + [1] * 5 +              # P0 region (60):  ends 5 frames early
        [1] * 38 + [2] * 2 +              # P1 region (40):  ends 2 frames early
        [2] * 10 + [3] * 5 + [2] * 40 +  # P2 region (55):  chattering burst
        [3] * 33 + [4] * 2 +              # P3 region (35):  ends 2 frames early
        [4] * 50 +                        # P4 region (50):  exact
        [5] * 28 + [4] * 2               # P5 region (30):  ends 2 frames early
    )

    # small unlabelled region injected into ground truth
    y_true_with_unlabelled = y_true.copy().astype(int)
    y_true_with_unlabelled[58:63] = -1   # boundary around P0→P1 transition

    print("=" * 55)
    print("  Demo 1 – clean prediction (with boundary offsets)")
    print("=" * 55)
    evaluator = SegmentationEvaluator()
    result = evaluator.evaluate(y_true_with_unlabelled, y_pred, tau=15)
    evaluator.print_report(result)

    # ── perfect prediction ────────────────────────────────────────────────────
    print("=" * 55)
    print("  Demo 2 – perfect prediction")
    print("=" * 55)
    result_perfect = evaluator.evaluate(y_true, y_true.copy(), tau=15)
    evaluator.print_report(result_perfect, title="Perfect Prediction Baseline")

    # ── highly over-segmented prediction (chattering) ─────────────────────────
    print("=" * 55)
    print("  Demo 3 – severe chattering")
    print("=" * 55)
    rng = np.random.default_rng(0)
    y_noisy = y_true.copy()
    flip_idx = rng.choice(len(y_noisy), size=40, replace=False)
    y_noisy[flip_idx] = rng.integers(0, 6, size=40)
    result_noisy = evaluator.evaluate(y_true, y_noisy, tau=15)
    evaluator.print_report(result_noisy, title="Chattering / Over-segmented")
