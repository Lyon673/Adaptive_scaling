"""
segmentation_metrics.py
=======================
Advanced segmental evaluation metrics for surgical phase segmentation.

Metrics
-------
1. Relaxed Frame Acc  - Accuracy with tolerance tau (e.g., tau=7, 3, 0)
2. Boundary F1-Score  – transition-point detection with a tolerance window τ
3. Edit Score         – workflow-level correctness via Levenshtein distance
4. Segmental F1@k     – IoU-thresholded segment-level TP/FP/FN for k (e.g., 0.70)
5. Over-segmentation Error – relative excess of predicted segments vs ground truth

Usage
-----
    evaluator = SegmentationEvaluator()
    results   = evaluator.evaluate(y_true, y_pred)
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
    frame_accuracy:       float          # tau=0 (Standard Accuracy, kept for backward compatibility)
    frame_acc_tau3:       float          # tau=3
    frame_acc_tau7:       float          # tau=7


# ─────────────────────────────────────────────────────────────────────────────
# Main evaluator
# ─────────────────────────────────────────────────────────────────────────────

class SegmentationEvaluator:
    """
    Compute advanced segmental metrics between a ground-truth and predicted
    label sequence for temporal action / surgical phase segmentation.
    """

    # ── public API ────────────────────────────────────────────────────────────

    def evaluate(
        self,
        y_true: np.ndarray | list,
        y_pred: np.ndarray | list,
        tau: int = 7,                                # 修改默认容差为 7
        segmental_thresholds: List[float] = (0.70,), # 修改默认阈值为 0.70
    ) -> EvaluationResult:
        """
        Run all metrics in one call.
        """
        y_true, y_pred = self._prepare(y_true, y_pred)

        return EvaluationResult(
            boundary_f1=self.boundary_f1(y_true, y_pred, tau=tau),
            edit_score=self.edit_score(y_true, y_pred),
            segmental_f1=self.segmental_f1(y_true, y_pred, thresholds=list(segmental_thresholds)),
            oversegmentation_err=self.oversegmentation_error(y_true, y_pred),
            frame_accuracy=self.relaxed_frame_accuracy(y_true, y_pred, tau=0),
            frame_acc_tau3=self.relaxed_frame_accuracy(y_true, y_pred, tau=3),
            frame_acc_tau7=self.relaxed_frame_accuracy(y_true, y_pred, tau=7),
        )

    def relaxed_frame_accuracy(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        tau: int
    ) -> float:
        """
        计算带有边界容差的帧准确率。
        忽略真实边界前后 tau 帧内的所有预测，承认人类标注的边界模糊性。
        """
        y_true, y_pred = self._prepare(y_true, y_pred)
        if tau == 0:
            return float(np.mean(y_true == y_pred))
            
        gt_bounds = self._get_boundaries(y_true)
        ignore_mask = np.zeros_like(y_true, dtype=bool)
        
        for b in gt_bounds:
            start_idx = max(0, b - tau)
            end_idx = min(len(y_true), b + tau)
            ignore_mask[start_idx:end_idx] = True
            
        valid_mask = ~ignore_mask
        
        if np.sum(valid_mask) == 0:
            return float(np.mean(y_true == y_pred))
            
        return float(np.mean(y_true[valid_mask] == y_pred[valid_mask]))

    def boundary_f1(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        tau: int = 7,
    ) -> BoundaryF1Result:
        """
        Boundary F1-Score with tolerance window τ.
        """
        y_true, y_pred = self._prepare(y_true, y_pred)

        gt_bounds   = self._get_boundaries(y_true)
        pred_bounds = self._get_boundaries(y_pred)

        if len(gt_bounds) == 0 and len(pred_bounds) == 0:
            return BoundaryF1Result(1.0, 1.0, 1.0, tau, 0, 0, 0)

        matched_gt = set()
        tp = 0
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
        thresholds: List[float] = (0.70,),
    ) -> SegmentalF1Result:
        """
        Segmental F1@k
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
                    if gs[2] != ps[2]:          
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
        """Print a formatted summary of all metrics in the specific requested order."""
        sep = "─" * 55
        print(f"\n{'═' * 55}")
        print(f"  {title}")
        print(f"{'═' * 55}")

        # 1-3. Accuracies
        print(f"\n  Frame Acc (Relaxed τ=7) : {result.frame_acc_tau7 * 100:6.2f} %")
        print(f"  Frame Acc (Relaxed τ=3) : {result.frame_acc_tau3 * 100:6.2f} %")
        print(f"  Frame Acc (Standard τ=0): {result.frame_accuracy * 100:6.2f} %")

        # 4. Boundary F1
        bf = result.boundary_f1
        print(f"\n  Boundary F1  (τ={bf.tau:>2d} frames)")
        print(f"  {sep}")
        print(f"    GT boundaries         : {bf.n_gt_boundaries}")
        print(f"    Pred boundaries       : {bf.n_pred_boundaries}")
        print(f"    True Positives        : {bf.n_true_positives}")
        print(f"    Precision             : {bf.precision * 100:6.2f} %")
        print(f"    Recall                : {bf.recall * 100:6.2f} %")
        print(f"    F1                    : {bf.f1 * 100:6.2f} %")

        # 5. Edit Score
        es = result.edit_score
        print(f"\n  Edit Score")
        print(f"  {sep}")
        print(f"    Compressed GT length  : {es.len_compressed_true}")
        print(f"    Compressed Pred length: {es.len_compressed_pred}")
        print(f"    Levenshtein distance  : {es.levenshtein_dist}")
        print(f"    Edit Score            : {es.score:6.2f} / 100")

        # 6. Segmental F1
        sf = result.segmental_f1
        print(f"\n  Segmental F1@k")
        print(f"  {sep}")
        print(f"    {'Threshold (k)':>14}  {'Precision':>10}  {'Recall':>8}  {'F1':>8}")
        for k in sf.thresholds:
            print(f"    {k:>14.2f}  "
                  f"{sf.precision_at_k[k] * 100:>9.2f}%  "
                  f"{sf.recall_at_k[k] * 100:>7.2f}%  "
                  f"{sf.f1_at_k[k] * 100:>7.2f}%")

        # 7. Oversegmentation Error
        print(f"\n  Over-segmentation Error : {result.oversegmentation_err:.4f}")
        print(f"{'═' * 55}\n")

    # ── private helpers ───────────────────────────────────────────────────────

    @staticmethod
    def _prepare(
        y_true: np.ndarray | list,
        y_pred: np.ndarray | list,
    ) -> Tuple[np.ndarray, np.ndarray]:
        y_true = np.asarray(y_true, dtype=int)
        y_pred = np.asarray(y_pred, dtype=int)
        assert y_true.shape == y_pred.shape, (
            f"Shape mismatch: y_true {y_true.shape} vs y_pred {y_pred.shape}"
        )
        valid = y_true != -1
        return y_true[valid], y_pred[valid]

    @staticmethod
    def _get_boundaries(seq: np.ndarray) -> List[int]:
        if len(seq) < 2:
            return []
        changes = np.where(seq[1:] != seq[:-1])[0] + 1
        return changes.tolist()

    @staticmethod
    def _compress(seq: np.ndarray) -> List[int]:
        if len(seq) == 0:
            return []
        compressed = [seq[0]]
        for v in seq[1:]:
            if v != compressed[-1]:
                compressed.append(int(v))
        return compressed

    @staticmethod
    def _get_segments(seq: np.ndarray) -> List[Tuple[int, int, int]]:
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

    y_true = np.array(
        [0] * 60 +    
        [1] * 40 +    
        [2] * 55 +    
        [3] * 35 +    
        [4] * 50 +    
        [5] * 30      
    )

    y_pred = np.array(
        [0] * 55 + [1] * 5 +              
        [1] * 38 + [2] * 2 +              
        [2] * 10 + [3] * 5 + [2] * 40 +  
        [3] * 33 + [4] * 2 +              
        [4] * 50 +                        
        [5] * 28 + [4] * 2               
    )

    y_true_with_unlabelled = y_true.copy().astype(int)
    y_true_with_unlabelled[58:63] = -1   

    print("=" * 55)
    print("  Demo 1 – clean prediction (with boundary offsets)")
    print("=" * 55)
    evaluator = SegmentationEvaluator()
    # 采用新的默认参数进行测试
    result = evaluator.evaluate(y_true_with_unlabelled, y_pred)
    evaluator.print_report(result)

    print("=" * 55)
    print("  Demo 2 – perfect prediction")
    print("=" * 55)
    result_perfect = evaluator.evaluate(y_true, y_true.copy())
    evaluator.print_report(result_perfect, title="Perfect Prediction Baseline")

    print("=" * 55)
    print("  Demo 3 – severe chattering")
    print("=" * 55)
    rng = np.random.default_rng(0)
    y_noisy = y_true.copy()
    flip_idx = rng.choice(len(y_noisy), size=40, replace=False)
    y_noisy[flip_idx] = rng.integers(0, 6, size=40)
    result_noisy = evaluator.evaluate(y_true, y_noisy)
    evaluator.print_report(result_noisy, title="Chattering / Over-segmented")