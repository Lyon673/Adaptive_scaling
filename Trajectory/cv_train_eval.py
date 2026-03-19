"""
cv_train_eval.py
================
5-fold cross-validation for the surgical-phase LSTM segmentation model.

Each fold:
  1. Splits all demo_ids into train / val subsets.
  2. Fits feature scalers on the training split (prevents data leakage).
  3. Trains a fresh SequenceLabelingLSTM for NUM_EPOCHS epochs.
  4. Evaluates the trained model with both frame-level and segmental metrics.

Results are printed per-fold and aggregated (mean ± std) at the end.
Fold models and a CSV summary are saved under  Trajectory/cv_results/ .

Usage
-----
    python cv_train_eval.py                      # 5-fold, 800 epochs (full run)
    python cv_train_eval.py --folds 5 --epochs 50 --quick   # quick smoke-test
"""

import argparse
import os
import csv
import datetime
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from sklearn.model_selection import KFold
from tqdm import tqdm

# ── project imports ───────────────────────────────────────────────────────────
from LSTM_seg_train import (
    SequenceLabelingLSTM,
    INPUT_SIZE, HIDDEN_SIZE, NUM_LAYERS, NUM_CLASSES, BATCH_SIZE,
    collate_fn,
)
from load_data import (
    load_demonstrations_state,
    load_demonstrations_label,
    _scale_demos,
    get_shuffled_demo_ids,
)
from segmentation_metrics import SegmentationEvaluator
from config import resample, without_quat


# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────

EXCLUDED_IDS  = [80, 81, 92, 109, 112, 117, 122, 144, 145]
LEARNING_RATE = 1e-3
NUM_EPOCHS    = 800       # overridden by CLI --epochs
N_FOLDS       = 5         # overridden by CLI --folds
RANDOM_SEED   = 42

DIR_PATH   = os.path.dirname(os.path.abspath(__file__))
RESULT_DIR = os.path.join(DIR_PATH, "cv_results")

CLASS_NAMES = [
    'P0 Right Move', 'P1 Pick Needle', 'P2 Right Move2',
    'P3 Pass Needle', 'P4 Left Move',  'P5 Left Pick',
    'P6 Pull Thread',
]


# ─────────────────────────────────────────────────────────────────────────────
# Dataset that wraps pre-loaded arrays
# ─────────────────────────────────────────────────────────────────────────────

class FoldDataset(Dataset):
    """
    Wraps lists of (state_array, label_array) pairs for one fold split.
    State arrays are already scaled before being passed in.
    """

    def __init__(self, states: list, labels: list):
        self.samples = []
        for state_seq, label_seq in zip(states, labels):
            s = torch.tensor(state_seq, dtype=torch.float32)
            l = torch.tensor(label_seq, dtype=torch.long)
            if s.dim() == 1:
                s = s.unsqueeze(1)
            # align lengths (safety guard)
            min_len = min(s.shape[0], l.shape[0])
            self.samples.append((s[:min_len], l[:min_len]))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


# ─────────────────────────────────────────────────────────────────────────────
# Training helper
# ─────────────────────────────────────────────────────────────────────────────

def train_one_fold(
    train_states: list,
    train_labels: list,
    device: torch.device,
    num_epochs: int,
    fold_idx: int,
) -> SequenceLabelingLSTM:
    """Train a fresh LSTM model on one fold's training split."""

    dataset = FoldDataset(train_states, train_labels)
    loader  = DataLoader(dataset, batch_size=BATCH_SIZE,
                         shuffle=True, collate_fn=collate_fn, drop_last=False)

    model     = SequenceLabelingLSTM(INPUT_SIZE, HIDDEN_SIZE, NUM_LAYERS, NUM_CLASSES).to(device)
    criterion = nn.CrossEntropyLoss(ignore_index=-1)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    pbar = tqdm(range(num_epochs),
                desc=f"  Fold {fold_idx + 1} training",
                leave=False, dynamic_ncols=True)

    for epoch in pbar:
        model.train()
        total_loss = 0.0
        for sequences, labels, lengths in loader:
            sequences = sequences.to(device)
            labels    = labels.to(device)
            outputs   = model(sequences, lengths)
            loss      = criterion(outputs.view(-1, NUM_CLASSES), labels.view(-1))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(loader)
        if (epoch + 1) % max(1, num_epochs // 10) == 0:
            pbar.set_postfix(loss=f"{avg_loss:.4f}")

    return model


# ─────────────────────────────────────────────────────────────────────────────
# Evaluation helper
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_fold(
    model: SequenceLabelingLSTM,
    val_states: list,
    val_labels: list,
    val_demo_ids: list,
    device: torch.device,
    evaluator: SegmentationEvaluator,
    tau: int = 15,
    seg_thresholds: tuple = (0.10, 0.25, 0.50),
) -> dict:
    """
    Evaluate a trained model on the validation split.
    Returns a dict with per-sequence and aggregate metrics.
    """
    model.eval()
    dataset = FoldDataset(val_states, val_labels)

    # Frame-level accumulators (all sequences flattened)
    all_preds_flat, all_labels_flat = [], []

    # Segmental accumulators (per sequence)
    seg_accs, seg_bfs, seg_eds, seg_oses = [], [], [], []
    seg_f1s = {k: [] for k in seg_thresholds}

    seq_records = []   # for per-sequence reporting

    with torch.no_grad():
        for idx, (state_seq, label_seq) in enumerate(dataset):
            seq_t  = state_seq.to(device)
            logits = model(seq_t.unsqueeze(0), [seq_t.shape[0]])
            pred_np = torch.argmax(logits, dim=2).squeeze(0).cpu().numpy()
            true_np = label_seq.numpy()

            min_len = min(len(true_np), len(pred_np))
            true_np = true_np[:min_len]
            pred_np = pred_np[:min_len]

            # frame-level (filter out -1)
            valid  = true_np != -1
            all_labels_flat.extend(true_np[valid])
            all_preds_flat.extend(pred_np[valid])

            # segmental
            if not np.all(true_np == -1):
                result = evaluator.evaluate(true_np, pred_np,
                                            tau=tau,
                                            segmental_thresholds=seg_thresholds)
                seg_accs.append(result.frame_accuracy)
                seg_bfs.append(result.boundary_f1.f1)
                seg_eds.append(result.edit_score.score)
                seg_oses.append(result.oversegmentation_err)
                for k in seg_thresholds:
                    seg_f1s[k].append(result.segmental_f1.f1_at_k[k])
                seq_records.append({
                    'demo_id'   : val_demo_ids[idx],
                    'acc'       : result.frame_accuracy,
                    'bound_f1'  : result.boundary_f1.f1,
                    'edit_score': result.edit_score.score,
                    **{f'f1@{k:.2f}': result.segmental_f1.f1_at_k[k]
                       for k in seg_thresholds},
                    'ose'       : result.oversegmentation_err,
                })

    # overall frame accuracy (using full flattened arrays)
    overall_acc = np.mean(np.array(all_labels_flat) == np.array(all_preds_flat))

    aggregate = {
        'overall_frame_acc' : overall_acc,
        'mean_acc'          : np.mean(seg_accs)  if seg_accs else 0.0,
        'mean_bound_f1'     : np.mean(seg_bfs)   if seg_bfs  else 0.0,
        'mean_edit_score'   : np.mean(seg_eds)   if seg_eds  else 0.0,
        'mean_ose'          : np.mean(seg_oses)  if seg_oses else 0.0,
        **{f'mean_f1@{k:.2f}': np.mean(seg_f1s[k]) for k in seg_thresholds},
    }

    return {'aggregate': aggregate, 'per_sequence': seq_records}


# ─────────────────────────────────────────────────────────────────────────────
# Data loading  (raw, unscaled, fixed order)
# ─────────────────────────────────────────────────────────────────────────────

def load_all_data():
    """
    Load every available demo in a **fixed, reproducible order** (no train/test
    split, no scaling).  Returns parallel lists of raw state arrays and label
    arrays, plus the corresponding demo_id list.
    """
    all_ids = np.arange(148)
    all_ids = np.delete(all_ids, EXCLUDED_IDS)

    # shuffle=False → preserve deterministic order for CV splits
    raw_states = load_demonstrations_state(
        shuffle=False, without_quat=without_quat,
        resample=resample, demo_id_list=all_ids,
    )
    raw_labels = load_demonstrations_label(
        shuffle=False, resample=resample, demo_id_list=all_ids,
    )

    # align lengths per demo (same safety guard as in TestDataset)
    states, labels, demo_ids = [], [], []
    for i, (s, l) in enumerate(zip(raw_states, raw_labels)):
        min_len = min(len(s), len(l))
        if min_len == 0:
            continue
        states.append(s[:min_len])
        labels.append(l[:min_len])
        demo_ids.append(int(all_ids[i]))

    return states, labels, demo_ids


# ─────────────────────────────────────────────────────────────────────────────
# Main cross-validation loop
# ─────────────────────────────────────────────────────────────────────────────

def run_cross_validation(n_folds: int, num_epochs: int):
    device    = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    evaluator = SegmentationEvaluator()
    seg_thresholds = (0.10, 0.25, 0.50)

    print(f"\n{'═' * 65}")
    print(f"  5-Fold Cross-Validation  │  folds={n_folds}  epochs={num_epochs}")
    print(f"  device={device}  input_size={INPUT_SIZE}  "
          f"hidden={HIDDEN_SIZE}  layers={NUM_LAYERS}")
    print(f"{'═' * 65}\n")

    os.makedirs(RESULT_DIR, exist_ok=True)

    # ── load raw data once ────────────────────────────────────────────────────
    print("Loading all demonstrations …")
    all_states, all_labels, all_demo_ids = load_all_data()
    n_demos = len(all_states)
    print(f"  {n_demos} demos loaded  →  {n_folds}-fold split "
          f"(~{n_demos // n_folds} val / ~{n_demos - n_demos // n_folds} train per fold)\n")

    kf = KFold(n_splits=n_folds, shuffle=True, random_state=RANDOM_SEED)
    indices = np.arange(n_demos)

    fold_results = []   # list of aggregate dicts, one per fold

    for fold_idx, (train_idx, val_idx) in enumerate(kf.split(indices)):
        print(f"{'─' * 65}")
        print(f"  FOLD {fold_idx + 1} / {n_folds}   "
              f"(train={len(train_idx)}  val={len(val_idx)})")
        print(f"{'─' * 65}")

        train_states_raw = [all_states[i] for i in train_idx]
        train_labels_raw = [all_labels[i] for i in train_idx]
        val_states_raw   = [all_states[i] for i in val_idx]
        val_labels_raw   = [all_labels[i] for i in val_idx]
        val_demo_ids     = [all_demo_ids[i] for i in val_idx]

        # ── scale: fit on train, apply to both ───────────────────────────────
        print("  Fitting scalers on training split …")
        train_states_scaled, scalers = _scale_demos(train_states_raw)

        val_states_scaled = []
        for arr in val_states_raw:
            from load_data import VELSCALAR_COLS, POS_COL_GROUPS, VEL3_COL_GROUPS
            out = arr.copy()
            out[:, VELSCALAR_COLS] = scalers['vel_scalar'].transform(arr[:, VELSCALAR_COLS])
            for group, sc in zip(POS_COL_GROUPS, scalers['pos']):
                out[:, group] = sc.transform(arr[:, group])
            for group, sc in zip(VEL3_COL_GROUPS, scalers['vel3']):
                out[:, group] = sc.transform(arr[:, group])
            val_states_scaled.append(out)

        # ── train ─────────────────────────────────────────────────────────────
        model = train_one_fold(
            train_states_scaled, train_labels_raw,
            device, num_epochs, fold_idx,
        )

        # ── save fold model ───────────────────────────────────────────────────
        model_path = os.path.join(RESULT_DIR, f"fold{fold_idx + 1}_model.pth")
        torch.save({
            'model_state_dict': model.state_dict(),
            'model_config': {
                'input_size': INPUT_SIZE, 'hidden_size': HIDDEN_SIZE,
                'num_layers': NUM_LAYERS, 'num_classes': NUM_CLASSES,
                'bidirectional': False,
            },
            'fold': fold_idx + 1,
            'val_demo_ids': val_demo_ids,
        }, model_path)

        # ── evaluate ──────────────────────────────────────────────────────────
        print(f"\n  Evaluating fold {fold_idx + 1} …")
        eval_out = evaluate_fold(
            model, val_states_scaled, val_labels_raw, val_demo_ids,
            device, evaluator,
            tau=15, seg_thresholds=seg_thresholds,
        )

        agg = eval_out['aggregate']
        fold_results.append({'fold': fold_idx + 1, **agg})

        # ── per-fold console summary ──────────────────────────────────────────
        print(f"\n  ┌─ Fold {fold_idx + 1} Results ────────────────────────────┐")
        print(f"  │  Frame Accuracy (overall)  : {agg['overall_frame_acc']*100:6.2f} %")
        print(f"  │  Boundary F1   (mean)      : {agg['mean_bound_f1']*100:6.2f} %")
        print(f"  │  Edit Score    (mean)      : {agg['mean_edit_score']:6.2f} / 100")
        for k in seg_thresholds:
            print(f"  │  F1@{k:.2f}        (mean)      : "
                  f"{agg[f'mean_f1@{k:.2f}']*100:6.2f} %")
        print(f"  │  Over-seg. Err (mean)      : {agg['mean_ose']:6.4f}")
        print(f"  └───────────────────────────────────────────────────┘\n")

        # ── per-sequence detail for this fold ────────────────────────────────
        k_list = list(seg_thresholds)
        hdr = (f"  {'demo_id':>8}  {'Acc':>6}  {'BndF1':>6}  {'EditSc':>7}"
               + "  ".join(f"  F1@{k:.2f}" for k in k_list)
               + "  {'OSE':>7}")
        print(f"  Per-sequence breakdown:")
        print(f"  {'demo_id':>8}  {'Acc%':>5}  {'BndF1%':>6}  {'EditSc':>7}"
              + "  " + "  ".join(f"F1@{k:.2f}%" for k in k_list)
              + "  {'OSE':>7}")
        print("  " + "─" * 60)
        for rec in eval_out['per_sequence']:
            f1_str = "  ".join(f"{rec[f'f1@{k:.2f}']*100:>7.1f}" for k in k_list)
            print(f"  {rec['demo_id']:>8}  "
                  f"{rec['acc']*100:>5.1f}  "
                  f"{rec['bound_f1']*100:>6.1f}  "
                  f"{rec['edit_score']:>7.1f}  "
                  f"{f1_str}  "
                  f"{rec['ose']:>7.4f}")

    # ── cross-validation aggregate summary ───────────────────────────────────
    print(f"\n{'═' * 65}")
    print(f"  Cross-Validation Summary  ({n_folds} folds × {num_epochs} epochs)")
    print(f"{'═' * 65}")
    metric_keys = [
        ('overall_frame_acc', 'Frame Accuracy (overall)',  '%', 100),
        ('mean_bound_f1',     'Boundary F1    (mean)',     '%', 100),
        ('mean_edit_score',   'Edit Score     (mean)',     '/100', 1),
        *[(f'mean_f1@{k:.2f}', f'F1@{k:.2f}       (mean)', '%', 100)
          for k in seg_thresholds],
        ('mean_ose',          'Over-seg. Err  (mean)',    '',   1),
    ]

    csv_rows = [['fold'] + [mk for mk, *_ in metric_keys]]
    for res in fold_results:
        csv_rows.append([res['fold']] + [res[mk] for mk, *_ in metric_keys])

    print(f"\n  {'Metric':<35}  {'Mean':>8}  {'Std':>8}  {'Min':>8}  {'Max':>8}")
    print("  " + "─" * 65)
    for mk, label, unit, scale in metric_keys:
        vals = [r[mk] * scale for r in fold_results]
        print(f"  {label:<35}  "
              f"{np.mean(vals):>7.2f}{unit}  "
              f"{np.std(vals):>7.2f}{unit}  "
              f"{np.min(vals):>7.2f}{unit}  "
              f"{np.max(vals):>7.2f}{unit}")

    print(f"{'═' * 65}\n")

    # ── save CSV ──────────────────────────────────────────────────────────────
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = os.path.join(RESULT_DIR, f"cv_summary_{ts}.csv")
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['fold'] + [mk for mk, *_ in metric_keys])
        for res in fold_results:
            writer.writerow([res['fold']]
                            + [f"{res[mk]:.6f}" for mk, *_ in metric_keys])
        # mean / std rows
        writer.writerow(['MEAN']
                        + [f"{np.mean([r[mk] for r in fold_results]):.6f}"
                           for mk, *_ in metric_keys])
        writer.writerow(['STD']
                        + [f"{np.std([r[mk] for r in fold_results]):.6f}"
                           for mk, *_ in metric_keys])
    print(f"  CSV saved → {csv_path}\n")

    return fold_results


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="5-Fold Cross-Validation for LSTM Surgical Phase Segmentation")
    p.add_argument('--folds',  type=int, default=N_FOLDS,    help="Number of CV folds (default 5)")
    p.add_argument('--epochs', type=int, default=NUM_EPOCHS, help="Training epochs per fold (default 800)")
    p.add_argument('--quick',  action='store_true',
                   help="Smoke-test: override epochs to 10 and folds to 3")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.quick:
        args.folds  = 3
        args.epochs = 10
        print("[quick mode] folds=3, epochs=10")

    run_cross_validation(n_folds=args.folds, num_epochs=args.epochs)
