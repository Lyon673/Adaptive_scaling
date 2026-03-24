"""
TeCNO_seg_test.py
=================
Evaluation script for TeCNO surgical phase recognition model.

Metrics:
  1. Frame-level: Accuracy, Precision, Recall, F1 per class
  2. Visualisation dashboard: grouped bar chart, classification report table,
     normalised confusion matrix, overall scorecard
  3. Segmental metrics: Boundary F1, Edit Score, F1@{0.10,0.25,0.50},
     Over-segmentation Error  (via SegmentationEvaluator)
  4. Real-time simulation: feed frames one-by-one and evaluate

Results saved to  Trajectory/TeCNO_eval_results/
"""

import torch
import numpy as np
import os
from typing import Optional
from torch.utils.data import Dataset, DataLoader

import matplotlib
matplotlib.use('Agg')

from TeCNO_seg_train import (
    TeCNO, collate_fn,
    NUM_CLASSES, BATCH_SIZE, INPUT_SIZE, HIDDEN_SIZE, NUM_LAYERS,
    NUM_STAGES, KERNEL_SIZE, DROPOUT,
)
from load_data import load_test_state, load_test_label
from sklearn.metrics import classification_report, accuracy_score
from config import resample, without_quat
from segmentation_metrics import SegmentationEvaluator

from LSTM_seg_test import (
    visualize_report,
    _save_segmental_table,
    CLASS_NAMES,
)

# ── Model loading ────────────────────────────────────────────────────────

def load_tecno_model(filepath, device='cpu'):
    """Load a TeCNO checkpoint and return the model in eval mode."""
    checkpoint = torch.load(filepath, map_location=device)
    cfg = checkpoint['model_config']

    model = TeCNO(
        cfg['input_size'],
        cfg['hidden_size'],
        cfg['num_layers'],
        cfg['num_classes'],
        num_stages=cfg.get('num_stages', NUM_STAGES),
        kernel_size=cfg.get('kernel_size', KERNEL_SIZE),
        dropout=cfg.get('dropout', DROPOUT),
    )
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()

    print(f"TeCNO model loaded from {filepath}")
    print(f"  config: {cfg}")
    if 'training_info' in checkpoint:
        print(f"  training: {checkpoint['training_info']}")
    return model


# ── Test dataset ─────────────────────────────────────────────────────────

class TestDataset(Dataset):
    def __init__(self):
        demo_id_list = np.delete(np.arange(148),
                                 [80, 81, 92, 109, 112, 117, 122, 144, 145])
        self.states = load_test_state(without_quat=without_quat,
                                      resample=resample,
                                      demo_id_list=demo_id_list)
        self.labels = load_test_label(resample=resample,
                                      demo_id_list=demo_id_list)
        self.samples = []
        for s, l in zip(self.states, self.labels):
            st = torch.tensor(s, dtype=torch.float32)
            lt = torch.tensor(l, dtype=torch.long)
            if st.dim() == 1:
                st = st.unsqueeze(1)
            self.samples.append((st, lt))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


# ── Frame-level evaluation ───────────────────────────────────────────────

def evaluate_model(model, dataloader, device, save_path=None):
    """Evaluate frame-level accuracy and generate a four-panel dashboard."""
    model.eval()
    all_preds, all_labels = [], []

    with torch.no_grad():
        for sequences, labels, lengths in dataloader:
            sequences = sequences.to(device)
            labels = labels.to(device)
            logits = model(sequences, lengths)
            preds = torch.argmax(logits, dim=2)

            mask = labels != -1
            all_preds.extend(torch.masked_select(preds, mask).cpu().numpy())
            all_labels.extend(torch.masked_select(labels, mask).cpu().numpy())

    if not all_labels:
        print("[TeCNO evaluate_model] No valid labels found.")
        return

    target_names = [
        CLASS_NAMES[i].replace('\n', ' ') if i < len(CLASS_NAMES)
        else f'Class {i}'
        for i in range(NUM_CLASSES)
    ]
    print("\n--- TeCNO Frame-Level Evaluation ---")
    print(classification_report(all_labels, all_preds,
                                target_names=target_names,
                                digits=4, zero_division=0))
    print(f"Overall Accuracy: {accuracy_score(all_labels, all_preds):.4f}")
    print("------------------------------------\n")

    if save_path is None:
        save_path = os.path.join(os.path.dirname(__file__),
                                 'TeCNO_eval_results', 'eval_report.png')
    visualize_report(all_labels, all_preds,
                     title="TeCNO Evaluation Report", save_path=save_path)


# ── Segmental metrics ────────────────────────────────────────────────────

def evaluate_segmental_metrics(
    model,
    test_dataset,
    device,
    tau: int = 15,
    seg_thresholds: tuple = (0.10, 0.25, 0.50),
    save_path: Optional[str] = None,
) -> None:
    """
    Per-sequence segmental evaluation:
      Boundary F1, Edit Score, F1@k, Over-segmentation Error.
    """
    evaluator = SegmentationEvaluator()
    model.eval()

    seq_results = []
    demo_ids = (test_dataset.demo_ids
                if hasattr(test_dataset, 'demo_ids')
                else list(range(len(test_dataset))))

    with torch.no_grad():
        for idx in range(len(test_dataset)):
            sequence, true_labels = test_dataset[idx]
            seq_tensor = sequence.to(device)
            lengths = [seq_tensor.shape[0]]

            logits = model(seq_tensor.unsqueeze(0), lengths)
            pred_np = torch.argmax(logits, dim=2).squeeze(0).cpu().numpy()
            true_np = true_labels.numpy()

            min_len = min(len(true_np), len(pred_np))
            true_np = true_np[:min_len]
            pred_np = pred_np[:min_len]

            if np.all(true_np == -1):
                continue

            result = evaluator.evaluate(true_np, pred_np,
                                        tau=tau,
                                        segmental_thresholds=seg_thresholds)
            seq_results.append((demo_ids[idx], result))

    if not seq_results:
        print("[TeCNO evaluate_segmental_metrics] No valid sequences.")
        return

    k_list = list(seg_thresholds)
    header = (f"{'demo_id':>8}  {'Acc':>6}  {'BoundF1':>8}  "
              f"{'EditSc':>7}  "
              + "  ".join(f"F1@{k:.2f}" for k in k_list)
              + f"  {'OSE':>7}")
    sep = "─" * len(header)

    print(f"\n{'═' * len(header)}")
    print("  TeCNO Segmental Metrics – Per Sequence")
    print(f"{'═' * len(header)}")
    print(header)
    print(sep)

    accs, bfs, eds, oses = [], [], [], []
    f1s = {k: [] for k in k_list}

    for demo_id, r in seq_results:
        f1_str = "  ".join(f"{r.segmental_f1.f1_at_k[k]*100:>7.1f}"
                           for k in k_list)
        print(f"{demo_id:>8}  "
              f"{r.frame_accuracy*100:>5.1f}%  "
              f"{r.boundary_f1.f1*100:>7.1f}%  "
              f"{r.edit_score.score:>7.1f}  "
              f"{f1_str}  "
              f"{r.oversegmentation_err:>7.3f}")
        accs.append(r.frame_accuracy)
        bfs.append(r.boundary_f1.f1)
        eds.append(r.edit_score.score)
        oses.append(r.oversegmentation_err)
        for k in k_list:
            f1s[k].append(r.segmental_f1.f1_at_k[k])

    print(sep)
    mean_f1_str = "  ".join(f"{np.mean(f1s[k])*100:>7.1f}" for k in k_list)
    print(f"{'MEAN':>8}  "
          f"{np.mean(accs)*100:>5.1f}%  "
          f"{np.mean(bfs)*100:>7.1f}%  "
          f"{np.mean(eds):>7.1f}  "
          f"{mean_f1_str}  "
          f"{np.mean(oses):>7.3f}")
    std_f1_str = "  ".join(f"{np.std(f1s[k])*100:>7.1f}" for k in k_list)
    print(f"{'STD':>8}  "
          f"{np.std(accs)*100:>5.1f}%  "
          f"{np.std(bfs)*100:>7.1f}%  "
          f"{np.std(eds):>7.1f}  "
          f"{std_f1_str}  "
          f"{np.std(oses):>7.3f}")
    print(f"{'═' * len(header)}\n")

    if save_path:
        _save_segmental_table(seq_results, k_list, save_path)
        print(f"Segmental metrics table saved to {save_path}")


# ── Real-time simulation ─────────────────────────────────────────────────

def evaluate_realtime_simulation(model, test_dataset, device,
                                 min_frames=30, save_path=None):
    """
    Simulate online inference: feed frames 1..t cumulatively
    and take the last-frame prediction at each step.
    """
    from tqdm import tqdm

    print("\n--- TeCNO Real-Time Simulation ---")
    model.eval()
    all_preds, all_labels = [], []

    with torch.no_grad():
        for state_seq, label_seq in tqdm(test_dataset, desc="real-time sim"):
            for t in range(len(state_seq)):
                true_label = label_seq[t].item()
                if true_label == -1:
                    continue
                if t < min_frames:
                    all_preds.append(true_label)
                    all_labels.append(true_label)
                    continue
                history = state_seq[:t + 1].unsqueeze(0).to(device)
                logits = model(history, [t + 1])
                pred = torch.argmax(logits[0, -1, :]).item()
                all_preds.append(pred)
                all_labels.append(true_label)

    if not all_labels:
        print("No valid labels for real-time evaluation.")
        return

    target_names = [
        CLASS_NAMES[i].replace('\n', ' ') if i < len(CLASS_NAMES)
        else f'Class {i}'
        for i in range(NUM_CLASSES)
    ]
    print(classification_report(all_labels, all_preds,
                                target_names=target_names,
                                digits=4, zero_division=0))
    print(f"Overall Accuracy (Real-time): "
          f"{accuracy_score(all_labels, all_preds):.4f}\n")

    if save_path is None:
        save_path = os.path.join(os.path.dirname(__file__),
                                 'TeCNO_eval_results',
                                 'eval_report_realtime.png')
    visualize_report(all_labels, all_preds,
                     title="TeCNO Real-Time Simulation Report",
                     save_path=save_path)


# ── Main ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    dir_path = os.path.dirname(__file__)
    result_dir = os.path.join(dir_path, 'TeCNO_eval_results')
    os.makedirs(result_dir, exist_ok=True)

    model_path = os.path.join(dir_path, 'TeCNO_model',
                              'tecno_sequence_model.pth')
    model = load_tecno_model(model_path, device)

    test_dataset = TestDataset()
    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        collate_fn=collate_fn,
        drop_last=False,
    )

    # 1. Frame-level evaluation + dashboard
    print("\n" + "=" * 60)
    print("  Frame-Level Evaluation")
    print("=" * 60)
    evaluate_model(model, test_loader, device,
                   save_path=os.path.join(result_dir, 'eval_report.png'))

    # 2. Segmental metrics
    print("\n" + "=" * 60)
    print("  Segmental Metrics")
    print("=" * 60)
    evaluate_segmental_metrics(
        model, test_dataset, device,
        tau=15,
        seg_thresholds=(0.10, 0.25, 0.50),
        save_path=os.path.join(result_dir, 'segmental_metrics.png'),
    )

    # 3. Real-time simulation (uncomment to run — slow)
    # evaluate_realtime_simulation(
    #     model, test_dataset, device,
    #     save_path=os.path.join(result_dir, 'eval_report_realtime.png'),
    # )
