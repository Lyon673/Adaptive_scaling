import torch
from torch.utils.data import Dataset, DataLoader
# 从 TCN 的训练文件中导入必要的组件
from TCN_seg_train import SequenceLabelingTCN, NUM_CLASSES, BATCH_SIZE, collate_fn
from load_data import load_test_state, load_test_label,load_specific_test_state,load_specific_test_label
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from tqdm import tqdm
import os
from typing import Optional
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import LinearSegmentedColormap
from config import resample, without_quat
from segmentation_metrics import SegmentationEvaluator

# Surgical action class names
CLASS_NAMES = [
    '(P0)\nRight\nMove',
    '(P1)\nPick\nNeedle',
    '(P2)\nRight\nMove2',
    '(P3)\nPass\nNeedle',
    '(P4)\nLeft\nMove',
    '(P5)\nLeft\nPick',
    '(P6)\nPull\nThread',
]

# ── colour palette (soft light theme) ─────────────────────────────────────────
_BG      = '#FAFAFA'
_PANEL   = '#F4F6F9'
_ACCENT  = '#5B8DB8'   # muted steel-blue
_ACCENT2 = '#8E7DB5'   # muted lavender
_GREEN   = '#4A9E7F'   # muted teal-green
_YELLOW  = '#C49A3C'   # muted ochre
_RED     = '#B85C5C'   # muted rose-red
_TEXT    = '#2D3748'   # dark slate
_SUBTEXT = '#718096'   # medium slate
_GRID    = '#E8ECF0'   # very light grey
_BORDER  = '#D1D9E0'   # light blue-grey

def _metric_color(v: float) -> str:
    if v >= 0.80: return _GREEN
    if v >= 0.60: return _YELLOW
    return _RED

def visualize_report(all_labels, all_preds, title: str = "Model Evaluation Report", save_path: Optional[str] = None) -> None:
    target_names = [CLASS_NAMES[i] if i < len(CLASS_NAMES) else f'Class {i}' for i in range(NUM_CLASSES)]
    short_names = [n.replace('\n', ' ') for n in target_names]

    report_dict = classification_report(
        all_labels, all_preds, labels=list(range(NUM_CLASSES)),
        target_names=target_names, digits=4, output_dict=True, zero_division=0,
    )
    accuracy = accuracy_score(all_labels, all_preds)
    cm       = confusion_matrix(all_labels, all_preds, labels=list(range(NUM_CLASSES)))
    cm_norm  = cm.astype(float) / cm.sum(axis=1, keepdims=True).clip(min=1)

    cmap_cm = LinearSegmentedColormap.from_list('cm_soft', ['#EEF2F7', '#A8C4DC', '#5B8DB8'], N=256)

    fig = plt.figure(figsize=(26, 18), facecolor=_BG)
    fig.suptitle(title, fontsize=24, fontweight='bold', color=_TEXT, y=0.975)

    outer = gridspec.GridSpec(2, 2, figure=fig, left=0.05, right=0.97, top=0.92, bottom=0.05, wspace=0.32, hspace=0.46)

    # ── Panel A: grouped bar chart ──────────────────────────────────────────────
    ax_bar = fig.add_subplot(outer[0, 0])
    ax_bar.set_facecolor(_PANEL)
    ax_bar.set_title('Per-Class Metrics', color=_TEXT, fontsize=14, pad=12, fontweight='bold')

    metrics_cfg = [('precision', 'Precision', _ACCENT), ('recall', 'Recall', _ACCENT2), ('f1-score', 'F1-Score', _GREEN)]
    x = np.arange(NUM_CLASSES)
    w = 0.25
    for k, (metric, label, color) in enumerate(metrics_cfg):
        vals = [report_dict[tn][metric] if tn in report_dict else 0.0 for tn in target_names]
        bars = ax_bar.bar(x + (k - 1) * w, vals, w, label=label, color=color, alpha=0.82, linewidth=0, zorder=3)
        for bar, v in zip(bars, vals):
            if v > 0.05:
                ax_bar.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.013, f'{v:.2f}', ha='center', va='bottom', fontsize=7, color=_TEXT, fontweight='bold')

    ax_bar.set_xticks(x)
    ax_bar.set_xticklabels(short_names, rotation=30, ha='right', fontsize=10, color=_TEXT)
    ax_bar.set_ylim(0, 1.18)
    ax_bar.set_ylabel('Score', color=_SUBTEXT, fontsize=10)
    ax_bar.yaxis.set_tick_params(colors=_SUBTEXT, labelsize=9)
    ax_bar.xaxis.set_tick_params(colors=_SUBTEXT)
    ax_bar.spines['top'].set_visible(False)
    ax_bar.spines['right'].set_visible(False)
    ax_bar.spines['left'].set_color(_BORDER)
    ax_bar.spines['bottom'].set_color(_BORDER)
    ax_bar.yaxis.grid(True, color=_GRID, linewidth=0.8, zorder=0)
    ax_bar.set_axisbelow(True)
    ax_bar.legend(loc='upper right', framealpha=0.9, facecolor='white', edgecolor=_BORDER, labelcolor=_TEXT, fontsize=9, frameon=True)

    # ── Panel B: styled table ──────────────────────────────────────────────────
    ax_tbl = fig.add_subplot(outer[0, 1])
    ax_tbl.set_facecolor(_PANEL)
    ax_tbl.set_title('Classification Report', color=_TEXT, fontsize=14, pad=12, fontweight='bold')
    ax_tbl.axis('off')

    col_labels = ['Class', 'Precision', 'Recall', 'F1-Score', 'Support']
    rows = []
    for tn in target_names:
        if tn in report_dict:
            r = report_dict[tn]
            rows.append([tn.replace('\n', ' '), f"{r['precision']:.4f}", f"{r['recall']:.4f}", f"{r['f1-score']:.4f}", f"{int(r['support'])}"])
    for key, label in [('macro avg', 'Macro Avg'), ('weighted avg', 'Weighted Avg')]:
        if key in report_dict:
            r = report_dict[key]
            rows.append([label, f"{r['precision']:.4f}", f"{r['recall']:.4f}", f"{r['f1-score']:.4f}", f"{int(r['support'])}"])

    tbl = ax_tbl.table(cellText=rows, colLabels=col_labels, loc='center', cellLoc='center')
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9.5)
    tbl.scale(1, 1.7)

    n_class_rows = len(target_names)
    for (r, c), cell in tbl.get_celld().items():
        cell.set_edgecolor(_BORDER)
        if r == 0:
            cell.set_facecolor(_ACCENT)
            cell.set_text_props(color='white', fontweight='bold')
        elif r > n_class_rows:
            cell.set_facecolor('#EDF1F6')
            cell.set_text_props(color=_ACCENT, fontweight='bold')
        else:
            cell.set_facecolor('white' if r % 2 == 1 else _PANEL)
            cell.set_text_props(color=_TEXT)
            if c in (1, 2, 3):
                try:
                    v = float(rows[r - 1][c])
                    cell.set_text_props(color=_metric_color(v), fontweight='bold')
                except Exception:
                    pass

    # ── Panel C: confusion matrix ──────────────────────────────────────────────
    ax_cm = fig.add_subplot(outer[1, 0])
    ax_cm.set_facecolor(_PANEL)
    ax_cm.set_title('Confusion Matrix  (row-normalised)', color=_TEXT, fontsize=14, pad=12, fontweight='bold')

    im = ax_cm.imshow(cm_norm, cmap=cmap_cm, vmin=0, vmax=1, aspect='auto')
    cbar = plt.colorbar(im, ax=ax_cm, fraction=0.046, pad=0.04)
    cbar.ax.yaxis.set_tick_params(colors=_SUBTEXT, labelsize=9)

    ax_cm.set_xticks(range(NUM_CLASSES))
    ax_cm.set_yticks(range(NUM_CLASSES))
    ax_cm.set_xticklabels(short_names, rotation=40, ha='right', fontsize=10, color=_TEXT)
    ax_cm.set_yticklabels(short_names, fontsize=10, color=_TEXT)
    ax_cm.set_xlabel('Predicted', color=_SUBTEXT, fontsize=10)
    ax_cm.set_ylabel('True', color=_SUBTEXT, fontsize=10)
    for spine in ax_cm.spines.values():
        spine.set_edgecolor(_BORDER)

    for i in range(NUM_CLASSES):
        for j in range(NUM_CLASSES):
            v_norm = cm_norm[i, j]
            v_raw  = cm[i, j]
            txt_col = 'white' if v_norm > 0.55 else _TEXT
            ax_cm.text(j, i, f'{v_norm:.2f}\n({v_raw})', ha='center', va='center', fontsize=8, color=txt_col, fontweight='bold')

    # ── Panel D: overall scorecard ─────────────────────────────────────────────
    ax_sc = fig.add_subplot(outer[1, 1])
    ax_sc.set_facecolor(_PANEL)
    ax_sc.set_title('Overall Scorecard', color=_TEXT, fontsize=14, pad=12, fontweight='bold')
    ax_sc.axis('off')

    macro  = report_dict.get('macro avg', {})
    weight = report_dict.get('weighted avg', {})

    scorecard = [
        ('Accuracy',         accuracy,                       _ACCENT),
        ('Macro Precision',  macro.get('precision',  0),     _ACCENT2),
        ('Macro Recall',     macro.get('recall',     0),     _ACCENT2),
        ('Macro F1',         macro.get('f1-score',   0),     _GREEN),
        ('Weighted F1',      weight.get('f1-score',  0),     _YELLOW),
    ]

    bar_h   = 0.090
    gap     = 0.058
    y_start = 0.87
    for i, (label, value, color) in enumerate(scorecard):
        y = y_start - i * (bar_h + gap)
        ax_sc.add_patch(plt.Rectangle((0.0, y - 0.005), 1.0, bar_h + 0.010, transform=ax_sc.transAxes, color=_GRID, zorder=1))
        ax_sc.add_patch(plt.Rectangle((0.0, y - 0.005), value, bar_h + 0.010, transform=ax_sc.transAxes, color=color, alpha=0.28, zorder=2))
        ax_sc.add_patch(plt.Rectangle((0.0, y - 0.005), 0.006, bar_h + 0.010, transform=ax_sc.transAxes, color=color, zorder=3))
        ax_sc.text(0.025, y + bar_h / 2, label, color=_SUBTEXT, fontsize=10.5, va='center', transform=ax_sc.transAxes, zorder=4)
        ax_sc.text(0.97, y + bar_h / 2, f'{value:.4f}', color=color, fontsize=13, fontweight='bold', va='center', ha='right', transform=ax_sc.transAxes, fontfamily='monospace', zorder=4)

    ax_sc.text(0.5, 0.03, f'Total frames evaluated: {len(all_labels):,}', color=_SUBTEXT, fontsize=9, ha='center', va='bottom', transform=ax_sc.transAxes)

    if save_path is None:
        save_path = os.path.join(os.path.dirname(__file__), 'eval_results', 'eval_report_tcn.png')
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor=_BG)
    print(f"\n[Evaluation dashboard saved → {save_path}]")
    plt.close(fig)


# --- 模型加载功能 ---
def load_model(filepath, device='cpu'):
    """从文件加载训练好的 TCN 模型"""
    checkpoint = torch.load(filepath, map_location=device)
    model_config = checkpoint['model_config']

    print("Loading TCN Baseline model...")
    model = SequenceLabelingTCN(
        input_size=model_config.get('input_size', 16),
        num_channels=model_config.get('num_channels', [64, 128, 256, 256, 128]),
        num_classes=model_config.get('num_classes', 7),
        kernel_size=model_config.get('kernel_size', 3),
        dropout=model_config.get('dropout', 0.2) # 从 config 读取，默认 0.2
    )

    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()

    print(f"model loaded from {filepath}")
    return model


class TestDataset(Dataset):
    def __init__(self):
        # demo_id_list = np.arange(148)
        # demo_id_list = np.delete(demo_id_list, [80, 81, 92, 109, 112, 117, 122, 144, 145])

        # demonstrations_state = load_test_state(without_quat=without_quat, resample=resample, demo_id_list=demo_id_list)
        # demonstrations_label = load_test_label(resample=resample, demo_id_list=demo_id_list)

        demo_id_list = [35, 34, 65, 138, 4, 3, 71, 90, 131, 54, 140, 43, 80, 81, 92, 109, 112, 117, 122, 144, 145]
        demonstrations_state = load_specific_test_state(shuffle=False, without_quat=without_quat, resample=resample, demo_id_list=demo_id_list)
        demonstrations_label = load_specific_test_label(demo_id_list=demo_id_list)

        self.samples = []
        for state_seq, label_seq in zip(demonstrations_state, demonstrations_label):
            state_tensor = torch.tensor(state_seq, dtype=torch.float32)
            label_tensor = torch.tensor(label_seq, dtype=torch.long)
            if state_tensor.dim() == 1:
                state_tensor = state_tensor.unsqueeze(1)
            self.samples.append((state_tensor, label_tensor))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


# --- 评估函数 ---
def evaluate_model(model, dataloader, device, save_path=None):
    """在测试集上评估 TCN 模型并生成可视化报告。"""
    model.eval()
    all_preds, all_labels = [], []

    with torch.no_grad():
        for sequences, labels, lengths in dataloader:
            sequences = sequences.to(device)
            labels    = labels.to(device)
            
            # TCN 返回的是 (Batch, Time, Classes) 的 logits
            outputs   = model(sequences, lengths)
            preds     = torch.argmax(outputs, dim=-1)
            
            for i in range(len(lengths)):
                true_len = lengths[i]
                true_labels_seq = labels[i, :true_len].cpu().numpy()
                pred_labels_seq = preds[i, :true_len].cpu().numpy()
                
                # 清洗序列内部的非法 padding (-1)
                valid_mask = (true_labels_seq != -1)
                all_labels.extend(true_labels_seq[valid_mask])
                all_preds.extend(pred_labels_seq[valid_mask])

    if not all_labels:
        print("could not find any valid labels for evaluation")
        return

    target_names = [CLASS_NAMES[i] if i < len(CLASS_NAMES) else f'Class {i}' for i in range(NUM_CLASSES)]
    print("\n--- TCN model evaluation report ---")
    print(classification_report(all_labels, all_preds, target_names=[n.replace('\n', ' ') for n in target_names], digits=4, zero_division=0))
    print(f"Overall Accuracy: {accuracy_score(all_labels, all_preds):.4f}")
    print("---------------------\n")

    if save_path is None:
        save_path = os.path.join(os.path.dirname(__file__), 'eval_results', 'eval_report_tcn.png')
    visualize_report(all_labels, all_preds, title="TCN Evaluation Report", save_path=save_path)


def evaluate_segmental_metrics(
    model, test_dataset, device,
    tau: int = 15, seg_thresholds: tuple = (0.10, 0.25, 0.50), save_path: Optional[str] = None,
) -> None:
    evaluator = SegmentationEvaluator()
    model.eval()

    seq_results = []
    demo_ids    = (test_dataset.demo_ids if hasattr(test_dataset, 'demo_ids') else list(range(len(test_dataset))))

    with torch.no_grad():
        for idx in range(len(test_dataset)):
            sequence, true_labels = test_dataset[idx]
            seq_tensor = sequence.to(device)
            lengths    = [seq_tensor.shape[0]]

            # TCN 推理获取 logits 并取 argmax
            logits = model(seq_tensor.unsqueeze(0), lengths)
            pred_np = torch.argmax(logits, dim=-1).squeeze(0).cpu().numpy()
            true_np = true_labels.numpy()

            min_len = min(len(true_np), len(pred_np))
            true_np = true_np[:min_len]
            pred_np = pred_np[:min_len]

            valid_idx = np.where(true_np != -1)[0]
            if len(valid_idx) == 0:
                continue
            start_idx, end_idx = valid_idx[0], valid_idx[-1]
            
            true_np = true_np[start_idx:end_idx+1]
            pred_np = pred_np[start_idx:end_idx+1]

            if np.all(true_np == -1):
                continue

            result = evaluator.evaluate(true_np, pred_np, tau=tau, segmental_thresholds=seg_thresholds)
            seq_results.append((demo_ids[idx], result))

    if not seq_results:
        print("[evaluate_segmental_metrics] No valid sequences found.")
        return

    k_list = list(seg_thresholds)
    header = (f"{'demo_id':>8}  {'Acc':>6}  {'BoundF1':>8}  {'EditSc':>7}  " + 
              "  ".join(f"F1@{k:.2f}" for k in k_list) + f"  {'OSE':>7}")
    sep = "─" * len(header)

    print(f"\n{'═' * len(header)}")
    print("  Segmental Metrics – Per Sequence (TCN)")
    print(f"{'═' * len(header)}")
    print(header)
    print(sep)

    accs, bfs, eds, oses = [], [], [], []
    f1s = {k: [] for k in k_list}

    for demo_id, r in seq_results:
        f1_str = "  ".join(f"{r.segmental_f1.f1_at_k[k]*100:>7.1f}" for k in k_list)
        print(f"{demo_id:>8}  {r.frame_accuracy*100:>5.1f}%  {r.boundary_f1.f1*100:>7.1f}%  {r.edit_score.score:>7.1f}  {f1_str}  {r.oversegmentation_err:>7.3f}")
        accs.append(r.frame_accuracy)
        bfs.append(r.boundary_f1.f1)
        eds.append(r.edit_score.score)
        oses.append(r.oversegmentation_err)
        for k in k_list:
            f1s[k].append(r.segmental_f1.f1_at_k[k])

    print(sep)
    mean_f1_str = "  ".join(f"{np.mean(f1s[k])*100:>7.1f}" for k in k_list)
    print(f"{'MEAN':>8}  {np.mean(accs)*100:>5.1f}%  {np.mean(bfs)*100:>7.1f}%  {np.mean(eds):>7.1f}  {mean_f1_str}  {np.mean(oses):>7.3f}")
    std_f1_str = "  ".join(f"{np.std(f1s[k])*100:>7.1f}" for k in k_list)
    print(f"{'STD':>8}  {np.std(accs)*100:>5.1f}%  {np.std(bfs)*100:>7.1f}%  {np.std(eds):>7.1f}  {std_f1_str}  {np.std(oses):>7.3f}")
    print(f"{'═' * len(header)}\n")

    if save_path:
        _save_segmental_table(seq_results, k_list, save_path)
        print(f"Segmental metrics table saved to {save_path}")


def _save_segmental_table(seq_results: list, k_list: list, save_path: str) -> None:
    col_labels = (["demo_id", "Acc(%)", "BoundF1(%)", "EditScore"] + [f"F1@{k:.2f}(%)" for k in k_list] + ["OSE"])
    rows = []
    for demo_id, r in seq_results:
        row = [str(demo_id), f"{r.frame_accuracy*100:.1f}", f"{r.boundary_f1.f1*100:.1f}", f"{r.edit_score.score:.1f}"]
        for k in k_list: row.append(f"{r.segmental_f1.f1_at_k[k]*100:.1f}")
        row.append(f"{r.oversegmentation_err:.3f}")
        rows.append(row)

    accs = [r.frame_accuracy         for _, r in seq_results]
    bfs  = [r.boundary_f1.f1         for _, r in seq_results]
    eds  = [r.edit_score.score       for _, r in seq_results]
    oses = [r.oversegmentation_err   for _, r in seq_results]

    def _agg_row(fn, label):
        row = [label, f"{fn(accs)*100:.1f}", f"{fn(bfs)*100:.1f}", f"{fn(eds):.1f}"]
        for k in k_list:
            vals = [r.segmental_f1.f1_at_k[k] for _, r in seq_results]
            row.append(f"{fn(vals)*100:.1f}")
        row.append(f"{fn(oses):.3f}")
        return row

    rows.append(_agg_row(np.mean, "MEAN"))
    rows.append(_agg_row(np.std,  "STD"))

    n_rows = len(rows)
    n_cols = len(col_labels)
    fig, ax = plt.subplots(figsize=(max(10, n_cols * 1.4), max(4, n_rows * 0.4 + 1.5)))
    ax.axis('off')

    tbl = ax.table(cellText=rows, colLabels=col_labels, loc='center', cellLoc='center')
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9)
    tbl.scale(1, 1.4)

    for j in range(n_cols):
        tbl[0, j].set_facecolor('#2C3E50')
        tbl[0, j].set_text_props(color='white', fontweight='bold')
    for j in range(n_cols):
        tbl[n_rows - 1, j].set_facecolor('#D6EAF8')
        tbl[n_rows,     j].set_facecolor('#EBF5FB')

    plt.title("Segmental Evaluation Metrics (TCN)", fontsize=13, fontweight='bold', pad=12)
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def get_prediction_for_current_step(model, history_data, device):
    """根据历史数据获取当前时刻的预测标签（实时推理核心逻辑）。"""
    seq_len      = len(history_data)
    input_tensor = history_data.unsqueeze(0).to(device)
    lengths      = [seq_len]
    
    # TCN 直接通过 argmax 解析 logits
    logits = model(input_tensor, lengths)
    current_label = torch.argmax(logits[0, -1, :]).item()
    return current_label


def evaluate_model_real_time_simulation(model, test_dataset, device, save_path=None):
    """在测试集上模拟实时数据流进行评估，并生成可视化报告。"""
    print("\n--- start real-time simulation ---")
    model.eval()
    all_preds, all_labels = [], []
    
    with torch.no_grad():
        for state_seq, label_seq in tqdm(test_dataset, desc="real-time simulation"):
            for t in range(len(state_seq)):
                true_label = label_seq[t].item()
                if true_label == -1: continue
                if t < 30:
                    all_preds.append(true_label)
                    all_labels.append(true_label)
                    continue
                history_data    = state_seq[:t + 1]
                predicted_label = get_prediction_for_current_step(model, history_data, device)
                all_preds.append(predicted_label)
                all_labels.append(true_label)

    if not all_labels:
        print("could not find any valid labels for evaluation")
        return

    target_names = [CLASS_NAMES[i] if i < len(CLASS_NAMES) else f'Class {i}' for i in range(NUM_CLASSES)]
    print("\n--- real-time simulation report ---")
    print(classification_report(all_labels, all_preds, target_names=[n.replace('\n', ' ') for n in target_names], digits=4, zero_division=0))
    print(f"Overall Accuracy (Real-time Simulation): {accuracy_score(all_labels, all_preds):.4f}")
    print("---------------------\n")

    if save_path is None:
        save_path = os.path.join(os.path.dirname(__file__), 'eval_results', 'eval_report_tcn_realtime.png')
    visualize_report(all_labels, all_preds, title="TCN Real-Time Simulation Report", save_path=save_path)


if __name__ == "__main__":
    test_dataset = TestDataset()
    test_loader  = DataLoader(
        dataset=test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        collate_fn=collate_fn,
        drop_last=False,
    )

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    dir_path = os.path.dirname(__file__)
    
    # 默认读取最新训练出的 TCN 模型
    model_save_path = os.path.join(dir_path, "TCN_model", "tcn_sequence_model.pth")
    
    if not os.path.exists(model_save_path):
        print(f"找不到模型文件: {model_save_path}")
    else:
        model = load_model(model_save_path, device)

        # 1. 常规评估
        evaluate_model(model, test_loader, device)
            
        # 2. 分段级别的高级指标评估
        evaluate_segmental_metrics(
            model, test_dataset, device,
            tau=15,
            seg_thresholds=(0.10, 0.25, 0.50),
            save_path=os.path.join(dir_path, 'eval_results', 'segmental_metrics_tcn.png'),
        )
        
        # 3. 实时推理模拟测试 (如果需要可以取消注释)
        # evaluate_model_real_time_simulation(model, test_dataset, device)