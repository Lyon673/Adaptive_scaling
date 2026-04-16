"""
Attn_TeCNO_CRF_test.py
=======================
用于测试和评估融合架构 (Bimanual Spatial Cross-Attention + MS-TCN + CRF)
生成分类报告、混淆矩阵、总体分数卡以及分段评估指标 (Segmental Metrics)。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import LinearSegmentedColormap
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from tqdm import tqdm
import os
from typing import Optional
from torchcrf import CRF

# 导入您环境中的基础工具
from load_data import load_test_state, load_test_label, load_specific_test_state, load_specific_test_label
from config import resample, without_quat
from segmentation_metrics import SegmentationEvaluator

# 如果原来文件里有 collate_fn 也可以直接用，这里为了独立运行提供一份
def collate_fn(batch):
    sequences, labels = zip(*batch)
    lengths = [len(seq) for seq in sequences]
    padded_sequences = pad_sequence(sequences, batch_first=True, padding_value=0)
    padded_labels = pad_sequence(labels, batch_first=True, padding_value=-1) 
    return padded_sequences, padded_labels, lengths

BATCH_SIZE = 16
NUM_CLASSES = 7

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
_ACCENT  = '#5B8DB8'   
_ACCENT2 = '#8E7DB5'   
_GREEN   = '#4A9E7F'   
_YELLOW  = '#C49A3C'   
_RED     = '#B85C5C'   
_TEXT    = '#2D3748'   
_SUBTEXT = '#718096'   
_GRID    = '#E8ECF0'   
_BORDER  = '#D1D9E0'   

def _metric_color(v: float) -> str:
    if v >= 0.80: return _GREEN
    if v >= 0.60: return _YELLOW
    return _RED

# =========================================================================
# --- 1. 学术级评估报告可视化组件 ---
# =========================================================================

def visualize_report(all_labels, all_preds, title: str = "Model Evaluation Report", save_path: Optional[str] = None) -> None:
    target_names = [CLASS_NAMES[i] if i < len(CLASS_NAMES) else f'Class {i}' for i in range(NUM_CLASSES)]
    short_names = [n.replace('\n', ' ') for n in target_names]

    report_dict = classification_report(all_labels, all_preds, labels=list(range(NUM_CLASSES)), target_names=target_names, digits=4, output_dict=True, zero_division=0)
    accuracy = accuracy_score(all_labels, all_preds)
    cm       = confusion_matrix(all_labels, all_preds, labels=list(range(NUM_CLASSES)))
    cm_norm  = cm.astype(float) / cm.sum(axis=1, keepdims=True).clip(min=1)

    cmap_cm = LinearSegmentedColormap.from_list('cm_soft', ['#EEF2F7', '#A8C4DC', '#5B8DB8'], N=256)

    fig = plt.figure(figsize=(26, 18), facecolor=_BG)
    fig.suptitle(title, fontsize=24, fontweight='bold', color=_TEXT, y=0.975)

    outer = gridspec.GridSpec(2, 2, figure=fig, left=0.05, right=0.97, top=0.92, bottom=0.05, wspace=0.32, hspace=0.46)

    # Panel A: grouped bar chart
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
    ax_bar.spines['top'].set_visible(False); ax_bar.spines['right'].set_visible(False)
    ax_bar.spines['left'].set_color(_BORDER); ax_bar.spines['bottom'].set_color(_BORDER)
    ax_bar.yaxis.grid(True, color=_GRID, linewidth=0.8, zorder=0)
    ax_bar.set_axisbelow(True)
    ax_bar.legend(loc='upper right', framealpha=0.9, facecolor='white', edgecolor=_BORDER, labelcolor=_TEXT, fontsize=9, frameon=True)

    # Panel B: styled table
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
            cell.set_facecolor(_ACCENT); cell.set_text_props(color='white', fontweight='bold')
        elif r > n_class_rows:
            cell.set_facecolor('#EDF1F6'); cell.set_text_props(color=_ACCENT, fontweight='bold')
        else:
            cell.set_facecolor('white' if r % 2 == 1 else _PANEL); cell.set_text_props(color=_TEXT)
            if c in (1, 2, 3):
                try: v = float(rows[r - 1][c]); cell.set_text_props(color=_metric_color(v), fontweight='bold')
                except Exception: pass

    # Panel C: confusion matrix
    ax_cm = fig.add_subplot(outer[1, 0])
    ax_cm.set_facecolor(_PANEL)
    ax_cm.set_title('Confusion Matrix  (row-normalised)', color=_TEXT, fontsize=14, pad=12, fontweight='bold')

    im = ax_cm.imshow(cm_norm, cmap=cmap_cm, vmin=0, vmax=1, aspect='auto')
    cbar = plt.colorbar(im, ax=ax_cm, fraction=0.046, pad=0.04)
    cbar.ax.yaxis.set_tick_params(colors=_SUBTEXT, labelsize=9)

    ax_cm.set_xticks(range(NUM_CLASSES)); ax_cm.set_yticks(range(NUM_CLASSES))
    ax_cm.set_xticklabels(short_names, rotation=40, ha='right', fontsize=10, color=_TEXT)
    ax_cm.set_yticklabels(short_names, fontsize=10, color=_TEXT)
    ax_cm.set_xlabel('Predicted', color=_SUBTEXT, fontsize=10)
    ax_cm.set_ylabel('True', color=_SUBTEXT, fontsize=10)
    for spine in ax_cm.spines.values(): spine.set_edgecolor(_BORDER)

    for i in range(NUM_CLASSES):
        for j in range(NUM_CLASSES):
            v_norm, v_raw = cm_norm[i, j], cm[i, j]
            txt_col = 'white' if v_norm > 0.55 else _TEXT
            ax_cm.text(j, i, f'{v_norm:.2f}\n({v_raw})', ha='center', va='center', fontsize=8, color=txt_col, fontweight='bold')

    # Panel D: overall scorecard
    ax_sc = fig.add_subplot(outer[1, 1])
    ax_sc.set_facecolor(_PANEL)
    ax_sc.set_title('Overall Scorecard', color=_TEXT, fontsize=14, pad=12, fontweight='bold')
    ax_sc.axis('off')

    macro, weight = report_dict.get('macro avg', {}), report_dict.get('weighted avg', {})
    scorecard = [
        ('Accuracy', accuracy, _ACCENT), ('Macro Precision', macro.get('precision', 0), _ACCENT2),
        ('Macro Recall', macro.get('recall', 0), _ACCENT2), ('Macro F1', macro.get('f1-score', 0), _GREEN),
        ('Weighted F1', weight.get('f1-score', 0), _YELLOW),
    ]

    bar_h, gap, y_start = 0.090, 0.058, 0.87
    for i, (label, value, color) in enumerate(scorecard):
        y = y_start - i * (bar_h + gap)
        ax_sc.add_patch(plt.Rectangle((0.0, y - 0.005), 1.0, bar_h + 0.010, transform=ax_sc.transAxes, color=_GRID, zorder=1))
        ax_sc.add_patch(plt.Rectangle((0.0, y - 0.005), value, bar_h + 0.010, transform=ax_sc.transAxes, color=color, alpha=0.28, zorder=2))
        ax_sc.add_patch(plt.Rectangle((0.0, y - 0.005), 0.006, bar_h + 0.010, transform=ax_sc.transAxes, color=color, zorder=3))
        ax_sc.add_patch(plt.Rectangle((value, y - 0.005), 0.003, bar_h + 0.010, transform=ax_sc.transAxes, color=color, zorder=4))
        ax_sc.text(0.025, y + bar_h / 2, label, color=_SUBTEXT, fontsize=10.5, va='center', transform=ax_sc.transAxes, zorder=4)
        ax_sc.text(0.97, y + bar_h / 2, f'{value:.4f}', color=color, fontsize=13, fontweight='bold', va='center', ha='right', transform=ax_sc.transAxes, fontfamily='monospace', zorder=4)

    ax_sc.text(0.5, 0.03, f'Total frames evaluated: {len(all_labels):,}', color=_SUBTEXT, fontsize=9, ha='center', va='bottom', transform=ax_sc.transAxes)

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor=_BG)
    print(f"\n[Evaluation dashboard saved → {save_path}]")
    plt.close(fig)


# =========================================================================
# --- 2. 模型定义结构 (保障测试脚本独立性) ---
# =========================================================================

class BimanualSpatialAttention(nn.Module):
    def __init__(self, input_dim=8, proj_dim=16):
        super(BimanualSpatialAttention, self).__init__()
        self.shared_proj = nn.Linear(input_dim, proj_dim)
        self.dominance_mlp = nn.Sequential(nn.Linear(proj_dim * 2, proj_dim), nn.ReLU(), nn.Linear(proj_dim, 2))

    def forward(self, x):
        X_L, X_R = x[:, :, :8], x[:, :, 8:]
        H_L, H_R = F.relu(self.shared_proj(X_L)), F.relu(self.shared_proj(X_R))
        alpha_logits = self.dominance_mlp(torch.cat([H_L, H_R], dim=-1))
        alphas = F.softmax(alpha_logits, dim=-1)
        tilde_X = torch.cat([alphas[:, :, 0].unsqueeze(-1) * X_L, alphas[:, :, 1].unsqueeze(-1) * X_R], dim=-1)
        return tilde_X, alphas

class CausalDilatedConv1d(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size, dilation):
        super().__init__()
        self.pad_len = dilation * (kernel_size - 1)
        self.conv = nn.Conv1d(in_ch, out_ch, kernel_size, dilation=dilation)
    def forward(self, x):
        return self.conv(F.pad(x, (self.pad_len, 0)))

class DilatedResidualLayer(nn.Module):
    def __init__(self, hidden_dim, kernel_size, dilation, dropout):
        super().__init__()
        self.dconv = CausalDilatedConv1d(hidden_dim, hidden_dim, kernel_size, dilation)
        self.conv1x1 = nn.Conv1d(hidden_dim, hidden_dim, 1)
        self.drop = nn.Dropout(dropout)
    def forward(self, x):
        z = self.drop(F.relu(self.dconv(x)))
        return x + self.conv1x1(z)

class TCNStage(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, num_layers, kernel_size, dropout):
        super().__init__()
        self.conv_in = nn.Conv1d(in_dim, hidden_dim, 1)
        self.layers = nn.ModuleList([
            DilatedResidualLayer(hidden_dim, kernel_size, dilation=2 ** i, dropout=dropout)
            for i in range(num_layers)
        ])
        self.conv_out = nn.Conv1d(hidden_dim, out_dim, 1)
    def forward(self, x):
        h = self.conv_in(x)
        for layer in self.layers: h = layer(h)
        return self.conv_out(h)

class SequenceLabelingAttnTeCNO_CRF(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes, num_stages, proj_dim, dropout):
        super(SequenceLabelingAttnTeCNO_CRF, self).__init__()
        self.num_stages = num_stages
        self.spatial_attention = BimanualSpatialAttention(input_dim=8, proj_dim=proj_dim)
        self.stages = nn.ModuleList()
        for s in range(num_stages):
            in_dim = input_size if s == 0 else num_classes
            self.stages.append(TCNStage(in_dim, hidden_size, num_classes, num_layers, kernel_size=3, dropout=dropout))
            
        self.crf = CRF(num_classes, batch_first=True)
        with torch.no_grad():
            self.crf.transitions.fill_(0) 
            for i in range(num_classes):
                self.crf.transitions[i, i] = 2.0
                if i < num_classes - 1: self.crf.transitions[i, i+1] = 2.0

    def forward_all_stages(self, x):
        tilde_X, alphas = self.spatial_attention(x)
        h = tilde_X.permute(0, 2, 1)
        stage_outputs = []
        for i, stage in enumerate(self.stages):
            h = stage(h)
            stage_outputs.append(h.permute(0, 2, 1))
            if i < self.num_stages - 1: h = F.softmax(h, dim=1) 
        return stage_outputs, alphas

    def forward(self, x, lengths=None, return_alphas=False):
        stage_outputs, alphas = self.forward_all_stages(x)
        final_emissions = stage_outputs[-1]
        return (final_emissions, alphas) if return_alphas else final_emissions

    def decode(self, x, lengths, return_alphas=False):
        max_len = x.size(1)
        mask = torch.arange(max_len, device=x.device)[None, :] < torch.tensor(lengths, device=x.device)[:, None]
        stage_outputs, alphas = self.forward_all_stages(x)
        final_emissions = stage_outputs[-1]
        best_path = self.crf.decode(final_emissions, mask=mask)
        return (best_path, alphas) if return_alphas else best_path

def load_attn_tecno_crf_model(filepath, device='cpu'):
    checkpoint = torch.load(filepath, map_location=device)
    cfg = checkpoint.get('model_config', {})
    state_dict = checkpoint['model_state_dict']

    print(f"Loading Attn-TeCNO-CRF model...")
    model = SequenceLabelingAttnTeCNO_CRF(
        input_size=cfg.get('input_size', 16),
        hidden_size=cfg.get('hidden_size', 64),
        num_layers=cfg.get('num_layers', 6),
        num_classes=cfg.get('num_classes', 7),
        num_stages=cfg.get('num_stages', 2),
        proj_dim=cfg.get('proj_dim', 16),
        dropout=cfg.get('dropout', 0.4)
    )

    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model


# =========================================================================
# --- 3. 评估与测试管线 ---
# =========================================================================

class TestDataset(Dataset):
    def __init__(self):
        demo_id_list = np.arange(148)
        demo_id_list = np.delete(demo_id_list, [80, 81, 92, 109, 112, 117, 122, 144, 145])

        demonstrations_state = load_test_state(without_quat=without_quat, resample=resample, demo_id_list=demo_id_list)
        demonstrations_label = load_test_label(resample=resample, demo_id_list=demo_id_list)

        try:
            from load_data import get_shuffled_demo_ids, ratio
            all_ids = get_shuffled_demo_ids(demo_id_list=demo_id_list)
            bound = round(ratio * len(all_ids))
            self.demo_ids = all_ids[bound:] 
        except Exception:
            self.demo_ids = list(range(len(demonstrations_state)))

        self.samples = []
        for state_seq, label_seq in zip(demonstrations_state, demonstrations_label):
            state_tensor = torch.tensor(state_seq, dtype=torch.float32)
            label_tensor = torch.tensor(label_seq, dtype=torch.long)
            if state_tensor.dim() == 1:
                state_tensor = state_tensor.unsqueeze(1)
            self.samples.append((state_tensor, label_tensor))

    def __len__(self): return len(self.samples)
    def __getitem__(self, idx): return self.samples[idx]


def evaluate_model_CRF(model, dataloader, device, save_path=None):
    model.eval()
    all_preds, all_labels = [], []

    with torch.no_grad():
        for sequences, labels, lengths in dataloader:
            sequences = sequences.to(device)
            labels    = labels.to(device)
            
            # 使用 Viterbi 解码
            predicted_paths = model.decode(sequences, lengths)
            
            for i in range(len(lengths)):
                true_len   = lengths[i]
                true_labels_seq = labels[i, :true_len].cpu().numpy()
                pred_labels_seq = np.array(predicted_paths[i])
                
                valid_mask = (true_labels_seq != -1)
                all_labels.extend(true_labels_seq[valid_mask])
                all_preds.extend(pred_labels_seq[valid_mask])

    if not all_labels:
        print("could not find any valid labels for evaluation")
        return

    target_names = [CLASS_NAMES[i] if i < len(CLASS_NAMES) else f'Class {i}' for i in range(NUM_CLASSES)]
    print("\n--- Attn-TeCNO-CRF Model Evaluation Report ---")
    print(classification_report(all_labels, all_preds, target_names=[n.replace('\n', ' ') for n in target_names], digits=4, zero_division=0))
    print(f"Overall Accuracy: {accuracy_score(all_labels, all_preds):.4f}")
    print("---------------------\n")

    if save_path is None:
        save_path = os.path.join(os.path.dirname(__file__), 'eval_results', 'eval_report_attn_tecno_crf.png')
    visualize_report(all_labels, all_preds, title="Attn-TeCNO-CRF Evaluation Report", save_path=save_path)


def evaluate_segmental_metrics(model, test_dataset, device, tau=15, seg_thresholds=(0.10, 0.25, 0.50), save_path=None):
    evaluator = SegmentationEvaluator()
    model.eval()

    seq_results = []
    demo_ids = (test_dataset.demo_ids if hasattr(test_dataset, 'demo_ids') else list(range(len(test_dataset))))

    with torch.no_grad():
        for idx in range(len(test_dataset)):
            sequence, true_labels = test_dataset[idx]
            seq_tensor = sequence.to(device)
            lengths    = [seq_tensor.shape[0]]

            paths = model.decode(seq_tensor.unsqueeze(0), lengths)
            pred_np = np.array(paths[0], dtype=int)
            true_np = true_labels.numpy()

            min_len = min(len(true_np), len(pred_np))
            true_np = true_np[:min_len]
            pred_np = pred_np[:min_len]

            valid_idx = np.where(true_np != -1)[0]
            if len(valid_idx) == 0: continue
            
            start_idx, end_idx = valid_idx[0], valid_idx[-1]
            true_np = true_np[start_idx:end_idx+1]
            pred_np = pred_np[start_idx:end_idx+1]

            if np.all(true_np == -1): continue

            result = evaluator.evaluate(true_np, pred_np, tau=tau, segmental_thresholds=seg_thresholds)
            seq_results.append((demo_ids[idx], result))

    if not seq_results: return

    k_list = list(seg_thresholds)
    header = (f"{'demo_id':>8}  {'Acc':>6}  {'BoundF1':>8}  {'EditSc':>7}  " + 
              "  ".join(f"F1@{k:.2f}" for k in k_list) + f"  {'OSE':>7}")
    sep = "─" * len(header)

    print(f"\n{'═' * len(header)}")
    print("  Segmental Metrics – Per Sequence (Attn-TeCNO-CRF)")
    print(f"{'═' * len(header)}")
    print(header); print(sep)

    accs, bfs, eds, oses = [], [], [], []
    f1s = {k: [] for k in k_list}

    for demo_id, r in seq_results:
        f1_str = "  ".join(f"{r.segmental_f1.f1_at_k[k]*100:>7.1f}" for k in k_list)
        print(f"{demo_id:>8}  {r.frame_accuracy*100:>5.1f}%  {r.boundary_f1.f1*100:>7.1f}%  {r.edit_score.score:>7.1f}  {f1_str}  {r.oversegmentation_err:>7.3f}")
        accs.append(r.frame_accuracy); bfs.append(r.boundary_f1.f1); eds.append(r.edit_score.score); oses.append(r.oversegmentation_err)
        for k in k_list: f1s[k].append(r.segmental_f1.f1_at_k[k])

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

    accs = [r.frame_accuracy for _, r in seq_results]; bfs = [r.boundary_f1.f1 for _, r in seq_results]
    eds = [r.edit_score.score for _, r in seq_results]; oses = [r.oversegmentation_err for _, r in seq_results]

    def _agg_row(fn, label):
        row = [label, f"{fn(accs)*100:.1f}", f"{fn(bfs)*100:.1f}", f"{fn(eds):.1f}"]
        for k in k_list:
            vals = [r.segmental_f1.f1_at_k[k] for _, r in seq_results]
            row.append(f"{fn(vals)*100:.1f}")
        row.append(f"{fn(oses):.3f}")
        return row

    rows.append(_agg_row(np.mean, "MEAN")); rows.append(_agg_row(np.std,  "STD"))

    fig, ax = plt.subplots(figsize=(max(10, len(col_labels) * 1.4), max(4, len(rows) * 0.4 + 1.5)))
    ax.axis('off')
    tbl = ax.table(cellText=rows, colLabels=col_labels, loc='center', cellLoc='center')
    tbl.auto_set_font_size(False); tbl.set_fontsize(9); tbl.scale(1, 1.4)

    for j in range(len(col_labels)):
        tbl[0, j].set_facecolor('#2C3E50'); tbl[0, j].set_text_props(color='white', fontweight='bold')
        tbl[len(rows) - 1, j].set_facecolor('#D6EAF8')
        tbl[len(rows),     j].set_facecolor('#EBF5FB')

    plt.title("Segmental Evaluation Metrics (Attn-TeCNO-CRF)", fontsize=13, fontweight='bold', pad=12)
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()

if __name__ == "__main__":
    test_dataset = TestDataset()
    test_loader  = DataLoader(dataset=test_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn, drop_last=False)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    dir_path = os.path.dirname(__file__)
    model_save_path = os.path.join(dir_path, "TeCNO_model", "attn_tecno_crf_sequence_model.pth")
    
    if not os.path.exists(model_save_path):
        print(f"找不到模型文件: {model_save_path}。请先运行 Attn_TeCNO_CRF_train.py")
    else:
        model = load_attn_tecno_crf_model(model_save_path, device)

        # 1. 常规评估 (基于 Viterbi 解码的全局报告与混淆矩阵)
        evaluate_model_CRF(model, test_loader, device, 
                           save_path=os.path.join(dir_path, 'eval_results', 'eval_report_attn_tecno_crf.png'))
            
        # 2. 分段级别的高级指标评估 (Edit Score, OSE, F1@k)
        evaluate_segmental_metrics(
            model, test_dataset, device, tau=15, seg_thresholds=(0.10, 0.25, 0.50),
            save_path=os.path.join(dir_path, 'eval_results', 'segmental_metrics_attn_tecno_crf.png'),
        )