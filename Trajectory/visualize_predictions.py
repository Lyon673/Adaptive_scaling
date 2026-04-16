"""
序列预测标签可视化工具
用于展示真实标签和预测标签的对比分析
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.colors import ListedColormap
import seaborn as sns
from torch.utils.data import DataLoader
from LSTM_seg_train import SequenceLabelingLSTM, SequenceLabelingLSTM_CRF, collate_fn
from load_data import load_test_state, load_test_label, load_specific_test_state, load_specific_test_label, get_test_demo_id_list
import os
from config import resample, without_quat
from torchcrf import CRF

plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

class TestDataset:
    def __init__(self):
        demo_id_list = np.arange(148)
        demo_id_list = np.delete(demo_id_list, [80, 81, 92, 109, 112, 117, 122, 144, 145])
        test_demo_id_list = get_test_demo_id_list(demo_id_list)
        self.demo_ids = list(test_demo_id_list)   # 保存每条序列对应的原始 demo_id
        demonstrations_state = load_test_state(without_quat=without_quat, resample=resample, demo_id_list=demo_id_list)
        demonstrations_label = load_test_label(resample=resample, demo_id_list=demo_id_list)
        
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
        idx = np.min([idx,len(self.samples)-1])
        return self.samples[idx]


def load_model(filepath, device='cpu'):
    """加载训练好的模型"""
    checkpoint = torch.load(filepath, map_location=device)
    model_config = checkpoint['model_config']
    
    if any(key.startswith('crf.') for key in checkpoint['model_state_dict'].keys()):
        model = SequenceLabelingLSTM_CRF(
            input_size=model_config['input_size'],
            hidden_size=model_config['hidden_size'],
            num_layers=model_config['num_layers'],
            num_classes=model_config['num_classes']
        )
    else:
        model = SequenceLabelingLSTM(
            input_size=model_config['input_size'],
            hidden_size=model_config['hidden_size'],
            num_layers=model_config['num_layers'],
            num_classes=model_config['num_classes']
        )
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    return model


def predict_sequence(model, sequence, device):
    """预测单个序列的标签"""
    model.eval()
    
    with torch.no_grad():
        input_tensor = sequence.unsqueeze(0).to(device)
        lengths = [sequence.shape[0]]
        
        if hasattr(model, 'crf'):
            # CRF模型使用decode方法
            preds_list = model.decode(input_tensor, lengths)
            predictions = preds_list[0]
            # 确保是numpy数组
            if isinstance(predictions, list):
                predictions = np.array(predictions)
        else:
            # 标准LSTM模型
            outputs = model(input_tensor, lengths)
            predictions = torch.argmax(outputs, dim=2).squeeze().cpu().numpy()
    
    return predictions


def visualize_sequence_predictions(model, test_dataset, device, num_sequences=7, save_path=None):
    """
    可视化序列预测结果
    """
    class_colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7', '#DDA15E', '#BA68C8', '#A9A2B5']
    class_names = ['Class 0', 'Class 1', 'Class 2', 'Class 3', 'Class 4', 'Class 5', 'Class 6', 'Unlabeled']
    
    fig, axes = plt.subplots(num_sequences, 1, figsize=(15, 3*num_sequences))
    if num_sequences == 1:
        axes = [axes]
    
    for seq_idx in range(num_sequences):
        ax = axes[seq_idx]
        sequence, true_labels = test_dataset[seq_idx]
        predicted_labels = predict_sequence(model, sequence, device)
        if isinstance(predicted_labels, list): predicted_labels = np.array(predicted_labels)
        
        min_len = min(len(true_labels), len(predicted_labels))
        true_labels = true_labels[:min_len].numpy()
        predicted_labels = predicted_labels[:min_len]
        
        time_steps = np.arange(min_len)
        valid_mask = true_labels != -1
        valid_time_steps = time_steps[valid_mask]
        valid_true_labels = true_labels[valid_mask]
        valid_predicted_labels = predicted_labels[valid_mask]
        
        for i in range(len(class_names) - 1): 
            mask = true_labels == i
            if np.any(mask):
                ax.scatter(time_steps[mask], np.ones(np.sum(mask)) * 1.1, 
                          c=class_colors[i], label=f'True {class_names[i]}', 
                          s=50, alpha=0.8, marker='o')
        
        unlabeled_mask = true_labels == -1
        if np.any(unlabeled_mask):
            ax.scatter(time_steps[unlabeled_mask], np.ones(np.sum(unlabeled_mask)) * 1.1, 
                      c=class_colors[-1], label='True Unlabeled (-1)', 
                      s=50, alpha=0.5, marker='x')
        
        for i in range(len(class_names) - 1): 
            mask = valid_predicted_labels == i
            if np.any(mask):
                ax.scatter(valid_time_steps[mask], np.ones(np.sum(mask)) * 0.9, 
                          c=class_colors[i], label=f'Pred {class_names[i]}', 
                          s=30, alpha=0.6, marker='^')
        
        demo_id = test_dataset.demo_ids[seq_idx] if hasattr(test_dataset, 'demo_ids') else seq_idx
        if len(valid_true_labels) > 0:
            accuracy = np.mean(valid_true_labels == valid_predicted_labels)
            valid_count = len(valid_true_labels)
            total_count = len(true_labels)
            title = f'Seq {seq_idx + 1}  [demo_id={demo_id}]  Accuracy: {accuracy:.3f}  (Valid: {valid_count}/{total_count})'
        else:
            accuracy = 0.0
            title = f'Seq {seq_idx + 1}  [demo_id={demo_id}]  No valid labels'
        
        ax.set_ylim(0.5, 1.5)
        ax.set_xlabel('Time Steps')
        ax.set_title(title)
        ax.set_yticks([0.9, 1.1])
        ax.set_yticklabels(['Predicted', 'True'])
        ax.grid(True, alpha=0.3)
        
        if seq_idx == 0:
            ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"可视化结果已保存到: {save_path}")
    plt.show()
    plt.close('all') 


def visualize_confusion_matrix(model, test_dataset, device, save_path=None):
    from sklearn.metrics import confusion_matrix
    all_true, all_pred = [], []
    for i in range(len(test_dataset)):
        sequence, true_labels = test_dataset[i]
        predicted_labels = predict_sequence(model, sequence, device)
        if isinstance(predicted_labels, list): predicted_labels = np.array(predicted_labels)
        
        min_len = min(len(true_labels), len(predicted_labels))
        true_labels_np = true_labels[:min_len].numpy()
        predicted_labels_np = predicted_labels[:min_len]
        
        valid_mask = true_labels_np != -1
        valid_true = true_labels_np[valid_mask]
        valid_pred = predicted_labels_np[valid_mask]
        
        all_true.extend(valid_true)
        all_pred.extend(valid_pred)
    
    cm = confusion_matrix(all_true, all_pred)
    plt.figure(figsize=(12, 10))
    num_classes = cm.shape[0]
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=[f'Class {i}' for i in range(num_classes)],
                yticklabels=[f'Class {i}' for i in range(num_classes)])
    plt.title('Confusion Matrix - Sequence Labeling')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"混淆矩阵已保存到: {save_path}")
    plt.show()

def _draw_prob_subplot(ax, probs, true_labels_np, seq_idx,
                       coarse_classes, fine_classes, CLASS_NAMES, STAGE_COLORS,
                       show_xlabel=False, show_legend_bands=False, show_legend_lines=False,
                       demo_id=None):
    seq_len    = probs.shape[0]
    time_steps = np.arange(seq_len)

    coarse_prob = probs[:, coarse_classes].sum(axis=1)
    fine_prob   = probs[:, fine_classes  ].sum(axis=1)

    prev_label, span_start, spans = true_labels_np[0], 0, []
    for t in range(1, seq_len):
        if true_labels_np[t] != prev_label:
            spans.append((span_start, t - 1, prev_label))
            span_start, prev_label = t, true_labels_np[t]
    spans.append((span_start, seq_len - 1, prev_label))

    drawn_labels = set()
    band_handles, band_labels = [], []
    for (s, e, lbl) in spans:
        color    = STAGE_COLORS.get(int(lbl), '#DDDDDD')
        lbl_name = CLASS_NAMES[int(lbl)] if 0 <= lbl < len(CLASS_NAMES) else 'Unlabeled'
        patch = ax.axvspan(s - 0.5, e + 0.5, alpha=0.45, color=color, linewidth=0)
        if lbl not in drawn_labels:
            band_handles.append(patch)
            band_labels.append(lbl_name.replace('\n', ' '))
            drawn_labels.add(lbl)

    l_coarse, = ax.plot(time_steps, coarse_prob,
                        color='#1A6FA8', linewidth=2.0,
                        label='P(Coarse) – 0,2,4,6', zorder=3)
    l_fine,   = ax.plot(time_steps, fine_prob,
                        color='#C0392B', linewidth=2.0, linestyle='--',
                        label='P(Fine) – 1,3,5', zorder=3)

    ax.set_xlim(-0.5, seq_len - 0.5)
    ax.set_ylim(-0.03, 1.08)
    ax.set_ylabel('Probability', fontsize=10)
    demo_label = f'  [demo_id={demo_id}]' if demo_id is not None else ''
    ax.set_title(f'Sequence {seq_idx}{demo_label}', fontsize=11, fontweight='bold')
    ax.grid(True, alpha=0.25, axis='y')

    if show_xlabel: ax.set_xlabel('Time Step', fontsize=10)
    if show_legend_lines:
        ax.legend([l_coarse, l_fine],
                  ['P(Coarse) – 0,2,4,6', 'P(Fine) – 1,3,5'],
                  loc='upper right', fontsize=9, framealpha=0.85)
    return band_handles, band_labels

# =========================================================================
# 【新增】：仅用于替换软标签计算的前向 CRF 状态机
# =========================================================================
class RealTimeCRFProbEstimator:
    def __init__(self, crf_module):
        self.transitions = crf_module.transitions.detach().cpu()
        if hasattr(crf_module, 'start_transitions'):
            self.start_transitions = crf_module.start_transitions.detach().cpu()
        else:
            self.start_transitions = torch.zeros(self.transitions.size(0))
        self.num_classes = self.transitions.size(0)
        self.alpha = None

    def reset(self):
        self.alpha = None

    def step(self, emission):
        emission = emission.detach().cpu()
        if self.alpha is None:
            self.alpha = self.start_transitions + emission
        else:
            score = self.alpha.unsqueeze(1) + self.transitions + emission.unsqueeze(0)
            self.alpha = torch.logsumexp(score, dim=0)
        return torch.softmax(self.alpha, dim=0).numpy()

def get_realtime_probs_for_sequence(emissions, crf_module):
    estimator = RealTimeCRFProbEstimator(crf_module)
    estimator.reset()
    probs = []
    for t in range(emissions.shape[0]):
        p = estimator.step(emissions[t])
        probs.append(p)
    return np.array(probs)
# =========================================================================

def visualize_predicted_class_probabilities(model, test_dataset, device,
                                            seq_idx=0, save_path=None):
    CLASS_NAMES = [
        'P0 Right\nMove', 'P1 Pick\nNeedle', 'P2 Right\nMove2',
        'P3 Pass\nNeedle', 'P4 Left\nMove',  'P5 Left\nPick',
        'P6 Pull\nThread',
    ]
    STAGE_COLORS = {
        0: '#A8D5E2', 1: '#F4A9A8', 2: '#A8C5DA',
        3: '#F7C5A0', 4: '#B5D5C5', 5: '#D5A8D4',
        6: '#C5D5A8', -1: '#E0E0E0',
    }
    coarse_classes = [0, 2, 4, 6]
    fine_classes   = [1, 3, 5]

    if isinstance(seq_idx, int): seq_indices = [seq_idx]
    else: seq_indices = list(seq_idx)
    n = len(seq_indices)

    model.eval()
    if hasattr(model, 'crf') and not hasattr(model, 'spatial_attention'):
        print("Warning: CRF model does not provide per-step probabilities. Skipping.")
        return

    all_probs, all_true = [], []
    with torch.no_grad():
        for idx in seq_indices:
            sequence, true_labels = test_dataset[idx]
            sequence = sequence.to(device)
            # 兼容：如果模型包含 decode 以外的 forward 推理
            if hasattr(model, 'spatial_attention'):
                logits = model(sequence.unsqueeze(0), [sequence.shape[0]])
            else:
                logits = model(sequence.unsqueeze(0), [sequence.shape[0]])
                
            logits_squeeze = logits.squeeze(0)
            
            # 【核心修改】：针对含有 CRF 的网络，替换为因果滤波概率
            if hasattr(model, 'crf'):
                probs = get_realtime_probs_for_sequence(logits_squeeze, model.crf)
            else:
                probs = torch.softmax(logits_squeeze, dim=1).cpu().numpy()
                
            all_probs.append(probs)
            all_true.append(true_labels.numpy())

    fig, axes = plt.subplots(n, 1, figsize=(16, 4 * n), squeeze=False)

    def _get_demo_id(dataset, idx):
        if hasattr(dataset, 'demo_ids') and idx < len(dataset.demo_ids):
            return dataset.demo_ids[idx]
        return None

    all_band_handles, all_band_labels = [], []
    for row, (idx, probs, true_labels_np) in enumerate(zip(seq_indices, all_probs, all_true)):
        ax = axes[row, 0]
        is_last = (row == n - 1)
        bh, bl = _draw_prob_subplot(
            ax, probs, true_labels_np, idx,
            coarse_classes, fine_classes, CLASS_NAMES, STAGE_COLORS,
            show_xlabel=is_last, show_legend_bands=False, show_legend_lines=(row == 0),
            demo_id=_get_demo_id(test_dataset, idx),
        )
        for h, l in zip(bh, bl):
            if l not in all_band_labels:
                all_band_handles.append(h)
                all_band_labels.append(l)

    fig.legend(all_band_handles, all_band_labels,
               loc='lower center', ncol=min(7, len(all_band_labels)),
               fontsize=9, title='True Stage',
               bbox_to_anchor=(0.5, 0.0), framealpha=0.9)

    demo_ids_str = ([str(_get_demo_id(test_dataset, i)) for i in seq_indices])
    title_suffix = (f'Seq {seq_indices[0]} [demo_id={demo_ids_str[0]}]' if n == 1
                    else f'demo_ids={demo_ids_str}')
    fig.suptitle(f'Coarse vs Fine Operation Probabilities – {title_suffix}',
                 fontsize=13, fontweight='bold')

    plt.tight_layout(rect=[0, 0.06, 1, 0.97])
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Probability visualization saved to {save_path}")

    plt.show()
    plt.close(fig)


# =========================================================================
# ── 新增: Spatial Attention LSTM-CRF 架构与自适应加载器 ─────────────────────
# =========================================================================

class AdaptiveBimanualSpatialAttention(nn.Module):
    def __init__(self, input_dim=8, proj_dim=16, use_causal_conv=False):
        super(AdaptiveBimanualSpatialAttention, self).__init__()
        self.use_causal_conv = use_causal_conv
        
        if self.use_causal_conv:
            self.shared_causal_conv = nn.Conv1d(in_channels=input_dim, out_channels=proj_dim, kernel_size=3)
            self.layer_norm = nn.LayerNorm(16)
        else:
            self.shared_proj = nn.Linear(input_dim, proj_dim)
            
        self.dominance_mlp = nn.Sequential(
            nn.Linear(proj_dim * 2, proj_dim),
            nn.ReLU(),
            nn.Linear(proj_dim, 2)
        )

    def forward(self, x):
        X_L = x[:, :, :8]
        X_R = x[:, :, 8:]

        if self.use_causal_conv:
            X_L_t = X_L.transpose(1, 2)
            X_R_t = X_R.transpose(1, 2)
            X_L_pad = F.pad(X_L_t, (2, 0))
            X_R_pad = F.pad(X_R_t, (2, 0))
            H_L = F.relu(self.shared_causal_conv(X_L_pad)).transpose(1, 2)
            H_R = F.relu(self.shared_causal_conv(X_R_pad)).transpose(1, 2)
        else:
            H_L = F.relu(self.shared_proj(X_L))
            H_R = F.relu(self.shared_proj(X_R))

        H_cat = torch.cat([H_L, H_R], dim=-1)
        alpha_logits = self.dominance_mlp(H_cat)
        alphas = F.softmax(alpha_logits, dim=-1)

        alpha_L = alphas[:, :, 0].unsqueeze(-1)
        alpha_R = alphas[:, :, 1].unsqueeze(-1)

        tilde_X_L = alpha_L * X_L
        tilde_X_R = alpha_R * X_R
        tilde_X = torch.cat([tilde_X_L, tilde_X_R], dim=-1)
        
        if self.use_causal_conv:
            out_X = self.layer_norm(x + tilde_X)
        else:
            out_X = tilde_X
            
        return out_X, alphas

class AdaptiveSequenceLabelingLSTM_CRF(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes, proj_dim=16, use_causal_conv=False):
        super(AdaptiveSequenceLabelingLSTM_CRF, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.proj_dim = proj_dim
        
        self.spatial_attention = AdaptiveBimanualSpatialAttention(
            input_dim=8, proj_dim=proj_dim, use_causal_conv=use_causal_conv
        )
        
        self.lstm = nn.LSTM(
            input_size=16, 
            hidden_size=hidden_size, 
            num_layers=num_layers, 
            batch_first=True, 
            bidirectional=False,
            dropout=0.2 if num_layers > 1 else 0
        )
        
        self.fc = nn.Linear(hidden_size, num_classes)
        self.crf = CRF(num_classes, batch_first=True)

        with torch.no_grad():
            self.crf.transitions.fill_(0) 
            for i in range(num_classes):
                self.crf.transitions[i, i] = 2.0
                if i < num_classes - 1:
                    self.crf.transitions[i, i+1] = 2.0

    def forward(self, x, lengths, return_alphas=False):
        tilde_X, alphas = self.spatial_attention(x)
        
        packed_input = pack_padded_sequence(tilde_X, lengths, batch_first=True, enforce_sorted=False)
        packed_output, (h_n, c_n) = self.lstm(packed_input)
        lstm_out, _ = pad_packed_sequence(packed_output, batch_first=True)
        
        emissions = self.fc(lstm_out)
        if return_alphas:
            return emissions, alphas
        return emissions

    def decode(self, x, lengths, return_alphas=False):
        max_len = x.size(1)
        mask = torch.arange(max_len, device=x.device)[None, :] < torch.tensor(lengths, device=x.device)[:, None]
        
        if return_alphas:
            emissions, alphas = self.forward(x, lengths, return_alphas=True)
            best_path = self.crf.decode(emissions, mask=mask)
            return best_path, alphas
        else:
            emissions = self.forward(x, lengths)
            best_path = self.crf.decode(emissions, mask=mask)
            return best_path


def load_spatial_attn_lstmcrf_model(filepath, device='cpu'):
    """动态嗅探参数，支持因果卷积/全连接层加载的加载器"""
    checkpoint = torch.load(filepath, map_location=device)
    cfg = checkpoint.get('model_config', {})
    state_dict = checkpoint['model_state_dict']

    use_causal_conv = "spatial_attention.shared_causal_conv.weight" in state_dict.keys()
    arch_name = "Causal-Conv (New)" if use_causal_conv else "Linear (Old)"
    print(f"Loading Adaptive-Spatial-Attention-LSTM-CRF... [Architecture: {arch_name}]")
    
    model = AdaptiveSequenceLabelingLSTM_CRF(
        input_size=cfg.get('input_size', 16),
        hidden_size=cfg.get('hidden_size', 256),
        num_layers=cfg.get('num_layers', 3),
        num_classes=cfg.get('num_classes', 7),
        proj_dim=cfg.get('proj_dim', 16),
        use_causal_conv=use_causal_conv
    )

    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model


def visualize_bimanual_alphas(model, test_dataset, device, seq_idx, save_path=None):
    """
    针对 Spatial Attn LSTM-CRF，绘制左右臂特征主导权的 Alpha 曲线图。
    支持多序列并排展示。
    """
    STAGE_COLORS = {
        0: '#A8D5E2', 1: '#F4A9A8', 2: '#A8C5DA',
        3: '#F7C5A0', 4: '#B5D5C5', 5: '#D5A8D4',
        6: '#C5D5A8', -1: '#F0F0F0'
    }
    CLASS_NAMES = [
        'P0 Right Move', 'P1 Pick Needle', 'P2 Right Move2', 
        'P3 Pass Needle', 'P4 Left Move', 'P5 Left Pick', 'P6 Pull Thread'
    ]
    
    if isinstance(seq_idx, int): seq_indices = [seq_idx]
    else: seq_indices = list(seq_idx)
    
    n = len(seq_indices)
    fig, axes = plt.subplots(n, 1, figsize=(14, 4.5 * n), squeeze=False)
    
    with torch.no_grad():
        for row, idx in enumerate(seq_indices):
            ax = axes[row, 0]
            sequence, true_labels = test_dataset[idx]
            true_np = true_labels.numpy()
            T = sequence.shape[0]
            
            # 推理并截获 Attention 分配参数
            x = sequence.unsqueeze(0).to(device)
            best_path, alphas = model.decode(x, [T], return_alphas=True)
            
            alpha_L = alphas[0, :, 0].cpu().numpy()
            alpha_R = alphas[0, :, 1].cpu().numpy()
            t_steps = np.arange(T)
            
            # 绘制背景真实标签色带
            prev_label, span_start = true_np[0], 0
            for t in range(1, T):
                if true_np[t] != prev_label:
                    color = STAGE_COLORS.get(int(prev_label), '#F0F0F0')
                    ax.axvspan(span_start, t, color=color, alpha=0.4, lw=0)
                    span_start, prev_label = t, true_np[t]
            ax.axvspan(span_start, T, color=STAGE_COLORS.get(int(prev_label), '#F0F0F0'), alpha=0.4, lw=0)

            # 绘制 Alpha 分配曲线
            ax.plot(t_steps, alpha_L, color='#2980b9', lw=2.0, label=r'Left Arm Dominance ($\alpha_L$)')
            ax.plot(t_steps, alpha_R, color='#c0392b', lw=2.0, linestyle='--', label=r'Right Arm Dominance ($\alpha_R$)')
            ax.axhline(0.5, color='gray', linestyle=':', lw=1.5, alpha=0.7)

            ax.set_xlim(0, T)
            ax.set_ylim(0, 1.05)
            demo_id = test_dataset.demo_ids[idx] if hasattr(test_dataset, 'demo_ids') else idx
            ax.set_title(f"Bimanual Spatial Cross-Attention Dominance (Seq {idx} [demo_id={demo_id}])", fontsize=12, fontweight='bold', pad=12)
            
            if row == n - 1:
                ax.set_xlabel("Time Step (Frames)", fontsize=11)
            ax.set_ylabel("Attention Weight (0~1)", fontsize=11)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)

            if row == 0:
                line_legends = ax.legend(loc='upper right', framealpha=0.9)
                ax.add_artist(line_legends)

    # 全局图例
    bg_patches = [patches.Patch(color=STAGE_COLORS[i], label=CLASS_NAMES[i]) for i in range(7)]
    fig.legend(handles=bg_patches, loc='lower center', bbox_to_anchor=(0.5, 0.0), ncol=7, frameon=False, title="True Surgical Phases")

    plt.tight_layout(rect=[0, 0.05, 1, 1])
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"注意力分配曲线已保存至: {save_path}")
    plt.show()
    plt.close(fig)


def spatial_attn_lstmcrf_main():
    """Spatial Attn LSTM-CRF 可视化主函数"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")

    dir_path = os.path.dirname(__file__)
    # 请确保您有这个模型文件
    model_path = os.path.join(dir_path, "LSTM_model", "spatial_attn_lstmcrf_sequence_model2.pth")
    if not os.path.exists(model_path):
        print(f"模型文件不存在: {model_path}")
        return

    print("加载 Spatial Attn LSTM-CRF 模型...")
    model = load_spatial_attn_lstmcrf_model(model_path, device)
    print("模型加载成功")

    test_dataset = TestDataset()
    print(f"测试数据集大小: {len(test_dataset)}")

    save_dir = os.path.join(dir_path, "SpatialAttn_LSTMCRF_visualization_results")  
    os.makedirs(save_dir, exist_ok=True)

    all_seq_indices = list(range(min(20, len(test_dataset))))

    # 1. Attention 专属：绘制左右臂主导权 Alphas 曲线
    print("\n生成双臂空间注意力 Alpha 分布曲线...")
    visualize_bimanual_alphas(
        model, test_dataset, device,
        seq_idx=all_seq_indices,
        save_path=os.path.join(save_dir, "bimanual_attention_alphas.png")
    )

    # 2. 复用：粗大/精细操作概率可视化 (支持解析 CRF forward 发射的 Logits)
    print("\n生成粗大/精细操作概率可视化...")
    visualize_predicted_class_probabilities(
        model, test_dataset, device,
        seq_idx=all_seq_indices,
        save_path=os.path.join(save_dir, "coarse_fine_probabilities.png")
    )
    
    # 3. 复用：可视化序列预测对比
    print("\n生成序列预测对比图...")
    visualize_sequence_predictions(
        model, test_dataset, device, 
        num_sequences=min(6, len(test_dataset)), 
        save_path=os.path.join(save_dir, "sequence_predictions.png")
    )
    
    # 4. 复用：绘制混淆矩阵
    print("\n生成混淆矩阵...")
    visualize_confusion_matrix(
        model, test_dataset, device,
        save_path=os.path.join(save_dir, "confusion_matrix.png")
    )

    print(f"\n所有 Spatial Attn LSTM-CRF 可视化完成！结果保存在 {save_dir}")


if __name__ == "__main__":

    # lstm_main()

    # tecno_main()

    # attention_tecno_main()
    
    spatial_attn_lstmcrf_main()