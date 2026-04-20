import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import os
import sys
from sklearn.metrics import accuracy_score
from torchcrf import CRF

# 导入数据加载相关 (严格按照您指定 demo_id 的方式)
from load_data import load_specific_test_state, load_specific_test_label
from config import resample, without_quat

# 定义全局类名与对应的科研风格色彩
CLASS_NAMES = [
    'P0', 'P1', 'P2', 'P3', 'P4', 'P5', 'P6'
]
CLASS_COLORS = ['#FF6B6B', '#EDC58C', '#45B7D1', '#96CEB4', '#FFEAA7', '#DDA15E', '#B16DCF']

# =========================================================================
# 1. 自适应模型层 (Adaptive Architectures)
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


# 【新增】：无 CRF 版本的空间注意力 LSTM 架构
class AdaptiveSequenceLabelingLSTM_Attn(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes, proj_dim=16, use_causal_conv=False):
        super(AdaptiveSequenceLabelingLSTM_Attn, self).__init__()
        self.spatial_attention = AdaptiveBimanualSpatialAttention(
            input_dim=8, proj_dim=proj_dim, use_causal_conv=use_causal_conv
        )
        self.lstm = nn.LSTM(
            input_size=16, hidden_size=hidden_size, num_layers=num_layers, 
            batch_first=True, dropout=0.2 if num_layers > 1 else 0
        )
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x, lengths, return_alphas=False):
        tilde_X, alphas = self.spatial_attention(x)
        packed_input = pack_padded_sequence(tilde_X, lengths, batch_first=True, enforce_sorted=False)
        packed_output, (h_n, c_n) = self.lstm(packed_input)
        lstm_out, _ = pad_packed_sequence(packed_output, batch_first=True)
        logits = self.fc(lstm_out)
        if return_alphas:
            return logits, alphas
        return logits


class AdaptiveSequenceLabelingLSTM_CRF(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes, proj_dim=16, use_causal_conv=False):
        super(AdaptiveSequenceLabelingLSTM_CRF, self).__init__()
        self.spatial_attention = AdaptiveBimanualSpatialAttention(
            input_dim=8, proj_dim=proj_dim, use_causal_conv=use_causal_conv
        )
        self.lstm = nn.LSTM(
            input_size=16, hidden_size=hidden_size, num_layers=num_layers, 
            batch_first=True, dropout=0.2 if num_layers > 1 else 0
        )
        self.fc = nn.Linear(hidden_size, num_classes)
        self.crf = CRF(num_classes, batch_first=True)

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
            return self.crf.decode(emissions, mask=mask), alphas
        return self.crf.decode(self.forward(x, lengths), mask=mask)

# =========================================================================
# 2. 动态模型加载工厂 (Dynamic Model Loader)
# =========================================================================
def load_model_dynamically(filepath, device):
    """自动嗅探权重并实例化正确的模型结构"""
    ckpt = torch.load(filepath, map_location=device)
    cfg = ckpt.get('model_config', {})
    state_dict = ckpt['model_state_dict']
    
    is_tecno = 'model_type' in ckpt and 'TeCNO' in ckpt['model_type']
    has_spatial_attn = any(k.startswith('spatial_attention.') for k in state_dict.keys())
    has_crf = any(k.startswith('crf.') for k in state_dict.keys())
    has_base_tcn = any(k.startswith('tcn.') for k in state_dict.keys()) or 'num_channels' in cfg
    
    if is_tecno:
        try:
            from TeCNO_seg_train import TeCNO
        except ImportError:
            TeCNO = None
        if TeCNO is not None:
            model = TeCNO(
                input_size=cfg.get('input_size', 16),
                hidden_size=cfg.get('hidden_size', 64),
                num_layers=cfg.get('num_layers', 6), # default to 6 for standard TeCNO
                num_classes=cfg.get('num_classes', 7),
                num_stages=cfg.get('num_stages', 2)
            )
    elif has_spatial_attn and has_crf:
        # 带注意力和 CRF
        use_causal_conv = "spatial_attention.shared_causal_conv.weight" in state_dict.keys()
        proj_dim = cfg.get('proj_dim', cfg.get('embed_dim', 16))
        model = AdaptiveSequenceLabelingLSTM_CRF(
            cfg.get('input_size', 16), cfg.get('hidden_size', 256), cfg.get('num_layers', 3), cfg.get('num_classes', 7),
            proj_dim=proj_dim, use_causal_conv=use_causal_conv
        )
    elif has_spatial_attn and not has_crf:
        # 【新增】：带注意力但无 CRF (消融实验专用)
        use_causal_conv = "spatial_attention.shared_causal_conv.weight" in state_dict.keys()
        proj_dim = cfg.get('proj_dim', cfg.get('embed_dim', 16))
        model = AdaptiveSequenceLabelingLSTM_Attn(
            cfg.get('input_size', 16), cfg.get('hidden_size', 256), cfg.get('num_layers', 3), cfg.get('num_classes', 7),
            proj_dim=proj_dim, use_causal_conv=use_causal_conv
        )
    elif has_crf:
        # 基础 LSTM-CRF
        from LSTM_seg_train import SequenceLabelingLSTM_CRF
        model = SequenceLabelingLSTM_CRF(
            cfg.get('input_size', 16), cfg.get('hidden_size', 256), cfg.get('num_layers', 3), cfg.get('num_classes', 7)
        )
    elif has_base_tcn:
        # 基础 TCN
        from TCN_seg_train import SequenceLabelingTCN
        num_channels = cfg.get('num_channels', [64, 64, 64, 64, 128])
        model = SequenceLabelingTCN(
            input_size=cfg.get('input_size', 16), num_channels=num_channels, num_classes=cfg.get('num_classes', 7)
        )
    else:
        # 基础 LSTM
        from LSTM_seg_train import SequenceLabelingLSTM
        model = SequenceLabelingLSTM(
            cfg.get('input_size', 16), cfg.get('hidden_size', 256), cfg.get('num_layers', 3), cfg.get('num_classes', 7)
        )

    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model

# =========================================================================
# 3. 推理辅助函数 (Inference Wrapper)
# =========================================================================
def predict_sequence(model, sequence, device):
    """自适应执行前向传播并返回一维 Numpy 数组预测结果"""
    with torch.no_grad():
        x = sequence.unsqueeze(0).to(device)
        lengths = [sequence.shape[0]]
        
        # 区分带 CRF 和不带 CRF 的模型推理方式
        if hasattr(model, 'crf') or hasattr(model, 'decode'):
            out = model.decode(x, lengths)
            preds = out[0][0] if isinstance(out, tuple) else out[0]
            preds = np.array(preds, dtype=int)
        else:
            outputs = model(x, lengths)
            logits = outputs[0] if isinstance(outputs, tuple) else outputs
            preds = torch.argmax(logits, dim=-1).squeeze(0).cpu().numpy()
            
    return np.array(preds, dtype=int)

def get_continuous_segments(labels):
    """将一维标签数组转换为连续段: [(start, end, class_id), ...]"""
    segments = []
    if len(labels) == 0: return segments
    
    start = 0
    current_label = labels[0]
    for i in range(1, len(labels)):
        if labels[i] != current_label:
            segments.append((start, i, current_label))
            start = i
            current_label = labels[i]
            
    segments.append((start, len(labels), current_label))
    return segments

# =========================================================================
# 4. 主函数与可视化绘图 (Main Visualizer)
# =========================================================================
def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[{device}] 初始化预测管线...")

    # ----- A. 按照您要求的指定 demo_id 的方式加载数据 -----
    target_demo_id = 200
    
    print(f"正在读取指定的 Demo {target_demo_id} ...")
    try:
        states = load_specific_test_state(shuffle=False, without_quat=without_quat, resample=resample, demo_id_list=[target_demo_id])
        labels = load_specific_test_label(demo_id_list=[target_demo_id])
        
        if len(states) == 0 or len(labels) == 0:
            raise ValueError(f"无法读取 Demo {target_demo_id}，返回为空！")
            
        state_tensor = torch.tensor(states[0], dtype=torch.float32)
        true_labels = np.array(labels[0], dtype=int)
    except Exception as e:
        print(f"❌ 数据加载失败: {e}")
        return

    # ----- B. 配置待测模型路径 -----
    dir_path = os.path.dirname(__file__)
    models_config = [
        ("LSTM Baseline",         os.path.join(dir_path, "LSTM_model", "lstm_sequence_model.pth")),
        ("TCN Baseline",          os.path.join(dir_path, "TCN_model",  "tcn_sequence_model.pth")),
        ("Multi-stage TCN",       os.path.join(dir_path, "TeCNO_model",  "tecno_sequence_model.pth")),
        ("BiWeight-LSTM",         os.path.join(dir_path, "LSTM_model", "spatial_attn_lstmnocrf_sequence_model.pth")),
        ("LSTM-CRF",              os.path.join(dir_path, "LSTM_model", "spatial_attn_lstmcrf_sequence_model2.pth")),
        ("BiWeight-LSTM-CRF",     os.path.join(dir_path, "LSTM_model", "spatial_attn_lstmcrf_sequence_model1.pth"))
    ]

    # 用于绘图的存储容器
    plot_data = []
    plot_data.append(("Ground Truth", true_labels))

    # ----- C. 循环推理所有模型 -----
    valid_mask = (true_labels != -1)  # 找到真实标签中有效的部分
    
    for model_name, path in models_config:
        if not os.path.exists(path):
            print(f"⚠️ 跳过 {model_name}：未找到权重 ({path})")
            continue
            
        print(f"✓ 推理: {model_name}...")
        try:
            model = load_model_dynamically(path, device)
            preds = predict_sequence(model, state_tensor, device)
            
            # 计算准确率时严格过滤 -1
            valid_true = true_labels[valid_mask]
            valid_pred = preds[valid_mask]
            acc = accuracy_score(valid_true, valid_pred)
            
            display_name = f"{model_name}\n(Acc: {acc*100:.1f}%)"
            
            # 剥离预测结果，强行将绘图用的数组对应 padding 位置刷为 -1
            plot_preds = np.copy(preds)
            plot_preds[~valid_mask] = -1 
            
            plot_data.append((display_name, plot_preds))
        except Exception as e:
            print(f"❌ {model_name} 推理出错: {e}")

    # ----- D. 绘制横向对比色带图 (Gantt Chart Style) -----
    fig, ax = plt.subplots(figsize=(16, 2.0 + 1.2 * len(plot_data)))
    
    y_positions = np.arange(len(plot_data))
    y_labels = [item[0] for item in plot_data]
    
    for y_idx, (name, seq_preds) in enumerate(plot_data):
        segments = get_continuous_segments(seq_preds)
        
        for start, end, class_id in segments:
            if class_id == -1 or class_id >= len(CLASS_COLORS):
                color = '#E0E0E0'
            else:
                color = CLASS_COLORS[class_id]
                
            ax.barh(y_idx, end - start, left=start, height=0.5, 
                    color=color, edgecolor='none', align='center')
            
            if start > 0:
                ax.axvline(start, ymin=(len(plot_data) - y_idx - 0.75)/len(plot_data), 
                           ymax=(len(plot_data) - y_idx - 0.25)/len(plot_data), 
                           color='white', linewidth=0.6, alpha=0.5)

    ax.set_yticks(y_positions)
    ax.set_yticklabels(y_labels, fontsize=14, weight='bold')
    ax.invert_yaxis() 
    
    ax.set_xlabel('Frame', fontsize=12)
    ax.set_xlim(0, len(true_labels))
    ax.grid(axis='x', linestyle='--', alpha=0.3)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)

    # ----- E. 添加通用图例 -----
    legend_patches = [patches.Patch(color=CLASS_COLORS[i], label=CLASS_NAMES[i]) for i in range(7)]
    legend_patches.append(patches.Patch(color='#E0E0E0', label='Invalid Phase'))
    
    ax.legend(handles=legend_patches, loc='lower center', bbox_to_anchor=(0.5, -0.15),
              ncol=8, fontsize=12, frameon=False, title="Surgical Phases", title_fontsize=12)

    plt.title(f"Networks Phase Prediction Results Comparison", 
              fontsize=17, fontweight='bold', pad=20)
    
    # ----- F. 保存并展示 -----
    plt.tight_layout()
    save_path = os.path.join(dir_path, os.pardir, "Essay_image_results", f"Sequence_Networks_comparison.png")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\n🎉 图表已生成并保存至: {save_path}")
    plt.close()

if __name__ == '__main__':
    main()