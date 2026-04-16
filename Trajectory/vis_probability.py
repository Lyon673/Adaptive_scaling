"""
特定 Demo (Demo ID = 9) 的实时因果滤波概率曲线可视化脚本
基于 Spatial Attention LSTM-CRF 模型
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# 导入您项目中的依赖
from load_data import load_specific_test_state, load_specific_test_label
from config import resample, without_quat
from torchcrf import CRF

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# =========================================================================
# 1. 实时因果滤波概率估计器 (Real-Time CRF Probability Estimator)
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
# 2. 网络结构定义 (Spatial Attn LSTM-CRF)
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
            nn.Linear(proj_dim * 2, proj_dim), nn.ReLU(), nn.Linear(proj_dim, 2)
        )

    def forward(self, x):
        X_L, X_R = x[:, :, :8], x[:, :, 8:]
        if self.use_causal_conv:
            X_L_pad = F.pad(X_L.transpose(1, 2), (2, 0))
            X_R_pad = F.pad(X_R.transpose(1, 2), (2, 0))
            H_L = F.relu(self.shared_causal_conv(X_L_pad)).transpose(1, 2)
            H_R = F.relu(self.shared_causal_conv(X_R_pad)).transpose(1, 2)
        else:
            H_L, H_R = F.relu(self.shared_proj(X_L)), F.relu(self.shared_proj(X_R))

        H_cat = torch.cat([H_L, H_R], dim=-1)
        alphas = F.softmax(self.dominance_mlp(H_cat), dim=-1)
        tilde_X = torch.cat([alphas[:, :, 0].unsqueeze(-1) * X_L, alphas[:, :, 1].unsqueeze(-1) * X_R], dim=-1)
        return self.layer_norm(x + tilde_X) if self.use_causal_conv else tilde_X, alphas

class AdaptiveSequenceLabelingLSTM_CRF(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes, proj_dim=16, use_causal_conv=False):
        super(AdaptiveSequenceLabelingLSTM_CRF, self).__init__()
        self.spatial_attention = AdaptiveBimanualSpatialAttention(input_dim=8, proj_dim=proj_dim, use_causal_conv=use_causal_conv)
        self.lstm = nn.LSTM(input_size=16, hidden_size=hidden_size, num_layers=num_layers, batch_first=True, dropout=0.2 if num_layers > 1 else 0)
        self.fc = nn.Linear(hidden_size, num_classes)
        self.crf = CRF(num_classes, batch_first=True)

    def forward(self, x, lengths):
        tilde_X, _ = self.spatial_attention(x)
        packed_input = pack_padded_sequence(tilde_X, lengths, batch_first=True, enforce_sorted=False)
        packed_output, _ = self.lstm(packed_input)
        lstm_out, _ = pad_packed_sequence(packed_output, batch_first=True)
        return self.fc(lstm_out)

def load_spatial_attn_lstmcrf_model(filepath, device='cpu'):
    checkpoint = torch.load(filepath, map_location=device)
    cfg, state_dict = checkpoint.get('model_config', {}), checkpoint['model_state_dict']
    use_causal_conv = "spatial_attention.shared_causal_conv.weight" in state_dict.keys()
    
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

# =========================================================================
# 3. 单个 Demo 绘图逻辑
# =========================================================================
def plot_single_demo_probability(probs, true_labels_np, demo_id, save_path=None):
    CLASS_NAMES = ['P0 Right Hand\nMove', 'P1 Pick\nNeedle', 'P2 Right Hand\nMove', 'P3 Pass\nNeedle', 'P4 Left Hand\nMove',  'P5 Left Hand\nPick', 'P6 Pull\nThread']
    STAGE_COLORS = {0: '#FFA3A3', 1: '#F7BB71', 2: '#8FD3E8', 3: '#C2E6D8', 4: '#FFF2C2', 5: '#EBC8A2', 6: '#D7B5F0', -1: '#E0E0E0'}

    coarse_classes = [0, 2, 4, 6]
    fine_classes   = [1, 3, 5]
    
    seq_len = probs.shape[0]
    time_steps = np.arange(seq_len)

    coarse_prob = probs[:, coarse_classes].sum(axis=1)
    fine_prob   = probs[:, fine_classes  ].sum(axis=1)

    fig, ax = plt.subplots(figsize=(16, 5))

    # 1. 绘制真实阶段背景色带
    prev_label, span_start, spans = true_labels_np[0], 0, []
    for t in range(1, seq_len):
        if true_labels_np[t] != prev_label:
            spans.append((span_start, t - 1, prev_label))
            span_start, prev_label = t, true_labels_np[t]
    spans.append((span_start, seq_len - 1, prev_label))

    drawn_labels = set()
    band_handles, band_labels = [], []
    for (s, e, lbl) in spans:
        color = STAGE_COLORS.get(int(lbl), '#DDDDDD')
        lbl_name = CLASS_NAMES[int(lbl)] if 0 <= lbl < len(CLASS_NAMES) else 'Unlabeled'
        patch = ax.axvspan(s - 0.5, e + 0.5, alpha=0.45, color=color, linewidth=0)
        if lbl not in drawn_labels:
            band_handles.append(patch)
            band_labels.append(lbl_name.replace('\n', ' '))
            drawn_labels.add(lbl)

    # 2. 绘制概率曲线
    l_coarse, = ax.plot(time_steps, coarse_prob, color='#1A6FA8', linewidth=2.5, linestyle='--', label='P(Coarse)', zorder=3)
    l_fine,   = ax.plot(time_steps, fine_prob, color='#B23F3F', linewidth=2.5, label='P(Fine)', zorder=3)

    # 3. 设置图表样式
    ax.set_xlim(-0.5, seq_len - 0.5)
    ax.set_ylim(-0.03, 1.08)
    ax.set_xlabel('Frame', fontsize=12)
    ax.set_ylabel('Predicted Probability', fontsize=12)
    ax.set_title(f'Coarse and Fine Operation Probabilities', fontsize=15, fontweight='bold', pad=15)
    ax.grid(True, alpha=0.3, axis='y')

    # 图例设置
    ax.legend([l_coarse, l_fine], ['P(Coarse)', 'P(Fine)'], loc='upper right', fontsize=11, framealpha=0.9)
    fig.legend(band_handles, band_labels, loc='lower center', ncol=min(7, len(band_labels)), fontsize=10, title='Surgical Phase', bbox_to_anchor=(0.5, -0.05), framealpha=0.9)

    plt.tight_layout(rect=[0, 0.08, 1, 1])
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✅ Demo {demo_id} 概率曲线已保存至: {save_path}")
    plt.show()
    plt.close(fig)

# =========================================================================
# 4. 主函数逻辑
# =========================================================================
if __name__ == "__main__":
    TARGET_DEMO_ID = 54  # 指定需要单独绘制的 demo_id

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")

    # 1. 寻找并加载模型
    dir_path = os.path.dirname(__file__)
    model_path = os.path.join(dir_path, "LSTM_model", "spatial_attn_lstmcrf_sequence_model2.pth") # 注意核对您的模型文件名
    
    if not os.path.exists(model_path):
        print(f"❌ 模型文件不存在: {model_path}")
        exit()

    print("正在加载模型...")
    model = load_spatial_attn_lstmcrf_model(model_path, device)

    # 2. 单独加载 Demo 9 的数据
    print(f"正在加载 Demo ID {TARGET_DEMO_ID} 的数据...")
    states = load_specific_test_state(shuffle=False, without_quat=without_quat, resample=resample, demo_id_list=[TARGET_DEMO_ID])
    labels = load_specific_test_label(demo_id_list=[TARGET_DEMO_ID])
    
    if len(states) == 0 or len(labels) == 0:
        print(f"❌ 未找到 Demo {TARGET_DEMO_ID} 的数据！")
        exit()

    sequence = torch.tensor(states[0], dtype=torch.float32).to(device)
    true_labels_np = np.array(labels[0], dtype=int)
    if sequence.dim() == 1:
        sequence = sequence.unsqueeze(1)

    # 3. 模型推理与因果滤波概率计算
    print("正在推演实时概率...")
    with torch.no_grad():
        logits = model(sequence.unsqueeze(0), [sequence.shape[0]])
        logits_squeeze = logits.squeeze(0)  # (seq_len, num_classes)
        
        if hasattr(model, 'crf'):
            probs = get_realtime_probs_for_sequence(logits_squeeze, model.crf)
        else:
            probs = torch.softmax(logits_squeeze, dim=1).cpu().numpy()

    # 4. 绘制并保存结果
    save_dir = os.path.join(dir_path, os.pardir, "Essay_image_results")
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f"Sequence_Probability_Curve.png")

    print("正在生成高清图表...")
    plot_single_demo_probability(probs[15:160], true_labels_np[15:160], demo_id=TARGET_DEMO_ID, save_path=save_path)