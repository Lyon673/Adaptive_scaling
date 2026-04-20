"""
综合可视化脚本：实时操作概率曲线 (Probabilities) + 双臂空间注意力权重 (Alphas)
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

# 设置中文字体与负号显示
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
# 2. 网络结构定义 (严格对齐以确保权重加载成功)
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
        packed_output, _ = self.lstm(packed_input)
        lstm_out, _ = pad_packed_sequence(packed_output, batch_first=True)
        emissions = self.fc(lstm_out)
        return (emissions, alphas) if return_alphas else emissions

def load_combined_model(filepath, device='cpu'):
    checkpoint = torch.load(filepath, map_location=device)
    cfg = checkpoint.get('model_config', {})
    state_dict = checkpoint['model_state_dict']
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
# 3. 组合式双子图绘制引擎
# =========================================================================
def plot_combined_features(probs, alpha_L, alpha_R, true_labels_np, demo_id, save_path=None):
    CLASS_NAMES = ['P0 Right Hand\nMove', 'P1 Pick\nNeedle', 'P2 Right Hand\nMove', 'P3 Pass\nNeedle', 'P4 Left Hand\nMove',  'P5 Left Hand\nPick', 'P6 Pull\nThread']
    STAGE_COLORS = {0: '#FFA3A3', 1: '#F7BB71', 2: '#8FD3E8', 3: '#C2E6D8', 4: '#FFF2C2', 5: '#EBC8A2', 6: '#D7B5F0', -1: '#E0E0E0'}

    coarse_classes = [0, 2, 4, 6]
    fine_classes   = [1, 3, 5]
    
    seq_len = probs.shape[0]
    time_steps = np.arange(seq_len)

    coarse_prob = probs[:, coarse_classes].sum(axis=1)
    fine_prob   = probs[:, fine_classes  ].sum(axis=1)

    # 创建上下两个子图，共享 X 轴
    fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=1, figsize=(16, 9), sharex=True, gridspec_kw={'hspace': 0.15})

    # ==========================
    # 背景阶段色带提取 (共享)
    # ==========================
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
        
        # 在两个子图上都画背景色带
        ax1.axvspan(s - 0.5, e + 0.5, alpha=0.45, color=color, linewidth=0)
        ax2.axvspan(s - 0.5, e + 0.5, alpha=0.45, color=color, linewidth=0)
        
        # 收集一次图例即可
        if lbl not in drawn_labels:
            band_handles.append(patches.Patch(color=color, label=lbl_name.replace('\n', ' ')))
            band_labels.append(lbl_name.replace('\n', ' '))
            drawn_labels.add(lbl)

    # ==========================
    # 子图 1: Probability (上)
    # ==========================
    l_coarse, = ax1.plot(time_steps, coarse_prob, color='#1A6FA8', linewidth=2.5, linestyle='--', label='P(Coarse Phase)', zorder=3)
    l_fine,   = ax1.plot(time_steps, fine_prob, color='#B23F3F', linewidth=2.5, label='P(Fine Phase)', zorder=3)

    ax1.set_ylim(-0.03, 1.08)
    ax1.set_ylabel('Probability', fontsize=13)
    ax1.set_title(f'(a) Real-Time Sequence Operation Probabilities', fontsize=15, fontweight='bold', pad=10)
    ax1.grid(True, alpha=0.3, axis='y')
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    ax1.legend([l_coarse, l_fine], ['P(Coarse Phase)', 'P(Fine Phase)'], loc='upper right', fontsize=11, framealpha=0.9)

    # ==========================
    # 子图 2: Attention Alpha (下)
    # ==========================
    l_alphaL, = ax2.plot(time_steps, alpha_L, color='#5681B9', lw=2.5, label=r'Left PSM Weight ($\alpha_L$)')
    l_alphaR, = ax2.plot(time_steps, alpha_R, color='#E18283', lw=2.5, label=r'Right PSM Weight ($\alpha_R$)')
    
    ax2.axhline(0.5, color='gray', linestyle=':', lw=1.5, alpha=0.7) # 辅助中线

    ax2.set_xlim(-0.5, seq_len - 0.5)
    ax2.set_ylim(-0.03, 1.05)
    ax2.set_xlabel('Frame Index', fontsize=13)
    ax2.set_ylabel('Attention Weight', fontsize=13)
    ax2.set_title(f'(b) Bimanual Spatial Dominance Weights', fontsize=15, fontweight='bold', pad=10)
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    ax2.legend([l_alphaL, l_alphaR], [r'Left PSM Weight ($\alpha_L$)', r'Right PSM Weight ($\alpha_R$)'], loc='upper right', fontsize=11, framealpha=0.9)

    # ==========================
    # 全局阶段底部图例
    # ==========================
    fig.legend(band_handles, band_labels, loc='lower center', ncol=min(7, len(band_labels)), 
               fontsize=12, title='Ground Truth Surgical Phase', title_fontsize=12, 
               bbox_to_anchor=(0.5, -0.05), framealpha=0.9)

    plt.tight_layout()
    # 为底部图例留出空间
    plt.subplots_adjust(bottom=0.1)
    

    plt.show()
    plt.close(fig)

# =========================================================================
# 4. 主函数执行流
# =========================================================================
if __name__ == "__main__":
    for i in range(0,163):
        TARGET_DEMO_ID = i  # 指定需要绘制的 demo_id

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"使用设备: {device}")

        dir_path = os.path.dirname(__file__)
        model_path = os.path.join(dir_path, "LSTM_model", "spatial_attn_lstmcrf_sequence_model2.pth")
        
        if not os.path.exists(model_path):
            print(f"❌ 找不到模型文件: {model_path}")
            exit()

        print("正在加载模型...")
        model = load_combined_model(model_path, device)

        # 加载数据
        print(f"正在抓取 Demo ID {TARGET_DEMO_ID} 的状态与标签...")
        states = load_specific_test_state(shuffle=False, without_quat=without_quat, resample=resample, demo_id_list=[TARGET_DEMO_ID])
        labels = load_specific_test_label(demo_id_list=[TARGET_DEMO_ID])

        if len(states) == 0 or len(labels) == 0:
            print(f"❌ 无法在测试集中找到 Demo {TARGET_DEMO_ID}！")
            exit()

        sequence = torch.tensor(states[0], dtype=torch.float32).to(device)
        true_labels_np = np.array(labels[0], dtype=int)
        
        if sequence.dim() == 1:
            sequence = sequence.unsqueeze(1)

        print("正在进行联合前向推演...")
        with torch.no_grad():
            # 一次性获取 Emissions 和 Alphas
            emissions, alphas = model(sequence.unsqueeze(0), [sequence.shape[0]], return_alphas=True)
            emissions_squeeze = emissions.squeeze(0)  # (seq_len, num_classes)
            
            # 提取 Alphas
            alpha_L = alphas[0, :, 0].cpu().numpy()
            alpha_R = alphas[0, :, 1].cpu().numpy()
            
            # 计算 CRF 实时概率
            probs = get_realtime_probs_for_sequence(emissions_squeeze, model.crf)

        # 绘制并保存结果
        save_dir = os.path.join(dir_path, os.pardir, "Essay_image_results")
        save_path = os.path.join(save_dir, f"Combined_Features_Demo_{TARGET_DEMO_ID}.png")
        
        print("正在渲染并合并图表...")
        
        # 提示：如果您只想截取序列的一部分来展示（比如 [20:155]），可以在这里对数组进行切片：
        # plot_combined_features(probs[20:155], alpha_L[20:155], alpha_R[20:155], true_labels_np[20:155], TARGET_DEMO_ID, save_path=save_path)

        
        plot_combined_features(probs, alpha_L, alpha_R, true_labels_np, i, save_path=save_path)