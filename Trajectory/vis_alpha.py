"""
特定 Demo 的双臂空间注意力权重 (Bimanual Attention Alphas) 可视化脚本
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
# 1. 网络结构定义 (严格对齐以确保权重加载成功)
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
            X_L_t, X_R_t = X_L.transpose(1, 2), X_R.transpose(1, 2)
            X_L_pad, X_R_pad = F.pad(X_L_t, (2, 0)), F.pad(X_R_t, (2, 0))
            H_L = F.relu(self.shared_causal_conv(X_L_pad)).transpose(1, 2)
            H_R = F.relu(self.shared_causal_conv(X_R_pad)).transpose(1, 2)
        else:
            H_L = F.relu(self.shared_proj(X_L))
            H_R = F.relu(self.shared_proj(X_R))

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

    def decode(self, x, lengths, return_alphas=False):
        mask = torch.arange(x.size(1), device=x.device)[None, :] < torch.tensor(lengths, device=x.device)[:, None]
        if return_alphas:
            emissions, alphas = self.forward(x, lengths, return_alphas=True)
            return self.crf.decode(emissions, mask=mask), alphas
        return self.crf.decode(self.forward(x, lengths), mask=mask)

def load_spatial_attn_model(filepath, device='cpu'):
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
# 2. 高精度图表绘制引擎
# =========================================================================
def plot_single_demo_alphas(model, sequence, true_labels_np, demo_id, device, save_path=None):
    STAGE_COLORS = {0: '#FFA3A3', 1: '#F7BB71', 2: '#8FD3E8', 3: '#C2E6D8', 4: '#FFF2C2', 5: '#EBC8A2', 6: '#D7B5F0', -1: '#E0E0E0'}
    CLASS_NAMES = ['P0 Right Hand Move', 'P1 Pick Needle', 'P2 Right Hand Move', 'P3 Pass Needle', 'P4 Left Hand Move',  'P5 Left Hand Pick', 'P6 Pull Thread']
   
    
    T = sequence.shape[0]
    
    # 1. 运行前向传播并截获 alphas
    model.eval()
    with torch.no_grad():
        x = sequence.unsqueeze(0).to(device)
        _, alphas = model.decode(x, [T], return_alphas=True)
        alpha_L = alphas[0, :, 0].cpu().numpy()
        alpha_R = alphas[0, :, 1].cpu().numpy()

    t_steps = np.arange(T)
    
    fig, ax = plt.subplots(figsize=(15, 5))
    
    # 2. 绘制真实阶段背景色带 (Ground Truth Bands)
    prev_label, span_start = true_labels_np[0], 0
    for t in range(1, T):
        if true_labels_np[t] != prev_label:
            color = STAGE_COLORS.get(int(prev_label), '#F0F0F0')
            ax.axvspan(span_start, t, color=color, alpha=0.45, lw=0)
            span_start, prev_label = t, true_labels_np[t]
    ax.axvspan(span_start, T, color=STAGE_COLORS.get(int(prev_label), '#F0F0F0'), alpha=0.45, lw=0)

    # 3. 绘制 Alpha 曲线
    ax.plot(t_steps, alpha_L, color='#5681B9', lw=2.5, label=r'Left PSM Dominance Weight ($\alpha_L$)')
    ax.plot(t_steps, alpha_R, color='#E18283', lw=2.5, label=r'Right PSM Dominance Weight ($\alpha_R$)')

    # 辅助中线
    ax.axhline(0.5, color='gray', linestyle=':', lw=1.5, alpha=0.7)

    # 4. 图表结构修饰
    ax.set_xlim(0, T)
    ax.set_ylim(-0.02, 1.05)
    ax.set_title(f"Bimanual Motion Dominance Weights in Phases", fontsize=15, fontweight='bold', pad=15)
    ax.set_xlabel("Frames", fontsize=12)
    ax.set_ylabel("Weight", fontsize=12)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # 5. 双层图例设置
    # 上层图例：Alpha 曲线
    line_legends = ax.legend(loc='upper right', framealpha=0.9, fontsize=11)
    ax.add_artist(line_legends)

    # 下层图例：手术阶段颜色
    bg_patches = [patches.Patch(color=STAGE_COLORS[i], label=CLASS_NAMES[i]) for i in range(7)]
    fig.legend(handles=bg_patches, loc='lower center', bbox_to_anchor=(0.5, -0.05), ncol=7, frameon=False, fontsize=10)

    plt.tight_layout(rect=[0, 0.08, 1, 1])
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✅ Demo {demo_id} Alpha曲线已保存至: {save_path}")
    
    plt.show()
    plt.close(fig)


# =========================================================================
# 3. 主函数执行流
# =========================================================================
if __name__ == "__main__":
    TARGET_DEMO_ID = 65

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")

    dir_path = os.path.dirname(__file__)
    # 请确保您的权重文件路径与文件名一致
    model_path = os.path.join(dir_path, "LSTM_model", "spatial_attn_lstmcrf_sequence_model2.pth")
    
    if not os.path.exists(model_path):
        print(f"❌ 找不到模型文件: {model_path}")
        exit()

    print("正在加载模型...")
    model = load_spatial_attn_model(model_path, device)

    # 仅加载指定 Demo 的数据
    print(f"正在抓取 Demo ID {TARGET_DEMO_ID} 的状态与标签...")
    states = load_specific_test_state(shuffle=False, without_quat=without_quat, resample=resample, demo_id_list=[TARGET_DEMO_ID])
    labels = load_specific_test_label(demo_id_list=[TARGET_DEMO_ID])

    if len(states) == 0 or len(labels) == 0:
        print(f"❌ 无法在测试集中找到 Demo {TARGET_DEMO_ID}！请检查它是否属于过滤集。")
        exit()

    sequence = torch.tensor(states[0], dtype=torch.float32).to(device)
    true_labels_np = np.array(labels[0], dtype=int)
    
    if sequence.dim() == 1:
        sequence = sequence.unsqueeze(1)

    # 定义保存路径
    save_path = os.path.join(dir_path, os.pardir, "Essay_image_results", f"Bimanual_Weight_Alpha.png")
    
    print("正在计算 Attention Alpha 权重并渲染图表...")
    plot_single_demo_alphas(model, sequence[20:155], true_labels_np[20:155], TARGET_DEMO_ID, device, save_path=save_path)