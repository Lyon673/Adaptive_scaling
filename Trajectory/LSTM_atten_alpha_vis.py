import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import numpy as np
import matplotlib.pyplot as plt
import os
import matplotlib.patches as patches
from torchcrf import CRF

from load_data import load_test_state, load_test_label, get_test_demo_id_list, load_specific_test_state, load_specific_test_label
from config import resample, without_quat

STAGE_COLORS = {
    0: '#A8D5E2', 1: '#F4A9A8', 2: '#A8C5DA',
    3: '#F7C5A0', 4: '#B5D5C5', 5: '#D5A8D4',
    6: '#C5D5A8', -1: '#F0F0F0'
}

CLASS_NAMES = [
    'P0 Right Move', 'P1 Pick Needle', 'P2 Right Move2', 
    'P3 Pass Needle', 'P4 Left Move', 'P5 Left Pick', 'P6 Pull Thread'
]

# =========================================================================
# --- 1. 自适应模型层：支持旧版 nn.Linear 和新版 因果卷积 ---
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

        # 动态处理卷积与线性层映射
        if self.use_causal_conv:
            X_L_t = X_L.transpose(1, 2)
            X_R_t = X_R.transpose(1, 2)
            # 严格保护未来帧 (左侧 padding)
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
    """
    解耦的模型配置容器，通过 use_causal_conv 参数动态切换空间注意力机制的网络组件。
    """
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


# =========================================================================
# --- 2. 权重加载与注意力可视化流 ---
# =========================================================================

def plot_bimanual_alphas():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dir_path = os.path.dirname(__file__)
    
    # 指向您的目标权重文件
    model_path = os.path.join(dir_path, "LSTM_model", "spatial_attn_lstmcrf_sequence_model.pth")
    
    if not os.path.exists(model_path):
        print(f"未找到模型文件: {model_path}，请确认路径或先进行训练。")
        return
        
    # --- 1. 动态权重嗅探与模型实例化 ---
    ckpt = torch.load(model_path, map_location=device)
    cfg = ckpt.get('model_config', {})
    state_dict = ckpt['model_state_dict']

    # 嗅探当前参数字典中是否存在因果卷积组件
    use_causal_conv = "spatial_attention.shared_causal_conv.weight" in state_dict.keys()
    arch_name = "Causal-Conv (New)" if use_causal_conv else "Linear (Old)"
    print(f"正在加载模型用于注意力解析... [推断架构类型: {arch_name}]")
    
    model = AdaptiveSequenceLabelingLSTM_CRF(
        input_size=16, 
        hidden_size=cfg.get('hidden_size', 256),
        num_layers=cfg.get('num_layers', 3),
        num_classes=7,
        proj_dim=cfg.get('proj_dim', 16),
        use_causal_conv=use_causal_conv
    )
    
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    # --- 2. 加载测试数据 ---
    target_demo_id = 9
    demo_id_list = np.delete(np.arange(148), [80, 81, 92, 109, 112, 117, 122, 144, 145])
    test_demo_ids = get_test_demo_id_list(demo_id_list)
    
    # if target_demo_id not in test_demo_ids:
    #     print(f"⚠️ 目标序列 Demo ID {target_demo_id} 不在当前划分的测试集中。")
    #     return
        
    # idx = list(test_demo_ids).index(target_demo_id)
    
    # states = load_test_state(without_quat=without_quat, resample=resample, demo_id_list=demo_id_list)
    # labels = load_test_label(resample=resample, demo_id_list=demo_id_list)
    states = load_specific_test_state(shuffle=False, without_quat=without_quat, resample=resample, demo_id_list=[target_demo_id])
    labels = load_specific_test_label(demo_id_list=[target_demo_id])
    
    x = torch.tensor(states[0], dtype=torch.float32).unsqueeze(0).to(device)
    true_labels = np.array(labels[0], dtype=int)
    T = x.shape[1]

    # --- 3. 前向推理并截获 Alphas ---
    with torch.no_grad():
        # 调用 return_alphas=True 接口同步解析权重分配
        best_path, alphas = model.decode(x, [T], return_alphas=True)
        # alphas shape: (1, T, 2)
        alpha_L = alphas[0, :, 0].cpu().numpy()
        alpha_R = alphas[0, :, 1].cpu().numpy()

    # --- 4. 学术级可视化绘图 ---
    fig, ax = plt.subplots(figsize=(14, 5))
    t_steps = np.arange(T)
    
    # 绘制背景颜色带 (映射真实的阶段标签)
    prev_label, span_start = true_labels[0], 0
    for t in range(1, T):
        if true_labels[t] != prev_label:
            color = STAGE_COLORS.get(int(prev_label), '#F0F0F0')
            ax.axvspan(span_start, t, color=color, alpha=0.4, lw=0)
            span_start, prev_label = t, true_labels[t]
    ax.axvspan(span_start, T, color=STAGE_COLORS.get(int(prev_label), '#F0F0F0'), alpha=0.4, lw=0)

    # 绘制左臂与右臂的主导权重演变曲线
    ax.plot(t_steps, alpha_L, color='#2980b9', lw=2.0, label=r'Left Arm Dominance ($\alpha_L$)')
    ax.plot(t_steps, alpha_R, color='#c0392b', lw=2.0, linestyle='--', label=r'Right Arm Dominance ($\alpha_R$)')

    # 添加 0.5 决策基准线
    ax.axhline(0.5, color='gray', linestyle=':', lw=1.5, alpha=0.7)

    # 坐标轴与边界装饰
    ax.set_xlim(0, T)
    ax.set_ylim(0, 1.05)
    ax.set_xlabel("Time Step (Frames)", fontsize=12)
    ax.set_ylabel("Attention Weight (0~1)", fontsize=12)
    ax.set_title(f"Bimanual Spatial Cross-Attention Dominance (Demo ID: {target_demo_id})", fontsize=14, fontweight='bold', pad=15)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # 组装复合型图例
    line_legends = ax.legend(loc='upper right', framealpha=0.9)
    ax.add_artist(line_legends)
    
    bg_patches = [patches.Patch(color=STAGE_COLORS[i], label=CLASS_NAMES[i]) for i in range(7)]
    ax.legend(handles=bg_patches, loc='lower center', bbox_to_anchor=(0.5, -0.25), ncol=4, frameon=False, title="True Surgical Phases")

    plt.tight_layout()
    save_path = os.path.join(dir_path, f"bimanual_alphas_demo_{target_demo_id}.png")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"注意力分配曲线已保存至: {save_path}")

if __name__ == "__main__":
    plot_bimanual_alphas()