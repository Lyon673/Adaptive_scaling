import torch
import numpy as np
import matplotlib.pyplot as plt
import os
import matplotlib.patches as patches

from load_data import load_test_state, load_test_label, get_test_demo_id_list
from config import resample, without_quat
from LSTM_atten_seg_train import SequenceLabelingLSTM_CRF # 直接引入重构后的架构

STAGE_COLORS = {
    0: '#A8D5E2', 1: '#F4A9A8', 2: '#A8C5DA',
    3: '#F7C5A0', 4: '#B5D5C5', 5: '#D5A8D4',
    6: '#C5D5A8', -1: '#F0F0F0'
}

CLASS_NAMES = [
    'P0 Right Move', 'P1 Pick Needle', 'P2 Right Move2', 
    'P3 Pass Needle', 'P4 Left Move', 'P5 Left Pick', 'P6 Pull Thread'
]

def plot_bimanual_alphas():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dir_path = os.path.dirname(__file__)
    model_path = os.path.join(dir_path, "LSTM_model", "spatial_attn_lstmcrf_sequence_model.pth")
    
    if not os.path.exists(model_path):
        print("未找到模型，请先训练 spatial_attn_lstmcrf_sequence_model.pth")
        return
        
    # --- 1. 加载模型 ---
    ckpt = torch.load(model_path, map_location=device)
    cfg = ckpt['model_config']
    model = SequenceLabelingLSTM_CRF(
        input_size=16, 
        hidden_size=cfg.get('hidden_size', 256),
        num_layers=cfg.get('num_layers', 3),
        num_classes=7,
        proj_dim=cfg.get('proj_dim', 16)
    )
    model.load_state_dict(ckpt['model_state_dict'])
    model.to(device)
    model.eval()

    # --- 2. 加载数据 ---
    target_demo_id = 9
    demo_id_list = np.delete(np.arange(148), [80, 81, 92, 109, 112, 117, 122, 144, 145])
    test_demo_ids = get_test_demo_id_list(demo_id_list)
    idx = list(test_demo_ids).index(target_demo_id)
    
    states = load_test_state(without_quat=without_quat, resample=resample, demo_id_list=demo_id_list)
    labels = load_test_label(resample=resample, demo_id_list=demo_id_list)
    
    x = torch.tensor(states[idx], dtype=torch.float32).unsqueeze(0).to(device)
    true_labels = np.array(labels[idx], dtype=int)
    T = x.shape[1]

    # --- 3. 前向推理并截获 Alphas ---
    with torch.no_grad():
        # 调用新加入的 return_alphas=True 接口
        best_path, alphas = model.decode(x, [T], return_alphas=True)
        # alphas shape: (1, T, 2)
        alpha_L = alphas[0, :, 0].cpu().numpy()
        alpha_R = alphas[0, :, 1].cpu().numpy()

    # --- 4. 严谨的学术级可视化 ---
    fig, ax = plt.subplots(figsize=(14, 5))
    t_steps = np.arange(T)
    
    # 绘制背景颜色带 (根据真实标签)
    prev_label, span_start = true_labels[0], 0
    for t in range(1, T):
        if true_labels[t] != prev_label:
            color = STAGE_COLORS.get(int(prev_label), '#F0F0F0')
            ax.axvspan(span_start, t, color=color, alpha=0.4, lw=0)
            span_start, prev_label = t, true_labels[t]
    ax.axvspan(span_start, T, color=STAGE_COLORS.get(int(prev_label), '#F0F0F0'), alpha=0.4, lw=0)

    # 绘制左臂与右臂的主导权重曲线
    ax.plot(t_steps, alpha_L, color='#2980b9', lw=2.0, label=r'Left Arm Dominance ($\alpha_L$)')
    ax.plot(t_steps, alpha_R, color='#c0392b', lw=2.0, linestyle='--', label=r'Right Arm Dominance ($\alpha_R$)')

    # 添加 0.5 参考线
    ax.axhline(0.5, color='gray', linestyle=':', lw=1.5, alpha=0.7)

    # 装饰图表
    ax.set_xlim(0, T)
    ax.set_ylim(0, 1.05)
    ax.set_xlabel("Time Step (Frames)", fontsize=12)
    ax.set_ylabel("Attention Weight (0~1)", fontsize=12)
    ax.set_title(f"Bimanual Spatial Cross-Attention Dominance (Demo ID: {target_demo_id})", fontsize=14, fontweight='bold', pad=15)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # 创建复合图例
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