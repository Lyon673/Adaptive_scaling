import torch
import numpy as np
import matplotlib.pyplot as plt
import os
import sys

# 导入你修改后的网络和数据加载器
from TeCNO_attention_seg_train import load_model, collate_fn, KinematicDataset
from load_data import load_test_state, load_test_label
from config import resample, without_quat

def plot_attention_matrices(left_attn, right_attn, demo_length, save_path="attention_heatmap.png"):
    """
    绘制双臂交叉注意力热图
    """
    # 截取有效长度的矩阵部分
    left_matrix = left_attn[0, :demo_length, :demo_length].cpu().numpy()
    right_matrix = right_attn[0, :demo_length, :demo_length].cpu().numpy()

    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    # 1. 绘制 左臂看右臂 (Left from Right) 的注意力
    im1 = axes[0].imshow(left_matrix, cmap='viridis', aspect='auto', origin='upper')
    axes[0].set_title('Left Arm Attention (Looking at Right Arm)', fontsize=14)
    axes[0].set_xlabel('Key / Value Time Steps (Right Arm)', fontsize=12)
    axes[0].set_ylabel('Query Time Steps (Left Arm)', fontsize=12)
    fig.colorbar(im1, ax=axes[0])

    # 2. 绘制 右臂看左臂 (Right from Left) 的注意力
    im2 = axes[1].imshow(right_matrix, cmap='viridis', aspect='auto', origin='upper')
    axes[1].set_title('Right Arm Attention (Looking at Left Arm)', fontsize=14)
    axes[1].set_xlabel('Key / Value Time Steps (Left Arm)', fontsize=12)
    axes[1].set_ylabel('Query Time Steps (Right Arm)', fontsize=12)
    fig.colorbar(im2, ax=axes[1])

    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    print(f"注意力热图已保存至: {save_path}")
    plt.show()

if __name__ == "__main__":
    # 1. 加载模型
    current_dir = os.path.dirname(__file__)
    model_path = os.path.join(current_dir, "TeCNO_model", "tecno_sequence_model.pth")
    model, _ = load_model(model_path)
    model.eval()

    # 2. 获取测试集的一个 Demo
    excluded = [80, 81, 92, 109, 112, 117, 122, 144, 145]
    demo_id_list = np.delete(np.arange(148), excluded)
    
    test_states = load_test_state(without_quat=without_quat, resample=resample, demo_id_list=demo_id_list)
    test_labels = load_test_label(resample=resample, demo_id_list=demo_id_list)
    
    # 我们只取测试集的第一个 Demo
    demo_seq = test_states[0]
    demo_len = len(demo_seq)
    print(f"提取测试集 Demo 0，序列长度: {demo_len} 帧")

    # 3. 准备张量输入
    # Extract the device from the model's parameters safely
    device = next(model.parameters()).device
    seq_t = torch.tensor(demo_seq, dtype=torch.float32).unsqueeze(0).to(device) # (1, T, 16)
    lengths = [demo_len]

    # 4. 执行前向传播并获取 Attention Weights
    with torch.no_grad():
        logits, left_attn, right_attn = model(seq_t, lengths, return_attn=True)

    # 5. 画图
    plot_attention_matrices(left_attn, right_attn, demo_len)