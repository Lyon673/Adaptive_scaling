import torch
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def visualize_crf_transitions(model_path):
    # 1. 检查模型文件是否存在
    if not os.path.exists(model_path):
        print(f"找不到模型文件: {model_path}")
        return

    # 2. 加载模型 checkpoint (映射到 CPU 防止显存报错)
    print(f"正在加载模型: {model_path}")
    checkpoint = torch.load(model_path, map_location=torch.device('cpu'))
    
    # 获取 state_dict
    state_dict = checkpoint['model_state_dict']
    
    # 3. 提取 CRF 转移矩阵
    # 在 torchcrf 中，转移矩阵的 key 通常是 'crf.transitions'
    if 'crf.transitions' not in state_dict:
        print("错误: 在 state_dict 中找不到 'crf.transitions'。")
        print("当前包含的键值有:", state_dict.keys())
        return
        
    transition_matrix = state_dict['crf.transitions'].numpy()
    
    # 获取类别数量
    num_classes = transition_matrix.shape[0]
    class_labels = [f"Phase {i}" for i in range(num_classes)]
    
    # 4. 绘制热力图
    plt.figure(figsize=(10, 8))
    
    # 设置热力图参数，中心点设为0，暖色代表正分(鼓励转移)，冷色代表负分(惩罚转移)
    sns.heatmap(transition_matrix, 
                annot=True,          # 在格子里显示具体数值
                fmt=".2f",           # 保留两位小数
                cmap="RdBu_r",       # 红-蓝配色 (红正蓝负)
                center=0,            # 强制0为颜色分界点
                xticklabels=class_labels, 
                yticklabels=class_labels,
                linewidths=.5)
    
    plt.title("CRF Transition Matrix Learned from Training Data", fontsize=14, pad=20)
    plt.xlabel("To State (Next Phase)", fontsize=12, labelpad=10)
    plt.ylabel("From State (Current Phase)", fontsize=12, labelpad=10)
    
    # 旋转标签以便阅读
    plt.xticks(rotation=45)
    plt.yticks(rotation=0)
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # 根据你代码中的保存逻辑，构建正确的路径
    dir_path = os.path.dirname(os.path.abspath(__file__))
    model_save_path = os.path.join(dir_path, "LSTM_model", "attn_lstmcrf_sequence_model.pth")
    
    visualize_crf_transitions(model_save_path)