"""
快速可视化脚本 - 简化版本
用于快速查看序列预测结果
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from LSTM_seg_test import load_model, TestDataset
import os
import seaborn as sns

def quick_visualize(model, test_dataset, device, start, end):
    """快速可视化前几个序列的预测结果"""
    num_sequences = end - start + 1
    # 设置颜色
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7']
    class_names = ['Class 0', 'Class 1', 'Class 2', 'Class 3', 'Class 4']
    
    fig, axes = plt.subplots(num_sequences, 1, figsize=(12, 2*num_sequences))
    if num_sequences == 1:
        axes = [axes]
    
    for seq_idx in range(start, end + 1):
        ax = axes[seq_idx-start]
        
        # 获取数据
        sequence, true_labels = test_dataset[seq_idx]
        
        # 预测
        model.eval()
        with torch.no_grad():
            input_tensor = sequence.unsqueeze(0).to(device)
            lengths = [sequence.shape[0]]
            
            if hasattr(model, 'crf'):
                preds_list = model.decode(input_tensor, lengths)
                predictions = preds_list[0]
            else:
                outputs = model(input_tensor, lengths)
                predictions = torch.argmax(outputs, dim=2).squeeze().cpu().numpy()
        
        # 确保长度一致
        min_len = min(len(true_labels), len(predictions))
        true_labels = true_labels[:min_len].numpy()
        predictions = predictions[:min_len]
        
        # 确保predictions是numpy数组
        if isinstance(predictions, list):
            predictions = np.array(predictions)
        
        # 调试信息
        print(f"Sequence {seq_idx}:")
        print(f"  True labels unique: {np.unique(true_labels)}")
        print(f"  Pred labels unique: {np.unique(predictions)}")
        print(f"  True labels count: {[np.sum(true_labels == i) for i in range(5)]}")
        print(f"  Pred labels count: {[np.sum(predictions == i) for i in range(5)]}")
        print(f"  Accuracy: {np.mean(true_labels == predictions):.3f}")

        print("--------------------------------")
        for i in range(min_len-1):
            if true_labels[i] != true_labels[i+1]:
                print(f"Sequence {seq_idx}: at time {i} True label {true_labels[i]} -> {true_labels[i+1]}")
            if predictions[i] != predictions[i+1]:
                print(f"Sequence {seq_idx}: at time {i} Pred label {predictions[i]} -> {predictions[i+1]}")
        
        # 绘制
        time_steps = np.arange(min_len)
        
        # 真实标签（上方）
        for i in range(5):
            mask = true_labels == i
            if np.any(mask):
                ax.scatter(time_steps[mask], np.ones(np.sum(mask)) * 1.1, 
                          c=colors[i], s=30, alpha=0.8, marker='o')
        
        # 预测标签（下方）
        for i in range(5):
            mask = predictions == i
            if np.any(mask):
                ax.scatter(time_steps[mask], np.ones(np.sum(mask)) * 0.9, 
                          c=colors[i], s=20, alpha=0.6, marker='^')
        
        # 计算准确率
        accuracy = np.mean(true_labels == predictions)
        
        # 设置图形
        ax.set_ylim(0.7, 1.3)
        ax.set_title(f'Sequence {seq_idx + 1} - Accuracy: {accuracy:.3f}')
        ax.set_yticks([0.9, 1.1])
        ax.set_yticklabels(['Predicted', 'True'])
        ax.grid(True, alpha=0.3)
        
        # 添加图例（只在第一个子图）
        if seq_idx == 0:
            legend_elements = [plt.scatter([], [], c=colors[i], s=30, label=class_names[i]) 
                             for i in range(5)]
            ax.legend(handles=legend_elements, loc='upper right')

    # transition_matrix = model.crf.transitions.data.cpu().numpy()

    # print("Learned Transition Matrix:")
    # print(transition_matrix)    

    # plt.figure(figsize=(8, 6))
    # sns.heatmap(transition_matrix, annot=True, cmap='viridis')
    # plt.xlabel("From Label")
    # plt.ylabel("To Label")
    # plt.title("CRF Transition Scores")
    # plt.show()
    
    plt.tight_layout()
    plt.show()
    
    return accuracy


def main():
    """主函数"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    # 加载模型
    dir_path = os.path.dirname(__file__)
    model_path = os.path.join(dir_path, "LSTM_model", "lstm_sequence_model.pth")
    #model_path = "LSTM_model/lstm_sequence_model.pth"
    model = load_model(model_path, device)
    print("✅ 模型加载成功")
    
    # 创建测试数据集
    test_dataset = TestDataset()
    print(f"📊 测试数据集大小: {len(test_dataset)}")
    
    # 快速可视化
    print("🔄 生成快速可视化...")
    accuracy = quick_visualize(model, test_dataset, device, start=0, end=8)
    
    print(f"平均准确率: {accuracy:.3f}")


if __name__ == "__main__":
    main()
