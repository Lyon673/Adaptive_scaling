"""
序列预测标签可视化工具
用于展示真实标签和预测标签的对比分析
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.colors import ListedColormap
import seaborn as sns
from torch.utils.data import DataLoader
from LSTM_seg_train import SequenceLabelingLSTM, SequenceLabelingLSTM_CRF, collate_fn
from load_data import load_test_state, load_test_label
import os

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

class TestDataset:
    def __init__(self):
        demonstrations_state = load_test_state()
        demonstrations_label = load_test_label()
        
        
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


def visualize_sequence_predictions(model, test_dataset, device, num_sequences=5, save_path=None):
    """
    可视化序列预测结果
    
    参数:
    model: 训练好的模型
    test_dataset: 测试数据集
    device: 设备
    num_sequences: 要可视化的序列数量
    save_path: 保存图片的路径
    
    注意: 如果true_label是-1，则跳过该帧的预测和准确率计算
    """
    
    # 定义颜色和标签 (6个有效类 + 1个未标注类)
    class_colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7', '#DDA15E', '#A9A9A9']
    class_names = ['Class 0', 'Class 1', 'Class 2', 'Class 3', 'Class 4', 'Class 5', 'Unlabeled']
    
    # 创建图形
    fig, axes = plt.subplots(num_sequences, 1, figsize=(15, 3*num_sequences))
    if num_sequences == 1:
        axes = [axes]
    
    for seq_idx in range(num_sequences):
        ax = axes[seq_idx]
        
        # 获取序列数据
        sequence, true_labels = test_dataset[seq_idx]
        
        # 预测标签
        predicted_labels = predict_sequence(model, sequence, device)
        
        # 确保predicted_labels是numpy数组
        if isinstance(predicted_labels, list):
            predicted_labels = np.array(predicted_labels)
        
        # 确保预测标签长度与真实标签一致
        min_len = min(len(true_labels), len(predicted_labels))
        true_labels = true_labels[:min_len].numpy()
        predicted_labels = predicted_labels[:min_len]
        
        # 创建时间轴
        time_steps = np.arange(min_len)
        
        # 过滤掉label为-1的帧（用于准确率计算）
        valid_mask = true_labels != -1
        valid_time_steps = time_steps[valid_mask]
        valid_true_labels = true_labels[valid_mask]
        valid_predicted_labels = predicted_labels[valid_mask]
        
        # 绘制真实标签（包括-1的标签，用于显示）
        # range(6) 包括 Class 0-5，不包括Unlabeled
        for i in range(len(class_names) - 1):  # 0-5，不包括Unlabeled
            mask = true_labels == i
            if np.any(mask):
                ax.scatter(time_steps[mask], np.ones(np.sum(mask)) * 1.1, 
                          c=class_colors[i], label=f'True {class_names[i]}', 
                          s=50, alpha=0.8, marker='o')
        
        # 绘制label=-1的帧（用灰色显示）
        unlabeled_mask = true_labels == -1
        if np.any(unlabeled_mask):
            ax.scatter(time_steps[unlabeled_mask], np.ones(np.sum(unlabeled_mask)) * 1.1, 
                      c=class_colors[-1], label='True Unlabeled (-1)', 
                      s=50, alpha=0.5, marker='x')
        
        # 绘制预测标签（只绘制有效标签对应的预测）
        # range(6) 包括 Class 0-5，不包括Unlabeled
        for i in range(len(class_names) - 1):  # 0-5，不包括Unlabeled
            mask = valid_predicted_labels == i
            if np.any(mask):
                ax.scatter(valid_time_steps[mask], np.ones(np.sum(mask)) * 0.9, 
                          c=class_colors[i], label=f'Pred {class_names[i]}', 
                          s=30, alpha=0.6, marker='^')
        
        # 计算准确率（只计算有效标签）
        if len(valid_true_labels) > 0:
            accuracy = np.mean(valid_true_labels == valid_predicted_labels)
            valid_count = len(valid_true_labels)
            total_count = len(true_labels)
            title = f'Sequence {seq_idx + 1} - Accuracy: {accuracy:.3f} (Valid: {valid_count}/{total_count})'
        else:
            accuracy = 0.0
            title = f'Sequence {seq_idx + 1} - No valid labels'
        
        # 设置图形属性
        ax.set_ylim(0.5, 1.5)
        ax.set_xlabel('Time Steps')
        ax.set_title(title)
        ax.set_yticks([0.9, 1.1])
        ax.set_yticklabels(['Predicted', 'True'])
        ax.grid(True, alpha=0.3)
        
        # 添加图例（只在第一个子图添加）
        if seq_idx == 0:
            ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"可视化结果已保存到: {save_path}")
    
    plt.show()


def visualize_confusion_matrix(model, test_dataset, device, save_path=None):
    """
    绘制混淆矩阵
    
    注意: 跳过label为-1的帧，只计算有效标签
    """
    from sklearn.metrics import confusion_matrix
    
    # 收集所有预测和真实标签
    all_true = []
    all_pred = []
    
    for i in range(len(test_dataset)):
        sequence, true_labels = test_dataset[i]
        predicted_labels = predict_sequence(model, sequence, device)
        
        # 确保predicted_labels是numpy数组
        if isinstance(predicted_labels, list):
            predicted_labels = np.array(predicted_labels)
        
        min_len = min(len(true_labels), len(predicted_labels))
        true_labels_np = true_labels[:min_len].numpy()
        predicted_labels_np = predicted_labels[:min_len]
        
        # 过滤掉label为-1的帧
        valid_mask = true_labels_np != -1
        valid_true = true_labels_np[valid_mask]
        valid_pred = predicted_labels_np[valid_mask]
        
        all_true.extend(valid_true)
        all_pred.extend(valid_pred)
    
    # 计算混淆矩阵
    cm = confusion_matrix(all_true, all_pred)
    
    # 绘制混淆矩阵
    plt.figure(figsize=(12, 10))
    # 动态确定类别数量
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


def visualize_sequence_timeline(model, test_dataset, device, seq_idx=0, save_path=None):
    """
    可视化单个序列的时间线，展示标签变化和预测准确性
    
    注意: label为-1的帧会被标记为未标注，不计入准确率
    """
    # 获取序列数据
    sequence, true_labels = test_dataset[seq_idx]
    predicted_labels = predict_sequence(model, sequence, device)
    
    # 确保predicted_labels是numpy数组
    if isinstance(predicted_labels, list):
        predicted_labels = np.array(predicted_labels)
    
    min_len = min(len(true_labels), len(predicted_labels))
    true_labels = true_labels[:min_len].numpy()
    predicted_labels = predicted_labels[:min_len]
    time_steps = np.arange(min_len)
    
    # 创建图形
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(15, 10))
    
    # 定义颜色 (6个有效类 + 1个未标注类)
    class_colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7', '#DDA15E', '#A9A9A9']
    
    # 1. 真实标签时间线（包括Class 0-5）
    for i in range(6):  # Class 0-5
        mask = true_labels == i
        if np.any(mask):
            ax1.bar(time_steps[mask], np.ones(np.sum(mask)), 
                   color=class_colors[i], alpha=0.7, label=f'Class {i}')
    
    # 显示未标注的帧
    unlabeled_mask = true_labels == -1
    if np.any(unlabeled_mask):
        ax1.bar(time_steps[unlabeled_mask], np.ones(np.sum(unlabeled_mask)), 
               color=class_colors[6], alpha=0.5, label='Unlabeled (-1)')  # 使用第7个颜色（索引6）
    
    ax1.set_title(f'True Labels Timeline - Sequence {seq_idx + 1}')
    ax1.set_ylabel('True Labels')
    ax1.set_ylim(0, 1)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. 预测标签时间线（只显示有效标签对应的预测）
    valid_mask = true_labels != -1
    valid_time_steps = time_steps[valid_mask]
    valid_predicted_labels = predicted_labels[valid_mask]
    
    # 2. 预测标签时间线（包括Class 0-5）
    for i in range(6):  # Class 0-5
        mask = valid_predicted_labels == i
        if np.any(mask):
            ax2.bar(valid_time_steps[mask], np.ones(np.sum(mask)), 
                   color=class_colors[i], alpha=0.7, label=f'Class {i}')
    
    ax2.set_title(f'Predicted Labels Timeline - Sequence {seq_idx + 1}')
    ax2.set_ylabel('Predicted Labels')
    ax2.set_ylim(0, 1)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. 预测准确性（只计算有效标签）
    valid_true_labels = true_labels[valid_mask]
    accuracy_per_step = np.full(min_len, np.nan)  # 使用NaN表示未标注的帧
    accuracy_per_step[valid_mask] = (valid_true_labels == valid_predicted_labels).astype(float)
    
    # 绘制每步准确性（只显示有效标签的准确性）
    ax3.plot(valid_time_steps, accuracy_per_step[valid_mask], 
             color='red', alpha=0.5, label='Per-step Accuracy', linewidth=1)
    
    # 标记未标注的帧
    if np.any(~valid_mask):
        ax3.scatter(time_steps[~valid_mask], np.zeros(np.sum(~valid_mask)), 
                   color='gray', alpha=0.3, s=10, label='Unlabeled frames', marker='x')
    
    # 计算滑动窗口准确率（只在有效标签上计算）
    window_size = 50
    if len(valid_time_steps) > window_size:
        valid_accuracy = accuracy_per_step[valid_mask]
        sliding_accuracy = np.convolve(valid_accuracy, 
                                     np.ones(window_size)/window_size, mode='valid')
        sliding_time = valid_time_steps[window_size-1:]
        ax3.plot(sliding_time, sliding_accuracy, 
                color='green', linewidth=2, label=f'Sliding Accuracy (window={window_size})')
    
    ax3.set_title(f'Prediction Accuracy Over Time - Sequence {seq_idx + 1}')
    ax3.set_xlabel('Time Steps')
    ax3.set_ylabel('Accuracy')
    ax3.set_ylim(-0.1, 1.1)
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 添加整体准确率文本（只计算有效标签）
    if len(valid_true_labels) > 0:
        overall_accuracy = np.mean(valid_true_labels == valid_predicted_labels)
        valid_ratio = f'{len(valid_true_labels)}/{min_len}'
        ax3.text(0.02, 0.98, f'Overall Accuracy: {overall_accuracy:.3f}\nValid frames: {valid_ratio}', 
                 transform=ax3.transAxes, fontsize=12, 
                 verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    else:
        ax3.text(0.02, 0.98, 'No valid labels for accuracy calculation', 
                 transform=ax3.transAxes, fontsize=12, 
                 verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"时间线可视化已保存到: {save_path}")
    
    plt.show()


def visualize_feature_importance(model, test_dataset, device, seq_idx=0, save_path=None):
    """
    可视化特征重要性（通过观察不同特征的分布）
    
    注意: label为-1的帧会用灰色标记为未标注
    """
    sequence, true_labels = test_dataset[seq_idx]
    
    # 确保true_labels是numpy数组
    if isinstance(true_labels, torch.Tensor):
        true_labels = true_labels.numpy()
    
    # 获取特征数据
    features = sequence.numpy()  # shape: (seq_len, feature_dim)
    
    # 创建图形
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    axes = axes.flatten()
    
    feature_names = [f'Feature {i}' for i in range(features.shape[1])]
    
    for i in range(min(4, features.shape[1])):  # 只显示前4个特征
        ax = axes[i]
        
        # 绘制特征值随时间的变化
        time_steps = np.arange(features.shape[0])
        
        # 根据真实标签给不同颜色（只显示有效标签）
        # 定义与其他函数一致的颜色
        feature_colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7', '#DDA15E']
        for label in range(6):  # Class 0-5
            mask = true_labels == label
            if np.any(mask):
                ax.scatter(time_steps[mask], features[mask, i], 
                          c=feature_colors[label], alpha=0.6, s=20, label=f'Class {label}')
        
        # 标记未标注的帧
        unlabeled_mask = true_labels == -1
        if np.any(unlabeled_mask):
            ax.scatter(time_steps[unlabeled_mask], features[unlabeled_mask, i], 
                      c='gray', alpha=0.3, s=10, marker='x', label='Unlabeled (-1)')
        
        ax.set_title(f'{feature_names[i]} Over Time')
        ax.set_xlabel('Time Steps')
        ax.set_ylabel('Feature Value')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"特征可视化已保存到: {save_path}")
    
    plt.show()


def main():
    """主函数 - 运行所有可视化"""
    
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    # 加载模型
    dir_path = os.path.dirname(__file__)
    model_path = os.path.join(dir_path, "LSTM_model", "lstm_sequence_model.pth")
    if not os.path.exists(model_path):
        print(f"模型文件不存在: {model_path}")
        return
    
    print("加载模型...")
    model = load_model(model_path, device)
    print("模型加载成功")
    
    # 创建测试数据集
    test_dataset = TestDataset()
    print(f"测试数据集大小: {len(test_dataset)}")
    
    # 创建保存目录
    save_dir = os.path.join(dir_path, "LSTM_visualization_results")  
    os.makedirs(save_dir, exist_ok=True)

    print("\n生成粗大/精细操作概率可视化...")
    visualize_predicted_class_probabilities(
        model, test_dataset, device,
        seq_idx=0,
        save_path=os.path.join(save_dir, "coarse_fine_probabilities.png")
    )
    
    # 1. 可视化序列预测对比
    print("\n生成序列预测对比图...")
    visualize_sequence_predictions(
        model, test_dataset, device, 
        num_sequences=5, 
        save_path=os.path.join(save_dir, "sequence_predictions.png")
    )
    
    # 2. 绘制混淆矩阵
    print("\n生成混淆矩阵...")
    visualize_confusion_matrix(
        model, test_dataset, device,
        save_path=os.path.join(save_dir, "confusion_matrix.png")
    )
    
    # 3. 可视化单个序列的时间线
    print("\n生成时间线可视化...")
    visualize_sequence_timeline(
        model, test_dataset, device, 
        seq_idx=0,
        save_path=os.path.join(save_dir, "sequence_timeline.png")
    )
    
    # 4. 可视化特征重要性
    print("\n生成特征可视化...")
    visualize_feature_importance(
        model, test_dataset, device, 
        seq_idx=0,
        save_path=os.path.join(save_dir, "feature_analysis.png")
    )
    

    
    print(f"\n所有可视化完成！结果保存在 {save_dir} 目录中")

def visualize_predicted_class_probabilities(model, test_dataset, device, seq_idx=0, save_path=None):
    """
    Visualize coarse vs fine operation probabilities over time
    
    Coarse operations: classes 0, 1, 3, 5
    Fine operations: classes 2, 4
    
    Args:
        model: Trained model
        test_dataset: Test dataset
        device: Device
        seq_idx: Sequence index to visualize
        save_path: Path to save figure, if None then show
    """
    model.eval()
    
    # Get sequence data
    sequence, true_labels = test_dataset[seq_idx]
    sequence = sequence.to(device)
    lengths = [sequence.shape[0]]
    
    with torch.no_grad():
        if hasattr(model, 'crf'):
            print("Warning: CRF model does not provide per-step probabilities. Skipping visualization.")
            return
        
        # Get logits and probabilities
        logits = model(sequence.unsqueeze(0), lengths)  # (1, seq_len, num_classes)
        probs = torch.softmax(logits, dim=2).squeeze(0).cpu().numpy()  # (seq_len, num_classes)
    
    # Define operation categories
    coarse_classes = [0, 1, 3, 5]  # Coarse operations
    fine_classes = [2, 4]           # Fine operations
    
    # Calculate probabilities for each category
    coarse_prob = probs[:, coarse_classes].sum(axis=1)
    fine_prob = probs[:, fine_classes].sum(axis=1)
    
    seq_len = len(coarse_prob)
    time_steps = np.arange(seq_len)
    true_labels_np = true_labels.numpy()
    
    # Create figure with subplots
    fig, axes = plt.subplots(3, 1, figsize=(15, 10))
    
    # ========== Subplot 1: Probability curves ==========
    ax1 = axes[0]
    ax1.plot(time_steps, coarse_prob, label='Coarse Operations (0,1,3,5)', 
             color='#3498db', linewidth=2, alpha=0.8)
    ax1.plot(time_steps, fine_prob, label='Fine Operations (2,4)', 
             color='#e74c3c', linewidth=2, alpha=0.8)
    ax1.fill_between(time_steps, 0, coarse_prob, alpha=0.2, color='#3498db')
    ax1.fill_between(time_steps, 0, fine_prob, alpha=0.2, color='#e74c3c')
    
    ax1.set_ylabel('Probability', fontsize=12)
    ax1.set_title(f'Sequence {seq_idx}: Coarse vs Fine Operation Probabilities', 
                  fontsize=14, fontweight='bold')
    ax1.legend(loc='upper right', fontsize=11)
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(-1, seq_len)
    ax1.set_ylim(-0.05, 1.05)
    
    # ========== Subplot 2: True labels background ==========
    ax2 = axes[1]
    
    # Draw background colors for true labels
    for i in range(seq_len):
        if true_labels_np[i] == -1:
            color = '#D3D3D3'  # Gray for unlabeled
            label_type = 'Unlabeled'
        elif true_labels_np[i] in coarse_classes:
            color = '#AED6F1'  # Light blue for coarse
            label_type = 'Coarse'
        elif true_labels_np[i] in fine_classes:
            color = '#F5B7B1'  # Light red for fine
            label_type = 'Fine'
        else:
            color = '#E8E8E8'
            label_type = 'Unknown'
        
        ax2.axvspan(i-0.5, i+0.5, alpha=0.6, color=color)
    
    # Plot probability difference (coarse - fine)
    prob_diff = coarse_prob - fine_prob
    ax2.plot(time_steps, prob_diff, color='#2C3E50', linewidth=2, label='Prob(Coarse) - Prob(Fine)')
    ax2.axhline(y=0, color='black', linestyle='--', linewidth=1, alpha=0.5)
    
    ax2.set_ylabel('Probability Difference', fontsize=12)
    ax2.set_title('Probability Difference (Coarse - Fine) with True Label Background', 
                  fontsize=13, fontweight='bold')
    ax2.legend(loc='upper right', fontsize=11)
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(-1, seq_len)
    ax2.set_ylim(-1.05, 1.05)
    
    # ========== Subplot 3: Stacked area chart ==========
    ax3 = axes[2]
    
    # Show individual class probabilities as stacked areas
    class_colors_fine = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7', '#DDA15E']
    
    # Reorder: coarse first, then fine
    class_order = coarse_classes + fine_classes
    prob_stacked = probs[:, class_order].T  # (6, seq_len)
    
    ax3.stackplot(time_steps, prob_stacked, 
                  labels=[f'Class {c}' for c in class_order],
                  colors=[class_colors_fine[c] for c in class_order],
                  alpha=0.7)
    
    # Add separator line between coarse and fine
    coarse_cumsum = probs[:, coarse_classes].sum(axis=1)
    ax3.plot(time_steps, coarse_cumsum, 'k--', linewidth=2, alpha=0.8, 
             label='Coarse/Fine Boundary')
    
    ax3.set_xlabel('Time Step', fontsize=12)
    ax3.set_ylabel('Cumulative Probability', fontsize=12)
    ax3.set_title('Individual Class Probability Distribution (Stacked)', 
                  fontsize=13, fontweight='bold')
    ax3.legend(loc='upper left', fontsize=9, ncol=2)
    ax3.grid(True, alpha=0.3)
    ax3.set_xlim(-1, seq_len)
    ax3.set_ylim(0, 1.0)
    
    # Overall title
    fig.suptitle(f'Operation Type Probability Analysis - Sequence {seq_idx}', 
                 fontsize=16, fontweight='bold', y=0.995)

    plt.show()
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Probability visualization saved to {save_path}")



if __name__ == "__main__":
    main()
