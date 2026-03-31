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
from load_data import load_test_state, load_test_label, load_specific_test_state, load_specific_test_label, get_test_demo_id_list
import os
from config import resample, without_quat

plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

class TestDataset:
    def __init__(self):

        demonstrations_state = load_specific_test_state(shuffle=False, without_quat=without_quat, resample=resample, demo_id_list=[109,112,117])
        demonstrations_label = load_specific_test_label(demo_id_list=[109,112,117])
        
        # demo_id_list = np.arange(148)
        # demo_id_list = np.delete(demo_id_list, [80, 81, 92, 109, 112, 117, 122, 144, 145])
        # test_demo_id_list = get_test_demo_id_list(demo_id_list)
        # self.demo_ids = list(test_demo_id_list)   # 保存每条序列对应的原始 demo_id
        # demonstrations_state = load_test_state(without_quat=without_quat, resample=resample, demo_id_list=demo_id_list)
        # demonstrations_label = load_test_label(resample=resample, demo_id_list=demo_id_list)
        
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


def visualize_sequence_predictions(model, test_dataset, device, num_sequences=7, save_path=None):
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
    class_colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7', '#DDA15E', '#BA68C8', '#A9A2B5']
    class_names = ['Class 0', 'Class 1', 'Class 2', 'Class 3', 'Class 4', 'Class 5', 'Class 6', 'Unlabeled']
    
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
        demo_id = test_dataset.demo_ids[seq_idx] if hasattr(test_dataset, 'demo_ids') else seq_idx
        if len(valid_true_labels) > 0:
            accuracy = np.mean(valid_true_labels == valid_predicted_labels)
            valid_count = len(valid_true_labels)
            total_count = len(true_labels)
            title = f'Seq {seq_idx + 1}  [demo_id={demo_id}]  Accuracy: {accuracy:.3f}  (Valid: {valid_count}/{total_count})'
        else:
            accuracy = 0.0
            title = f'Seq {seq_idx + 1}  [demo_id={demo_id}]  No valid labels'
        
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
    plt.close('all')  # 关闭所有图形，避免空白窗口


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
    plt.close(fig)  # 关闭图形，避免空白窗口


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
    plt.close(fig)  # 关闭图形，避免空白窗口


def lstm_main():
    """主函数 - LSTM 可视化"""
    
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
    all_seq_indices = list(range(6))
    visualize_predicted_class_probabilities(
        model, test_dataset, device,
        seq_idx=all_seq_indices,
        save_path=os.path.join(save_dir, "coarse_fine_probabilities.png")
    )
    
    # 1. 可视化序列预测对比
    print("\n生成序列预测对比图...")
    visualize_sequence_predictions(
        model, test_dataset, device, 
        num_sequences=6, 
        save_path=os.path.join(save_dir, "sequence_predictions.png")
    )
    
    # 2. 绘制混淆矩阵
    print("\n生成混淆矩阵...")
    visualize_confusion_matrix(
        model, test_dataset, device,
        save_path=os.path.join(save_dir, "confusion_matrix.png")
    )


# ── TeCNO 可视化 ─────────────────────────────────────────────────────────

def load_tecno_model(filepath, device='cpu'):
    """加载 TeCNO 模型（不影响 LSTM 的 load_model）"""
    from TeCNO_seg_train import TeCNO
    checkpoint = torch.load(filepath, map_location=device)
    cfg = checkpoint['model_config']
    model = TeCNO(
        cfg['input_size'], cfg['hidden_size'],
        cfg['num_layers'], cfg['num_classes'],
        num_stages=cfg.get('num_stages', 2),
        kernel_size=cfg.get('kernel_size', 3),
        dropout=cfg.get('dropout', 0.5),
    )
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    return model


def visualize_tecno_per_phase_probs(model, test_dataset, device,
                                    seq_idx=0, save_path=None):
    """
    TeCNO 专用：绘制每个阶段的 softmax 概率曲线 + 真实标签色带。
    与 coarse/fine 互补，直接展示 7 条概率曲线。

    seq_idx: int 或 list[int]
    """
    CLASS_NAMES = [
        'P0 Right Move', 'P1 Pick Needle', 'P2 Right Move2',
        'P3 Pass Needle', 'P4 Left Move', 'P5 Left Pick', 'P6 Pull Thread',
    ]
    PHASE_COLORS = [
        '#1abc9c', '#3498db', '#9b59b6',
        '#e67e22', '#e74c3c', '#2ecc71', '#f1c40f',
    ]
    STAGE_BG = {
        0: '#A8D5E2', 1: '#F4A9A8', 2: '#A8C5DA',
        3: '#F7C5A0', 4: '#B5D5C5', 5: '#D5A8D4',
        6: '#C5D5A8', -1: '#E0E0E0',
    }

    if isinstance(seq_idx, int):
        seq_indices = [seq_idx]
    else:
        seq_indices = list(seq_idx)

    model.eval()
    all_probs, all_true = [], []
    with torch.no_grad():
        for idx in seq_indices:
            sequence, true_labels = test_dataset[idx]
            logits = model(sequence.unsqueeze(0).to(device), [sequence.shape[0]])
            probs = torch.softmax(logits, dim=2).squeeze(0).cpu().numpy()
            all_probs.append(probs)
            all_true.append(true_labels.numpy())

    n = len(seq_indices)
    fig, axes = plt.subplots(n, 1, figsize=(16, 4.5 * n), squeeze=False)

    from matplotlib.patches import Patch
    for row, (idx, probs, true_lbl) in enumerate(
            zip(seq_indices, all_probs, all_true)):
        ax = axes[row, 0]
        T = probs.shape[0]
        t = np.arange(T)

        # background: true stage bands
        prev, start = true_lbl[0], 0
        for i in range(1, T):
            if true_lbl[i] != prev:
                ax.axvspan(start - 0.5, i - 0.5, alpha=0.30,
                           color=STAGE_BG.get(int(prev), '#DDD'), linewidth=0)
                start, prev = i, true_lbl[i]
        ax.axvspan(start - 0.5, T - 0.5, alpha=0.30,
                   color=STAGE_BG.get(int(prev), '#DDD'), linewidth=0)

        # probability curves
        for ci in range(probs.shape[1]):
            ax.plot(t, probs[:, ci], color=PHASE_COLORS[ci],
                    linewidth=1.8, alpha=0.85, label=CLASS_NAMES[ci])

        # predicted label (argmax) as thin color bar at bottom
        pred = probs.argmax(axis=1)
        for ci in range(len(CLASS_NAMES)):
            mask = pred == ci
            if mask.any():
                ax.fill_between(t, -0.06, -0.01, where=mask,
                                color=PHASE_COLORS[ci], alpha=0.9, step='mid')

        ax.set_xlim(-0.5, T - 0.5)
        ax.set_ylim(-0.08, 1.05)
        ax.set_ylabel('Probability', fontsize=10)
        demo_id = test_dataset.demo_ids[idx] if hasattr(test_dataset, 'demo_ids') else idx
        ax.set_title(f'TeCNO Per-Phase Probabilities — Seq {idx}  [demo_id={demo_id}]',
                     fontsize=11, fontweight='bold')
        ax.grid(True, alpha=0.25, axis='y')
        if row == n - 1:
            ax.set_xlabel('Time Step', fontsize=10)
        if row == 0:
            ax.legend(loc='upper right', fontsize=8, ncol=4, framealpha=0.85)

    fig.suptitle('TeCNO — Per-Phase Softmax Probabilities', fontsize=13, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.97])

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"TeCNO per-phase probability visualization saved to {save_path}")
    plt.show()
    plt.close(fig)


def tecno_main():
    """TeCNO 可视化主函数（复用 LSTM 可视化函数 + TeCNO 专属图表）"""

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")

    dir_path = os.path.dirname(__file__)
    model_path = os.path.join(dir_path, "TeCNO_model", "tecno_sequence_model.pth")
    if not os.path.exists(model_path):
        print(f"模型文件不存在: {model_path}")
        return

    print("加载 TeCNO 模型...")
    model = load_tecno_model(model_path, device)
    print("模型加载成功")

    test_dataset = TestDataset()
    print(f"测试数据集大小: {len(test_dataset)}")

    save_dir = os.path.join(dir_path, "TeCNO_visualization_results")
    os.makedirs(save_dir, exist_ok=True)

    all_seq_indices = list(range(min(6, len(test_dataset))))

    # 1. TeCNO 专属：逐阶段概率曲线
    print("\n生成 TeCNO 逐阶段概率可视化...")
    visualize_tecno_per_phase_probs(
        model, test_dataset, device,
        seq_idx=all_seq_indices,
        save_path=os.path.join(save_dir, "per_phase_probabilities.png"),
    )

    # 2. 复用：粗大/精细操作概率
    print("\n生成粗大/精细操作概率可视化...")
    visualize_predicted_class_probabilities(
        model, test_dataset, device,
        seq_idx=all_seq_indices,
        save_path=os.path.join(save_dir, "coarse_fine_probabilities.png"),
    )

    # 3. 复用：序列预测对比
    print("\n生成序列预测对比图...")
    visualize_sequence_predictions(
        model, test_dataset, device,
        num_sequences=min(6, len(test_dataset)),
        save_path=os.path.join(save_dir, "sequence_predictions.png"),
    )

    # 4. 复用：混淆矩阵
    print("\n生成混淆矩阵...")
    visualize_confusion_matrix(
        model, test_dataset, device,
        save_path=os.path.join(save_dir, "confusion_matrix.png"),
    )

    print(f"\n所有 TeCNO 可视化完成！结果保存在 {save_dir}")

def _draw_prob_subplot(ax, probs, true_labels_np, seq_idx,
                       coarse_classes, fine_classes, CLASS_NAMES, STAGE_COLORS,
                       show_xlabel=False, show_legend_bands=False, show_legend_lines=False,
                       demo_id=None):
    """Draw a single probability subplot onto ax. Returns (band_handles, band_labels)."""
    seq_len    = probs.shape[0]
    time_steps = np.arange(seq_len)

    coarse_prob = probs[:, coarse_classes].sum(axis=1)
    fine_prob   = probs[:, fine_classes  ].sum(axis=1)

    # --- background stage bands ---
    prev_label, span_start, spans = true_labels_np[0], 0, []
    for t in range(1, seq_len):
        if true_labels_np[t] != prev_label:
            spans.append((span_start, t - 1, prev_label))
            span_start, prev_label = t, true_labels_np[t]
    spans.append((span_start, seq_len - 1, prev_label))

    drawn_labels = set()
    band_handles, band_labels = [], []
    for (s, e, lbl) in spans:
        color    = STAGE_COLORS.get(int(lbl), '#DDDDDD')
        lbl_name = CLASS_NAMES[int(lbl)] if 0 <= lbl < len(CLASS_NAMES) else 'Unlabeled'
        patch = ax.axvspan(s - 0.5, e + 0.5, alpha=0.45, color=color, linewidth=0)
        if lbl not in drawn_labels:
            band_handles.append(patch)
            band_labels.append(lbl_name.replace('\n', ' '))
            drawn_labels.add(lbl)

    # --- probability curves ---
    l_coarse, = ax.plot(time_steps, coarse_prob,
                        color='#1A6FA8', linewidth=2.0,
                        label='P(Coarse) – 0,2,4,6', zorder=3)
    l_fine,   = ax.plot(time_steps, fine_prob,
                        color='#C0392B', linewidth=2.0, linestyle='--',
                        label='P(Fine) – 1,3,5', zorder=3)

    ax.set_xlim(-0.5, seq_len - 0.5)
    ax.set_ylim(-0.03, 1.08)
    ax.set_ylabel('Probability', fontsize=10)
    demo_label = f'  [demo_id={demo_id}]' if demo_id is not None else ''
    ax.set_title(f'Sequence {seq_idx}{demo_label}', fontsize=11, fontweight='bold')
    ax.grid(True, alpha=0.25, axis='y')

    if show_xlabel:
        ax.set_xlabel('Time Step', fontsize=10)

    if show_legend_lines:
        ax.legend([l_coarse, l_fine],
                  ['P(Coarse) – 0,2,4,6', 'P(Fine) – 1,3,5'],
                  loc='upper right', fontsize=9, framealpha=0.85)

    return band_handles, band_labels


def visualize_predicted_class_probabilities(model, test_dataset, device,
                                            seq_idx=0, save_path=None):
    """
    Visualize coarse vs fine operation probabilities for one or multiple sequences.

    seq_idx: int  → single sequence
             list[int] → one subplot per sequence, arranged in a column

    Coarse operations: classes 0, 2, 4, 6  (Move / Pull – gross motions)
    Fine   operations: classes 1, 3, 5     (Pick / Pass / LeftPick – precision motions)
    """
    # ── class metadata ────────────────────────────────────────────────────────
    CLASS_NAMES = [
        'P0 Right\nMove', 'P1 Pick\nNeedle', 'P2 Right\nMove2',
        'P3 Pass\nNeedle', 'P4 Left\nMove',  'P5 Left\nPick',
        'P6 Pull\nThread',
    ]
    STAGE_COLORS = {
        0: '#A8D5E2', 1: '#F4A9A8', 2: '#A8C5DA',
        3: '#F7C5A0', 4: '#B5D5C5', 5: '#D5A8D4',
        6: '#C5D5A8', -1: '#E0E0E0',
    }
    coarse_classes = [0, 2, 4, 6]
    fine_classes   = [1, 3, 5]

    # normalise seq_idx to a list
    if isinstance(seq_idx, int):
        seq_indices = [seq_idx]
    else:
        seq_indices = list(seq_idx)

    n = len(seq_indices)

    # ── inference for all requested sequences ────────────────────────────────
    model.eval()
    if hasattr(model, 'crf'):
        print("Warning: CRF model does not provide per-step probabilities. Skipping.")
        return

    all_probs, all_true = [], []
    with torch.no_grad():
        for idx in seq_indices:
            sequence, true_labels = test_dataset[idx]
            sequence = sequence.to(device)
            logits   = model(sequence.unsqueeze(0), [sequence.shape[0]])  # (1,T,C)
            probs    = torch.softmax(logits, dim=2).squeeze(0).cpu().numpy()
            all_probs.append(probs)
            all_true.append(true_labels.numpy())

    # ── figure layout ─────────────────────────────────────────────────────────
    fig, axes = plt.subplots(n, 1, figsize=(16, 4 * n),
                             squeeze=False)  # always 2-D array

    # 取出每个序列对应的真实 demo_id（如果 dataset 有记录）
    def _get_demo_id(dataset, idx):
        if hasattr(dataset, 'demo_ids') and idx < len(dataset.demo_ids):
            return dataset.demo_ids[idx]
        return None

    all_band_handles, all_band_labels = [], []
    for row, (idx, probs, true_labels_np) in enumerate(
            zip(seq_indices, all_probs, all_true)):
        ax = axes[row, 0]
        is_last = (row == n - 1)
        bh, bl = _draw_prob_subplot(
            ax, probs, true_labels_np, idx,
            coarse_classes, fine_classes, CLASS_NAMES, STAGE_COLORS,
            show_xlabel=is_last,
            show_legend_bands=False,       # handled globally below
            show_legend_lines=(row == 0),  # show curve legend only on first panel
            demo_id=_get_demo_id(test_dataset, idx),
        )
        # collect unique band legend entries across all subplots
        for h, l in zip(bh, bl):
            if l not in all_band_labels:
                all_band_handles.append(h)
                all_band_labels.append(l)

    # ── shared figure-level legends ───────────────────────────────────────────
    # Stage-color legend below the figure
    fig.legend(all_band_handles, all_band_labels,
               loc='lower center', ncol=min(7, len(all_band_labels)),
               fontsize=9, title='True Stage',
               bbox_to_anchor=(0.5, 0.0), framealpha=0.9)

    demo_ids_str = ([str(_get_demo_id(test_dataset, i)) for i in seq_indices])
    title_suffix = (f'Seq {seq_indices[0]} [demo_id={demo_ids_str[0]}]' if n == 1
                    else f'demo_ids={demo_ids_str}')
    fig.suptitle(f'Coarse vs Fine Operation Probabilities – {title_suffix}',
                 fontsize=13, fontweight='bold')

    plt.tight_layout(rect=[0, 0.06, 1, 0.97])

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Probability visualization saved to {save_path}")

    plt.show()
    plt.close(fig)



if __name__ == "__main__":
    
    #lstm_main()
    
    tecno_main()
