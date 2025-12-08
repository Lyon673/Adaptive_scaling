import torch
from torch.utils.data import Dataset, DataLoader
from LSTM_seg_train import SequenceLabelingLSTM, SequenceLabelingLSTM_CRF, NUM_CLASSES, BATCH_SIZE, collate_fn
from load_data import load_test_state, load_test_label
from sklearn.metrics import classification_report, accuracy_score
from tqdm import tqdm # 引入tqdm来显示进度条，因为这个过程会慢很多
import os
from torchcrf import CRF

# --- 模型加载功能 ---
def load_model(filepath, device='cpu'):
    """
    从文件加载训练好的模型
    
    参数:
    filepath: 模型文件路径
    device: 设备 ('cpu' 或 'cuda')
    
    返回:
    model: 加载的模型
    """
    # 加载保存的数据
    checkpoint = torch.load(filepath, map_location=device)
    
    # 获取模型配置
    model_config = checkpoint['model_config']
    
    # 检查模型类型并创建相应的模型实例
    # 如果状态字典包含CRF相关的键，则使用CRF模型
    if any(key.startswith('crf.') for key in checkpoint['model_state_dict'].keys()):
        print("检测到CRF模型，使用 SequenceLabelingLSTM_CRF")
        model = SequenceLabelingLSTM_CRF(
            input_size=model_config['input_size'],
            hidden_size=model_config['hidden_size'],
            num_layers=model_config['num_layers'],
            num_classes=model_config['num_classes']
        )
    else:
        print("检测到标准LSTM模型，使用 SequenceLabelingLSTM")
        model = SequenceLabelingLSTM(
            input_size=model_config['input_size'],
            hidden_size=model_config['hidden_size'],
            num_layers=model_config['num_layers'],
            num_classes=model_config['num_classes']
        )
    
    # 加载模型权重
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # 移动到指定设备
    model.to(device)
    
    # 设置评估模式
    model.eval()
    
    print(f"模型已从 {filepath} 加载")
    print(f"模型配置: {model_config}")
    if 'training_info' in checkpoint:
        print(f"训练信息: {checkpoint['training_info']}")
    
    return model


class TestDataset(Dataset):
    def __init__(self):
        demonstrations_state = load_test_state()
        demonstrations_label = load_test_label()
        
        # 确保每个演示都是一个序列（2D数组）
        self.samples = []
        for state_seq, label_seq in zip(demonstrations_state, demonstrations_label):
            # 转换为张量
            state_tensor = torch.tensor(state_seq, dtype=torch.float32)
            label_tensor = torch.tensor(label_seq, dtype=torch.long)
            
            # 确保是2D张量 (seq_len, feature_size)
            if state_tensor.dim() == 1:
                state_tensor = state_tensor.unsqueeze(1)  # 如果是1D，添加特征维度
            
            
            self.samples.append((state_tensor, label_tensor))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]



# --- 3. 评估模型并计算指标 ---
def evaluate_model(model, dataloader, device):
    """
    在测试集上评估模型并打印分类报告。
    """
    model.eval()  # 将模型设置为评估模式
    all_preds = []
    all_labels = []

    with torch.no_grad():  # 在评估阶段不计算梯度，节省计算资源
        for sequences, labels, lengths in dataloader:
            sequences = sequences.to(device)
            labels = labels.to(device)

            # 前向传播获取预测
            outputs = model(sequences, lengths)
            
            # 获取概率最高的类别作为预测结果
            # outputs shape: (batch, seq_len, num_classes) -> preds shape: (batch, seq_len)
            preds = torch.argmax(outputs, dim=2)

            # !!! 关键步骤：过滤掉填充的标签 (-1) !!!
            # 我们只对真实的、非填充的数据点计算指标
            mask = labels != -1  # 创建一个布尔掩码，padding的位置是False
            
            # 使用掩码来选择非填充的预测和标签
            preds_unpadded = torch.masked_select(preds, mask)
            labels_unpadded = torch.masked_select(labels, mask)

            # 收集所有批次的非填充结果
            all_preds.extend(preds_unpadded.cpu().numpy())
            all_labels.extend(labels_unpadded.cpu().numpy())

    # --- 4. 计算并打印评估指标 ---
    if not all_labels:
        print("没有找到任何有效的标签进行评估。")
        return

    # 定义标签名称（可选，但能让报告更易读）
    # 请确保这里的顺序和你的 (str -> int) 映射是一致的
    # 例如：{ "准备": 0, "执行": 1, ... }
    target_names = [f'Class {i}' for i in range(NUM_CLASSES)]

    print("\n--- 模型评估报告 ---")
    print(classification_report(all_labels, all_preds, target_names=target_names, digits=4))
    
    # 也可以单独计算总体准确率
    accuracy = accuracy_score(all_labels, all_preds)
    print(f"Overall Accuracy: {accuracy:.4f}")
    print("---------------------\n")

def evaluate_model_CRF(model, dataloader, device):
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for sequences, labels, lengths in dataloader:
            sequences = sequences.to(device)
            labels = labels.to(device)

            # --- 修改：使用model.decode()进行预测 ---
            # model.decode返回的是一个列表的列表，每个子列表是可变长度的预测路径
            predicted_paths = model.decode(sequences, lengths)

            # --- 修改：处理预测结果和真实标签 ---
            # 过滤掉真实标签中的填充部分(-1)
            for i in range(len(lengths)):
                true_len = lengths[i]
                true_labels = labels[i, :true_len].cpu().numpy()
                
                # 将一个批次的所有真实标签和预测标签展平后加入总列表
                all_labels.extend(true_labels)
                all_preds.extend(predicted_paths[i])
    
    # --- 计算指标部分 (保持不变) ---
    if not all_labels:
        print("没有找到任何有效的标签进行评估。")
        return
    target_names = [f'Class {i}' for i in range(NUM_CLASSES)]
    print("\n--- 模型评估报告 ---")
    print(classification_report(all_labels, all_preds, target_names=target_names, digits=4))
    accuracy = accuracy_score(all_labels, all_preds)
    print(f"Overall Accuracy: {accuracy:.4f}")
    print("---------------------\n")


"""
    real-time simulation
"""

def get_prediction_for_current_step(model, history_data, device):
    """
    一个辅助函数，用于根据历史数据获取当前时刻的预测标签。
    这是在实时应用中每个时间步都会被调用的核心逻辑。

    参数:
    model (nn.Module): 训练好的模型。
    history_data (torch.Tensor): 从开始到当前时刻的完整数据，形状 (current_seq_len, input_size)。
    device: 'cuda' or 'cpu'
    
    返回:
    int: 预测的类别索引。
    """
    # 准备模型输入
    seq_len = len(history_data)
    # 增加batch维度，使其形状变为 (1, current_seq_len, input_size)
    input_tensor = history_data.unsqueeze(0).to(device)
    lengths = [seq_len]
    
    # 获取所有时间点的预测 logits
    # all_logits 的形状是 (1, current_seq_len, num_classes)
    all_logits = model(input_tensor, lengths)
    
    # 只取最后一个时间点的预测
    last_step_logits = all_logits[0, -1, :] # 形状 (num_classes,)
    
    # 获得最终的类别标签
    current_label = torch.argmax(last_step_logits).item()
    
    return current_label


def evaluate_model_real_time_simulation(model, test_dataset, device):
    """
    在测试集上模拟实时数据流进行评估，并打印分类报告。
    """
    print("\n--- start real-time simulation ---")

    model.eval()  # 将模型设置为评估模式
    all_preds = []
    all_labels = []

    # 使用 torch.no_grad() 来节约资源
    with torch.no_grad():
        # 1. 外层循环：遍历测试集中的每一个完整序列（demonstration）
        # 使用tqdm来显示进度
        for state_seq, label_seq in tqdm(test_dataset, desc="real-time simulation"):
            
            # 2. 内层循环：在单个序列内部，模拟时间步的推进
            for t in range(len(state_seq)):
                # b. 当前时间t的真实标签
                true_label = label_seq[t].item()
                
                # !!! 关键：跳过标签为-1的帧（准备阶段/空闲阶段） !!!
                if true_label == -1:
                    continue
                
                if t < 30:
                    all_preds.append(true_label)
                    all_labels.append(true_label)
                    continue
                # a. 截至当前时间t的历史数据
                history_data = state_seq[:t+1]
                
                # c. 调用模型，仅根据历史数据预测当前标签
                predicted_label = get_prediction_for_current_step(model, history_data, device)
                
                # d. 收集预测结果和真实标签（此时true_label必定不是-1）
                all_preds.append(predicted_label)
                all_labels.append(true_label)
                


    # --- 计算并打印评估指标 (这部分与您之前的代码完全相同) ---
    if not all_labels:
        print("没有找到任何有效的标签进行评估。")
        return

    target_names = [f'Class {i}' for i in range(NUM_CLASSES)]

    print("\n--- real-time simulation report ---")
    print(classification_report(all_labels, all_preds, target_names=target_names, digits=4))
    
    accuracy = accuracy_score(all_labels, all_preds)
    print(f"Overall Accuracy (Real-time Simulation): {accuracy:.4f}")
    print("---------------------\n")


# --- 运行新的实时模拟评估 ---
# 确保您的模型是单向的，并且已经训练完毕
# model = RealTimeClassifierLSTM(...)
# model.load_state_dict(...)

if __name__ == "__main__":      
    # 实例化测试数据集和DataLoader
    # 注意：测试时 shuffle 通常设置为 False
    test_dataset = TestDataset()
    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=BATCH_SIZE, # 可以使用和训练时一样的BATCH_SIZE
        shuffle=False,
        collate_fn=collate_fn, # 复用训练时的collate_fn
        drop_last=False
    )

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    dir_path = os.path.dirname(__file__)
    model_save_path = os.path.join(dir_path, "LSTM_model", "lstm_sequence_model.pth")
    model = load_model(model_save_path, device)
    # 运行新的评估函数
    evaluate_model(model, test_loader, device)
    # print("----------------------------------------------------------")
    #evaluate_model_real_time_simulation(model, test_dataset, device)

    # 您也可以运行原来的评估函数进行对比
    # 注意：理论上，两个评估函数的结果应该是完全一致的，因为它们都在计算每个时间点的 P/L

    
