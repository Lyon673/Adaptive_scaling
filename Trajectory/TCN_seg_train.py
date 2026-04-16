import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
import numpy as np
from load_data import (load_train_state, load_train_label,
                       load_demonstrations_state, _scale_demos,
                       get_shuffled_demo_ids,load_specific_test_state,load_specific_test_label)
import os
from torch.utils.tensorboard import SummaryWriter
import datetime
import time
from config import resample, without_quat
from torch.optim.lr_scheduler import StepLR

# --- 1. 定义超参数 (Hyperparameters) ---
INPUT_SIZE = 16       # 每个时间点的特征数
NUM_CLASSES = 7       # 标签的类别数量
# TCN 专属参数：列表的长度代表网络层数，列表里的数字代表每层的通道数（隐藏维度）
# [64, 64, 64, 64, 64] 代表 5 层，每层 64 通道。配合 kernel_size=3，感受野大约为 2^(5+1) = 64 帧
TCN_CHANNELS = [64, 128, 256, 256, 128] 
KERNEL_SIZE = 3
DROPOUT = 0.2

BATCH_SIZE = 16
NUM_EPOCHS = 800
LEARNING_RATE = 0.001

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# --- 2. 准备数据 (保持与原来完全一致) ---
class KinematicDataset(Dataset):
    def __init__(self, num_samples=100, is_train=True):
        self.samples = []
        demo_id_list = np.arange(148)
        demo_id_list = np.delete(demo_id_list, [80, 81, 92, 109, 112, 117, 122, 144, 145])

        demonstrations_state = load_train_state(without_quat=without_quat, resample=resample, demo_id_list=demo_id_list)
        demonstrations_label = load_train_label(resample=resample, demo_id_list=demo_id_list)
        
        for state_seq, label_seq in zip(demonstrations_state, demonstrations_label):
            if len(state_seq) == 0 or len(label_seq) == 0:
                continue
            state_tensor = torch.tensor(state_seq, dtype=torch.float32)
            label_tensor = torch.tensor(label_seq, dtype=torch.long)
            
            if state_tensor.dim() == 1:
                state_tensor = state_tensor.unsqueeze(1)
            self.samples.append((state_tensor, label_tensor))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]

class SmoothPhaseLoss(nn.Module):
    def __init__(self, num_classes=7, transition_weight=1.0, ignore_index=-1):
        super(SmoothPhaseLoss, self).__init__()
        self.ce_loss = nn.CrossEntropyLoss(ignore_index=ignore_index)
        self.transition_weight = transition_weight
        self.ignore_index = ignore_index
        
        M = torch.zeros((num_classes, num_classes))
        for i in range(num_classes):
            for j in range(num_classes):
                dist = abs(i - j)
                if dist > 1:
                    M[i, j] = float(dist - 1)  
        self.register_buffer('penalty_matrix', M)

    def forward(self, logits, labels):
        ce = self.ce_loss(logits.view(-1, logits.size(-1)), labels.view(-1))
        probs = F.softmax(logits, dim=-1)
        p_t_minus_1 = probs[:, :-1, :]
        p_t = probs[:, 1:, :]
        intermediate = torch.matmul(p_t_minus_1, self.penalty_matrix)
        step_penalties = torch.sum(intermediate * p_t, dim=-1) 
        
        mask_t_minus_1 = (labels[:, :-1] != self.ignore_index)
        mask_t = (labels[:, 1:] != self.ignore_index)
        valid_transitions = mask_t_minus_1 & mask_t  
        
        if valid_transitions.sum() > 0:
            trans_loss = (step_penalties * valid_transitions).sum() / valid_transitions.sum()
        else:
            trans_loss = 0.0
            
        total_loss = ce
        return total_loss, ce, trans_loss

def collate_fn(batch):
    sequences, labels = zip(*batch)
    lengths = [len(seq) for seq in sequences]
    padded_sequences = pad_sequence(sequences, batch_first=True, padding_value=0)
    padded_labels = pad_sequence(labels, batch_first=True, padding_value=-1)
    return padded_sequences, padded_labels, lengths

train_dataset = KinematicDataset(num_samples=200, is_train=True)
train_loader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn, drop_last=False)

# ==============================================================================
# --- 3. 定义 TCN 模型 (核心替换部分) ---
# ==============================================================================

class Chomp1d(nn.Module):
    """
    为了保证因果性 (Causality)，1D卷积在右侧多出来的 padding 必须被裁剪掉，
    这样 t 时刻的输出才绝对不会看到 t+1 时刻及以后的输入。
    """
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous()

class TemporalBlock(nn.Module):
    """ TCN 的基本残差块 (包含两个带有空洞的因果卷积层) """
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.2):
        super(TemporalBlock, self).__init__()
        
        self.conv1 = nn.Conv1d(n_inputs, n_outputs, kernel_size,
                               stride=stride, padding=padding, dilation=dilation)
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = nn.Conv1d(n_outputs, n_outputs, kernel_size,
                               stride=stride, padding=padding, dilation=dilation)
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        self.net = nn.Sequential(self.conv1, self.chomp1, self.relu1, self.dropout1,
                                 self.conv2, self.chomp2, self.relu2, self.dropout2)
        
        # 1x1 卷积用于通道数匹配（如果输入和输出通道不一致，残差连接前需要过一下 1x1 卷积）
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)

class TemporalConvNet(nn.Module):
    def __init__(self, num_inputs, num_channels, kernel_size=2, dropout=0.2):
        super(TemporalConvNet, self).__init__()
        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_size = 2 ** i   # 膨胀系数按 2^i 指数级增长
            in_channels = num_inputs if i == 0 else num_channels[i-1]
            out_channels = num_channels[i]
            
            # 为了保证因果性，单侧 padding 必须等于 (kernel_size - 1) * dilation_size
            padding = (kernel_size - 1) * dilation_size
            layers.append(TemporalBlock(in_channels, out_channels, kernel_size, stride=1, 
                                        dilation=dilation_size, padding=padding, dropout=dropout))

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)

class SequenceLabelingTCN(nn.Module):
    """
    包装成与 LSTM 完全一致的输入输出接口：
    输入: (Batch, Seq_Len, Features)
    输出: (Batch, Seq_Len, Num_Classes)
    """
    def __init__(self, input_size, num_channels, num_classes, kernel_size=3, dropout=0.2):
        super(SequenceLabelingTCN, self).__init__()
        
        # 核心 TCN 编码器
        self.tcn = TemporalConvNet(input_size, num_channels, kernel_size=kernel_size, dropout=dropout)
        
        # 最后的线性分类层
        self.fc = nn.Linear(num_channels[-1], num_classes)

    def forward(self, x, lengths=None):
        """
        Lengths 参数在这里保留以保持与训练循环接口兼容，TCN原生处理 Padded Sequence。
        """
        # PyTorch 的 Conv1d 要求输入格式为 (Batch, Channels, Length)
        x = x.transpose(1, 2)
        
        # 过 TCN 网络
        y = self.tcn(x)
        
        # 转换回 (Batch, Length, Channels) 准备给 Linear 层
        y = y.transpose(1, 2)
        
        # 生成 logits
        logits = self.fc(y)
        return logits

# ==============================================================================

# --- 保存训练好的模型 ---
def _build_train_scalers(demo_id_list):
    excluded = [80, 81, 92, 109, 112, 117, 122, 144, 145]
    if demo_id_list is None:
        import numpy as _np
        demo_id_list = _np.arange(148)
        demo_id_list = _np.delete(demo_id_list, excluded)

    from load_data import ratio
    raw_all = load_demonstrations_state(
        shuffle=True, without_quat=without_quat,
        resample=resample, demo_id_list=demo_id_list,
    )
    bound = round(ratio * len(raw_all))
    _, scalers = _scale_demos(raw_all[:bound])
    return scalers

def save_model(model, filepath, additional_info=None, demo_id_list=None):
    print("正在拟合并保存 scalers …")
    scalers = _build_train_scalers(demo_id_list)

    save_dict = {
        'model_state_dict': model.state_dict(),
        'model_config': {
            'input_size': INPUT_SIZE,
            'num_channels': TCN_CHANNELS,
            'kernel_size': KERNEL_SIZE,
            'num_classes': NUM_CLASSES,
        },
        'training_info': {
            'epochs': NUM_EPOCHS,
            'batch_size': BATCH_SIZE,
            'learning_rate': LEARNING_RATE,
        },
        'scalers': scalers,
        'without_quat': without_quat,
    }

    if additional_info:
        save_dict['additional_info'] = additional_info

    torch.save(save_dict, filepath)
    print(f"TCN 模型已保存到: {filepath}")

# --- 4. 训练循环 ---
def train_TCN():
    # 初始化 TCN 模型
    model = SequenceLabelingTCN(INPUT_SIZE, TCN_CHANNELS, NUM_CLASSES, KERNEL_SIZE, DROPOUT).to(device)
    print(model)

    dir_path = os.path.dirname(__file__)
    log_total_dir = os.path.join(dir_path, "logs", "TCN")
    exp_num = len(os.listdir(log_total_dir)) if os.path.exists(log_total_dir) else 0
    datetime_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = os.path.join(log_total_dir, f"exp{exp_num}_{datetime_str}")
    os.makedirs(log_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=log_dir)

    criterion = SmoothPhaseLoss(num_classes=NUM_CLASSES, transition_weight=1.0, ignore_index=-1).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    scheduler = StepLR(optimizer, step_size=200, gamma=0.5)

    train_start = time.time()
    for epoch in range(NUM_EPOCHS):
        epoch_start = time.time()
        model.train()
        total_loss = 0
        for i, (sequences, labels, lengths) in enumerate(train_loader):
            sequences = sequences.to(device)
            labels = labels.to(device)

            outputs = model(sequences, lengths)
            loss, ce_loss, trans_loss = criterion(outputs, labels)
            
            optimizer.zero_grad()
            loss.backward()
            
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            
            optimizer.step()
            total_loss += loss.item()

        scheduler.step()

        avg_loss = total_loss / len(train_loader)
        writer.add_scalar('loss', avg_loss, epoch)

        elapsed = time.time() - train_start
        avg_per_epoch = elapsed / (epoch + 1)
        eta = avg_per_epoch * (NUM_EPOCHS - epoch - 1)
        eta_m, eta_s = divmod(int(eta), 60)
        eta_h, eta_m = divmod(eta_m, 60)
        
        current_lr = optimizer.param_groups[0]['lr']
        print(f'Epoch [{epoch+1}/{NUM_EPOCHS}], Loss: {avg_loss:.4f}, CE Loss: {ce_loss.item():.4f}, Jump Penalty: {trans_loss.item():.4f}, LR: {current_lr:.6f}, '
              f'Epoch Time: {time.time()-epoch_start:.1f}s, '
              f'ETA: {eta_h:02d}:{eta_m:02d}:{eta_s:02d}')

    total_time = time.time() - train_start
    print(f"训练完成! 总耗时: {total_time/60:.1f} min")
    writer.close()
    return model

if __name__ == "__main__":
    # 训练 TCN
    model = train_TCN()

    # 保存模型
    dir_path = os.path.dirname(__file__)
    os.makedirs(os.path.join(dir_path, "TCN_model"), exist_ok=True)
    model_save_path = os.path.join(dir_path, "TCN_model", "tcn_sequence_model.pth")
    save_model(model, model_save_path)