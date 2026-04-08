import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
import numpy as np
from load_data import (load_train_state, load_train_label,
                       load_demonstrations_state, _scale_demos,
                       get_shuffled_demo_ids)
from sklearn.metrics import classification_report, accuracy_score
import os
from torch.utils.tensorboard import SummaryWriter
import datetime
import time
from config import resample, without_quat
from torch.optim.lr_scheduler import StepLR

# 导入 Mamba 核心模块
try:
    from mamba_ssm import Mamba
except ImportError:
    raise ImportError("请先安装 mamba-ssm: pip install mamba-ssm causal-conv1d")

# --- 1. 定义超参数 (Hyperparameters) ---
INPUT_SIZE = 16       # 每个时间点的特征数 (14 kinematic variables from both arms + time)
D_MODEL = 256         # Mamba 内部的特征维度 (等同于 LSTM 的 hidden_size)
NUM_LAYERS = 3        # Mamba 层数
NUM_CLASSES = 7       # 标签的类别数量 (0-6)
BATCH_SIZE = 16
NUM_EPOCHS = 800
LEARNING_RATE = 0.001

# 检查是否有可用的GPU (Mamba 强依赖 CUDA)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")


# --- 2. 准备数据 (与原版保持完全一致) ---
class KinematicDataset(Dataset):
    def __init__(self, num_samples=100, is_train=True):
        self.samples = []
        demo_id_list = np.arange(148)
        demo_id_list = np.delete(demo_id_list, [80, 81, 92, 109, 112, 117, 122, 144, 145])

        demonstrations_state = load_train_state(without_quat=without_quat, resample=resample, demo_id_list=demo_id_list)
        demonstrations_label = load_train_label(resample=resample, demo_id_list=demo_id_list)
        
        for state_seq, label_seq in zip(demonstrations_state, demonstrations_label):
            if len(state_seq) == 0 or len(label_seq) == 0:
                print("⚠️ 警告: 发现长度为 0 的空序列，已自动跳过！")
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
            
        total_loss = ce + self.transition_weight * trans_loss
        return total_loss, ce, trans_loss

def collate_fn(batch):
    sequences, labels = zip(*batch)
    lengths = [len(seq) for seq in sequences]
    padded_sequences = pad_sequence(sequences, batch_first=True, padding_value=0)
    padded_labels = pad_sequence(labels, batch_first=True, padding_value=-1)
    return padded_sequences, padded_labels, lengths

train_dataset = KinematicDataset(num_samples=200, is_train=True)
train_loader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn, drop_last=False)


# --- 3. 定义 Mamba 模型 (核心替换部分) ---
class SequenceLabelingMamba(nn.Module):
    def __init__(self, input_size, d_model, num_layers, num_classes):
        super(SequenceLabelingMamba, self).__init__()
        self.d_model = d_model
        
        # 1. 输入投影层: 将 16 维的特征投影到 Mamba 需要的高维特征空间 (d_model)
        self.embedding = nn.Linear(input_size, d_model)
        
        # 2. 堆叠多个 Mamba Block (带有 LayerNorm 和残差连接)
        self.layers = nn.ModuleList([
            nn.ModuleDict({
                'mamba': Mamba(
                    d_model=d_model, # Model dimension d_model
                    d_state=16,      # SSM state expansion factor (默认16)
                    d_conv=4,        # Local convolution width
                    expand=2,        # Block expansion factor
                ),
                'norm': nn.LayerNorm(d_model)
            }) for _ in range(num_layers)
        ])
        
        # 3. 输出分类层
        self.fc = nn.Linear(d_model, num_classes)

    def forward(self, x, lengths=None):
        """
        :param x: shape (batch_size, seq_len, input_size)
        :param lengths: 传入以保持与 LSTM 接口一致，但 Mamba 天然支持 Casual 处理 padded 数据
        """
        # (Batch, Seq_len, Input_Size) -> (Batch, Seq_len, D_Model)
        hidden = self.embedding(x)
        
        # 通过 Mamba 层
        for layer in self.layers:
            # 采用 Pre-Norm 结构 + 残差连接
            normed_hidden = layer['norm'](hidden)
            hidden = hidden + layer['mamba'](normed_hidden)
            
        # (Batch, Seq_len, D_Model) -> (Batch, Seq_len, Num_Classes)
        logits = self.fc(hidden)
        return logits


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
            'd_model': D_MODEL,
            'num_layers': NUM_LAYERS,
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
    print(f"Mamba 模型已保存到: {filepath}")


# --- 4. 训练循环 ---
def train_Mamba():
    # 初始化 Mamba 模型
    model = SequenceLabelingMamba(INPUT_SIZE, D_MODEL, NUM_LAYERS, NUM_CLASSES).to(device)
    print(model)

    dir_path = os.path.dirname(__file__)
    log_total_dir = os.path.join(dir_path, "logs", "Mamba")
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

            # Mamba 的前向传播
            outputs = model(sequences, lengths)
            loss, ce_loss, trans_loss = criterion(outputs, labels)
            
            optimizer.zero_grad()
            loss.backward()
            
            # Mamba 容易出现梯度爆炸，截断梯度非常重要
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
    # 训练模型
    model = train_Mamba()

    # 保存模型
    dir_path = os.path.dirname(__file__)
    os.makedirs(os.path.join(dir_path, "Mamba_model"), exist_ok=True)
    model_save_path = os.path.join(dir_path, "Mamba_model", "mamba_sequence_model.pth")
    save_model(model, model_save_path)