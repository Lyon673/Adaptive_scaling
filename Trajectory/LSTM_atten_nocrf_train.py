"""
LSTM_atten_nocrf_seg_train.py
=============================
纯双臂空间交叉注意力 + LSTM 架构 (移除 CRF 层)
用于手术阶段识别的消融实验 Baseline
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence
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

# --- 1. 定义超参数 (Hyperparameters) ---
INPUT_SIZE = 16       # 原始特征数 (14 kinematic variables + time)
PROJ_DIM = 16         # 潜空间映射维度 (W_p 矩阵的输出维度)
HIDDEN_SIZE = 256     # LSTM 隐藏层的大小
NUM_LAYERS = 3        # LSTM 层数
NUM_CLASSES = 7       # 标签类别数量
BATCH_SIZE = 16
NUM_EPOCHS = 800
LEARNING_RATE = 1e-3

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# =========================================================================
# --- 1. 双臂空间交叉注意力 (Bimanual Spatial Cross-Attention) ---
# =========================================================================

class BimanualSpatialAttention(nn.Module):
    def __init__(self, input_dim=8, proj_dim=16):
        super(BimanualSpatialAttention, self).__init__()
        self.shared_proj = nn.Linear(input_dim, proj_dim)
        self.dominance_mlp = nn.Sequential(
            nn.Linear(proj_dim * 2, proj_dim),
            nn.ReLU(),
            nn.Linear(proj_dim, 2)
        )

    def forward(self, x):
        X_L = x[:, :, :8]
        X_R = x[:, :, 8:]

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
        
        return tilde_X, alphas

# =========================================================================
# --- 2. 准备数据 (Data Preparation) ---
# =========================================================================

class KinematicDataset(Dataset):
    def __init__(self, num_samples=100, is_train=True):
        self.samples = []
        demo_id_list = np.arange(148)
        demo_id_list = np.delete(demo_id_list, [80, 81, 92, 109, 112, 117, 122, 144, 145])

        demonstrations_state = load_train_state(without_quat=without_quat, resample=resample, demo_id_list=demo_id_list)
        demonstrations_label = load_train_label(resample=resample, demo_id_list=demo_id_list)
        
        for state_seq, label_seq in zip(demonstrations_state, demonstrations_label):
            if len(state_seq) == 0 or len(label_seq) == 0: continue
            state_tensor = torch.tensor(state_seq, dtype=torch.float32)
            label_tensor = torch.tensor(label_seq, dtype=torch.long)
            if state_tensor.dim() == 1: state_tensor = state_tensor.unsqueeze(1)  
            self.samples.append((state_tensor, label_tensor))

    def __len__(self): return len(self.samples)
    def __getitem__(self, idx): return self.samples[idx]

def collate_fn(batch):
    sequences, labels = zip(*batch)
    lengths = [len(seq) for seq in sequences]
    padded_sequences = pad_sequence(sequences, batch_first=True, padding_value=0)
    padded_labels = pad_sequence(labels, batch_first=True, padding_value=-1) 
    return padded_sequences, padded_labels, lengths

train_dataset = KinematicDataset(num_samples=200, is_train=True)
train_loader = DataLoader(
    dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True,
    collate_fn=collate_fn, drop_last=False
)

# =========================================================================
# --- 3. 模型定义 (Spatial Attn + LSTM - 无 CRF 版本) ---
# =========================================================================

class SequenceLabelingLSTM_Attn(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes, proj_dim=PROJ_DIM):
        super(SequenceLabelingLSTM_Attn, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.proj_dim = proj_dim
        
        self.spatial_attention = BimanualSpatialAttention(input_dim=8, proj_dim=proj_dim)
        
        self.lstm = nn.LSTM(
            input_size=16, 
            hidden_size=hidden_size, 
            num_layers=num_layers, 
            batch_first=True, 
            bidirectional=False,
            dropout=0.2 if num_layers > 1 else 0
        )
        
        self.fc = nn.Linear(hidden_size, num_classes)
        
        # 定义忽略标签 -1 的交叉熵损失函数
        self.criterion = nn.CrossEntropyLoss(ignore_index=-1)

    def forward(self, x, lengths, return_alphas=False):
        tilde_X, alphas = self.spatial_attention(x)
        
        packed_input = pack_padded_sequence(tilde_X, lengths, batch_first=True, enforce_sorted=False)
        packed_output, (h_n, c_n) = self.lstm(packed_input)
        lstm_out, _ = pad_packed_sequence(packed_output, batch_first=True)
        
        logits = self.fc(lstm_out)
        
        if return_alphas:
            return logits, alphas
        return logits

    def compute_loss(self, x, labels, lengths):
        # x: (B, T, 16), labels: (B, T)
        logits = self.forward(x, lengths)  # (B, T, C)
        
        # 展平以便通过 CrossEntropyLoss
        logits_flat = logits.reshape(-1, logits.size(-1)) # (B * T, C)
        labels_flat = labels.reshape(-1)                  # (B * T,)
        
        loss = self.criterion(logits_flat, labels_flat)
        return loss

    # 保留 decode 接口以兼容部分评估脚本，直接取 argmax
    def decode(self, x, lengths, return_alphas=False):
        if return_alphas:
            logits, alphas = self.forward(x, lengths, return_alphas=True)
            preds = torch.argmax(logits, dim=-1)
            return preds, alphas
        else:
            logits = self.forward(x, lengths)
            preds = torch.argmax(logits, dim=-1)
            return preds

# =========================================================================
# --- 4. 训练与保存 (Training & Saving) ---
# =========================================================================

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
            'hidden_size': HIDDEN_SIZE,
            'num_layers': NUM_LAYERS,
            'num_classes': NUM_CLASSES,
            'proj_dim': PROJ_DIM,
            'bidirectional': False,
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
    print(f"模型已保存到: {filepath}")

def train_LSTM_Attn():
    model = SequenceLabelingLSTM_Attn(INPUT_SIZE, HIDDEN_SIZE, NUM_LAYERS, NUM_CLASSES).to(device)
    print(model)

    dir_path = os.path.dirname(__file__)
    log_total_dir = os.path.join(dir_path, "logs", "SpatialAttn_LSTM")
    exp_num = len(os.listdir(log_total_dir)) if os.path.exists(log_total_dir) else 0
    datetime_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = os.path.join(log_total_dir, f"exp{exp_num}_{datetime_str}")
    os.makedirs(log_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=log_dir)

    # 由于没有 CRF 层，全部使用统一的初始学习率
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
            
            optimizer.zero_grad()
            loss = model.compute_loss(sequences, labels, lengths)
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
        print(f'Epoch [{epoch+1}/{NUM_EPOCHS}], Loss: {avg_loss:.4f}, LR: {current_lr:.6f}, '
              f'Epoch Time: {time.time()-epoch_start:.1f}s, '
              f'ETA: {eta_h:02d}:{eta_m:02d}:{eta_s:02d}')

    total_time = time.time() - train_start
    print(f"训练完成! 总耗时: {total_time/60:.1f} min")
    writer.close()
    return model

if __name__ == "__main__":
    model = train_LSTM_Attn()

    dir_path = os.path.dirname(__file__)
    os.makedirs(os.path.join(dir_path, "LSTM_model"), exist_ok=True)
    
    # 更改了输出路径，以便将其与包含 CRF 的模型区分开来
    model_save_path = os.path.join(dir_path, "LSTM_model", "spatial_attn_lstmnocrf_sequence_model.pth")
    save_model(model, model_save_path)