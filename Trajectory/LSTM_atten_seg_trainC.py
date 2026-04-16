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
from torchcrf import CRF
from torch.utils.tensorboard import SummaryWriter
import datetime
import time
from config import resample, without_quat
# 引入余弦退火学习率
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts

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
# --- 组件升级: 双臂空间交叉注意力 (含因果卷积与残差归一化) ---
# =========================================================================

class BimanualSpatialAttention(nn.Module):
    def __init__(self, input_dim=8, proj_dim=16):
        super(BimanualSpatialAttention, self).__init__()
        
        # 改进 1: 使用因果卷积 (Causal Conv1d) 替代 Linear
        # 允许 t 时刻的特征融合 t, t-1, t-2 的运动学趋势，但不看到 t+1，严格保证实时推演合法性
        self.shared_causal_conv = nn.Conv1d(in_channels=input_dim, out_channels=proj_dim, kernel_size=3)
        
        # 评估主导态势的 MLP
        self.dominance_mlp = nn.Sequential(
            nn.Linear(proj_dim * 2, proj_dim),
            nn.ReLU(),
            nn.Linear(proj_dim, 2)
        )
        
        # 改进 2: 层归一化 (Layer Normalization) 稳定残差流的梯度
        self.layer_norm = nn.LayerNorm(16)

    def forward(self, x):
        """
        x: shape (B, T, 16)
        """
        # 物理拆解特征
        X_L = x[:, :, :8]
        X_R = x[:, :, 8:]

        # --- 准备因果卷积 (要求输入 shape 为 B, C, T) ---
        X_L_t = X_L.transpose(1, 2)
        X_R_t = X_R.transpose(1, 2)

        # 核心：只在左侧（过去的时间）进行 padding 2 帧，保证 kernel_size=3 时不泄露未来信息
        X_L_pad = F.pad(X_L_t, (2, 0))
        X_R_pad = F.pad(X_R_t, (2, 0))

        # 卷积提取局部时空高阶特征，转回 (B, T, proj_dim)
        H_L = F.relu(self.shared_causal_conv(X_L_pad)).transpose(1, 2)
        H_R = F.relu(self.shared_causal_conv(X_R_pad)).transpose(1, 2)

        # --- 主导权注意力计算 (Dominance Attention) ---
        H_cat = torch.cat([H_L, H_R], dim=-1)           # (B, T, proj_dim*2)
        alpha_logits = self.dominance_mlp(H_cat)        # (B, T, 2)
        alphas = F.softmax(alpha_logits, dim=-1)        # (B, T, 2) 归一化

        # 提取标量权重并扩展维度以匹配特征
        alpha_L = alphas[:, :, 0].unsqueeze(-1)         # (B, T, 1)
        alpha_R = alphas[:, :, 1].unsqueeze(-1)         # (B, T, 1)

        # --- 空间重加权 (Spatial Reweighting) ---
        tilde_X_L = alpha_L * X_L
        tilde_X_R = alpha_R * X_R
        tilde_X = torch.cat([tilde_X_L, tilde_X_R], dim=-1) # (B, T, 16)
        
        # 改进 3: 引入残差连接 (Residual) + LayerNorm，避免连乘导致深层梯度消失
        out_X = self.layer_norm(x + tilde_X)
        
        return out_X, alphas

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
# --- 3. 模型定义 (Causal Spatial Attn + LSTM + CRF) ---
# =========================================================================

class SequenceLabelingLSTM_CRF(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes, proj_dim=PROJ_DIM):
        super(SequenceLabelingLSTM_CRF, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.proj_dim = proj_dim
        
        # 1. 带有因果属性的双臂空间交叉注意力
        self.spatial_attention = BimanualSpatialAttention(input_dim=8, proj_dim=proj_dim)
        
        # 2. LSTM 层 (严格保持 bidirectional=False 以满足实时在线推演需求)
        self.lstm = nn.LSTM(
            input_size=16, 
            hidden_size=hidden_size, 
            num_layers=num_layers, 
            batch_first=True, 
            bidirectional=False, 
            dropout=0.2 if num_layers > 1 else 0
        )
        
        # 3. CRF 分类层
        self.fc = nn.Linear(hidden_size, num_classes)
        self.crf = CRF(num_classes, batch_first=True)

        with torch.no_grad():
            self.crf.transitions.fill_(0) 
            for i in range(num_classes):
                self.crf.transitions[i, i] = 2.0
                if i < num_classes - 1:
                    self.crf.transitions[i, i+1] = 2.0

    def forward(self, x, lengths, return_alphas=False):
        # # 改进 4: 运动学高斯噪声注入 (Kinematic Jittering) 
        # # 增加模型对测试集主从端设备微小标定漂移的鲁棒性
        # if self.training:
        #     noise = torch.randn_like(x) * 0.01
        #     x = x + noise
            
        # 1. 空间自适应重加权
        tilde_X, alphas = self.spatial_attention(x)
        
        # 2. 扔进 LSTM 处理全局因果时序
        packed_input = pack_padded_sequence(tilde_X, lengths, batch_first=True, enforce_sorted=False)
        packed_output, (h_n, c_n) = self.lstm(packed_input)
        lstm_out, _ = pad_packed_sequence(packed_output, batch_first=True)
        
        # 3. 映射为 CRF 的发射分数
        emissions = self.fc(lstm_out)
        
        if return_alphas:
            return emissions, alphas
        return emissions

    def compute_loss(self, x, labels, lengths):
        max_len = x.size(1)
        mask = torch.arange(max_len, device=x.device)[None, :] < torch.tensor(lengths, device=x.device)[:, None]

        safe_labels = labels.clone()
        valid_labels = (labels != -1)
        leading_mask = (valid_labels.cumsum(dim=1) == 0)
        
        safe_labels[leading_mask] = 0
        trailing_valid_mask = (safe_labels == -1) & mask
        safe_labels[trailing_valid_mask] = 6  
        safe_labels[~mask] = 0
        
        # 获取预测发射矩阵和注意力权重
        emissions, alphas = self.forward(x, lengths, return_alphas=True)
        
        # 1. CRF 主损失
        crf_loss = -self.crf(emissions, safe_labels, mask=mask, reduction='mean')
        
        # 改进 5: 注意力权重的时序平滑正则化 (Temporal Smoothness Penalty)
        # 惩罚主导权权重的相邻高频抖动，符合实际手术中主从臂物理切换的惯性连续性
        alpha_diff = alphas[:, 1:, :] - alphas[:, :-1, :]  # shape: (B, T-1, 2)
        mask_diff = mask[:, 1:]                            # 必须只计算有效帧的跳变
        
        if mask_diff.sum() > 0:
            smoothness_loss = torch.sum((alpha_diff ** 2)[mask_diff]) / mask_diff.sum()
        else:
            smoothness_loss = torch.tensor(0.0, device=x.device)
            
        # L2 平滑惩罚的权重设为 0.5，可抑制分割轨迹的碎片化现象 (Over-segmentation)
        total_loss = crf_loss 
        return total_loss

    def decode(self, x, lengths, return_alphas=False):
        max_len = x.size(1)
        mask = torch.arange(max_len, device=x.device)[None, :] < torch.tensor(lengths, device=x.device)[:, None]
        
        if return_alphas:
            emissions, alphas = self.forward(x, lengths, return_alphas=True)
            best_path = self.crf.decode(emissions, mask=mask)
            return best_path, alphas
        else:
            emissions = self.forward(x, lengths)
            best_path = self.crf.decode(emissions, mask=mask)
            return best_path


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
            'bidirectional': False,  # 确认强制在线因果
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

def train_LSTM_CRF():
    model = SequenceLabelingLSTM_CRF(INPUT_SIZE, HIDDEN_SIZE, NUM_LAYERS, NUM_CLASSES).to(device)
    print(model)

    dir_path = os.path.dirname(__file__)
    log_total_dir = os.path.join(dir_path, "logs", "SpatialAttn_LSTM_CRF")
    exp_num = len(os.listdir(log_total_dir)) if os.path.exists(log_total_dir) else 0
    datetime_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = os.path.join(log_total_dir, f"exp{exp_num}_{datetime_str}")
    os.makedirs(log_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=log_dir)

    crf_params = list(model.crf.parameters())
    base_params = (list(model.spatial_attention.parameters()) + 
                   list(model.lstm.parameters()) + 
                   list(model.fc.parameters()))
    
    optimizer = torch.optim.Adam([
        {'params': base_params, 'lr': LEARNING_RATE}, 
        {'params': crf_params, 'lr': LEARNING_RATE * 5.0} 
    ])
    
    # 改进 6: 余弦退火与预热重启 (Cosine Annealing with Warm Restarts)
    # 彻底替换 StepLR，帮助模型在参数空间寻找更加平坦稳健的局部极小值
    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=200, T_mult=1, eta_min=1e-5)

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
    model = train_LSTM_CRF()

    dir_path = os.path.dirname(__file__)
    os.makedirs(os.path.join(dir_path, "LSTM_model"), exist_ok=True)
    model_save_path = os.path.join(dir_path, "LSTM_model", "spatial_attn_lstmcrf_sequence_model.pth")
    save_model(model, model_save_path)