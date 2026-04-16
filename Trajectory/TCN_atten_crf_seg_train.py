"""
Attn_TeCNO_CRF_train.py
=======================
融合架构：Bimanual Spatial Cross-Attention + Multi-Stage TCN (TeCNO) + CRF
用于机器人辅助手术的实时高精度阶段识别 (Phase Recognition)。

创新点：
1. Spatial Attention: 动态分配左右臂的特征主导权。
2. MS-TCN: 利用因果膨胀卷积提供超大感受野，解决长程时序依赖与中段概率塌陷。
3. Multi-Stage CRF Loss: 所有的 TCN Stage 输出均接受 CRF 的序列级监督与状态转移约束。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
import numpy as np
from load_data import (load_train_state, load_train_label,
                       load_demonstrations_state, _scale_demos)
import os
from torchcrf import CRF
from torch.utils.tensorboard import SummaryWriter
import datetime
import time
from config import resample, without_quat
from torch.optim.lr_scheduler import CosineAnnealingLR

# =========================================================================
# --- 1. 超参数配置 (Hyperparameters) ---
# =========================================================================
INPUT_SIZE = 16       # 原始特征数 (14 kinematic variables + time)
PROJ_DIM = 16         # 注意力潜空间映射维度
HIDDEN_SIZE = 64      # TCN 通道宽度
NUM_LAYERS = 6        # 每个 Stage 的 TCN 层数 (感受野 RF = 2^(6+1)-1 = 127 帧)
NUM_STAGES = 2        # 多阶段精炼的级数
NUM_CLASSES = 7       # 阶段类别数
DROPOUT = 0.4         # TCN 内部 Dropout

BATCH_SIZE = 16
NUM_EPOCHS = 600
LEARNING_RATE = 1e-3  # 基础学习率 (CRF 层将自动使用 5 倍学习率)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")


# =========================================================================
# --- 2. 核心网络组件定义 ---
# =========================================================================

# ── A. 双臂空间交叉注意力层 ──
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

# ── B. TCN 基础因果膨胀卷积块 ──
class CausalDilatedConv1d(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size, dilation):
        super().__init__()
        self.pad_len = dilation * (kernel_size - 1)
        self.conv = nn.Conv1d(in_ch, out_ch, kernel_size, dilation=dilation)

    def forward(self, x):
        return self.conv(F.pad(x, (self.pad_len, 0)))

class DilatedResidualLayer(nn.Module):
    def __init__(self, hidden_dim, kernel_size, dilation, dropout):
        super().__init__()
        self.dconv = CausalDilatedConv1d(hidden_dim, hidden_dim, kernel_size, dilation)
        self.conv1x1 = nn.Conv1d(hidden_dim, hidden_dim, 1)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        z = self.drop(F.relu(self.dconv(x)))
        return x + self.conv1x1(z)

class TCNStage(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, num_layers, kernel_size, dropout):
        super().__init__()
        self.conv_in = nn.Conv1d(in_dim, hidden_dim, 1)
        self.layers = nn.ModuleList([
            DilatedResidualLayer(hidden_dim, kernel_size, dilation=2 ** i, dropout=dropout)
            for i in range(num_layers)
        ])
        self.conv_out = nn.Conv1d(hidden_dim, out_dim, 1)

    def forward(self, x):
        h = self.conv_in(x)
        for layer in self.layers:
            h = layer(h)
        return self.conv_out(h)

# ── C. 顶层融合架构 (Attn + TeCNO + CRF) ──
class SequenceLabelingAttnTeCNO_CRF(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes, num_stages, proj_dim, dropout):
        super(SequenceLabelingAttnTeCNO_CRF, self).__init__()
        self.num_stages = num_stages
        
        # 1. 空间注意力前端
        self.spatial_attention = BimanualSpatialAttention(input_dim=8, proj_dim=proj_dim)
        
        # 2. MS-TCN 多阶段序列编码器
        self.stages = nn.ModuleList()
        for s in range(num_stages):
            # Stage 1 接收物理特征(16维)，后续 Stage 接收上一层的软标签概率(7维)
            in_dim = input_size if s == 0 else num_classes
            self.stages.append(
                TCNStage(in_dim, hidden_size, num_classes, num_layers, kernel_size=3, dropout=dropout)
            )
            
        # 3. CRF 状态分类层
        self.crf = CRF(num_classes, batch_first=True)
        with torch.no_grad():
            self.crf.transitions.fill_(0) 
            for i in range(num_classes):
                self.crf.transitions[i, i] = 2.0
                if i < num_classes - 1:
                    self.crf.transitions[i, i+1] = 2.0

    def forward_all_stages(self, x):
        """返回所有 Stage 的未归一化 logits 用于计算联合损失"""
        # 1. 空间注意力加权 -> (B, T, 16)
        tilde_X, alphas = self.spatial_attention(x)
        
        # 2. TCN 需要 (B, C, T)
        h = tilde_X.permute(0, 2, 1)
        
        stage_outputs = []
        for i, stage in enumerate(self.stages):
            h = stage(h)
            # 转回 (B, T, C) 以匹配 CRF 的要求
            stage_outputs.append(h.permute(0, 2, 1))
            # 阶段间传递时转换为 softmax 概率特征
            if i < self.num_stages - 1:
                h = F.softmax(h, dim=1) 
                
        return stage_outputs, alphas

    def compute_loss(self, x, labels, lengths):
        """计算 Multi-Stage CRF Loss"""
        max_len = x.size(1)
        mask = torch.arange(max_len, device=x.device)[None, :] < torch.tensor(lengths, device=x.device)[:, None]

        # 保护边界标签以适应 CRF
        safe_labels = labels.clone()
        valid_labels = (labels != -1)
        leading_mask = (valid_labels.cumsum(dim=1) == 0)
        safe_labels[leading_mask] = 0
        trailing_valid_mask = (safe_labels == -1) & mask
        safe_labels[trailing_valid_mask] = 6  
        safe_labels[~mask] = 0
        
        # 获取所有 Stage 的输出
        stage_outputs, _ = self.forward_all_stages(x)
        
        # 累加每一个 Stage 的 CRF Loss
        total_loss = 0.0
        for emissions in stage_outputs:
            stage_loss = -self.crf(emissions, safe_labels, mask=mask, reduction='mean')
            total_loss += stage_loss
            
        # 平均损失
        return total_loss / self.num_stages

    def forward(self, x, lengths, return_alphas=False):
        """为了与某些纯算概率的接口兼容，默认返回最后一个 Stage 的 logits"""
        stage_outputs, alphas = self.forward_all_stages(x)
        final_emissions = stage_outputs[-1]
        if return_alphas:
            return final_emissions, alphas
        return final_emissions

    def decode(self, x, lengths, return_alphas=False):
        """推演时只使用最终 Stage 的特征过 Viterbi 解码"""
        max_len = x.size(1)
        mask = torch.arange(max_len, device=x.device)[None, :] < torch.tensor(lengths, device=x.device)[:, None]
        
        stage_outputs, alphas = self.forward_all_stages(x)
        final_emissions = stage_outputs[-1]
        
        best_path = self.crf.decode(final_emissions, mask=mask)
        
        if return_alphas:
            return best_path, alphas
        return best_path


# =========================================================================
# --- 3. 数据集处理 (Dataset Loading) ---
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

# =========================================================================
# --- 4. 训练引擎与日志 (Training Pipeline) ---
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

def save_model(model, filepath, demo_id_list=None):
    print("正在拟合并保存 scalers …")
    scalers = _build_train_scalers(demo_id_list)

    # 精心构造的 config，保证可视化脚本 models_seq_comp_vis.py 能智能嗅探
    save_dict = {
        'model_state_dict': model.state_dict(),
        'model_type': 'AttnTeCNO_CRF',
        'model_config': {
            'input_size': INPUT_SIZE,
            'hidden_size': HIDDEN_SIZE,
            'num_layers': NUM_LAYERS,
            'num_classes': NUM_CLASSES,
            'num_stages': NUM_STAGES,
            'proj_dim': PROJ_DIM,
            'dropout': DROPOUT,
            'kernel_size': 3,
        },
        'training_info': {
            'epochs': NUM_EPOCHS,
            'batch_size': BATCH_SIZE,
            'learning_rate': LEARNING_RATE,
        },
        'scalers': scalers,
        'without_quat': without_quat,
    }

    torch.save(save_dict, filepath)
    print(f"✅ 模型权重已成功保存到: {filepath}")

def train_fusion_model():
    # 1. 载入数据
    train_dataset = KinematicDataset(num_samples=200, is_train=True)
    train_loader = DataLoader(
        dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True,
        collate_fn=collate_fn, drop_last=False
    )
    print(f"Training sequences: {len(train_dataset)}")

    # 2. 实例化模型
    model = SequenceLabelingAttnTeCNO_CRF(
        input_size=INPUT_SIZE, hidden_size=HIDDEN_SIZE, 
        num_layers=NUM_LAYERS, num_classes=NUM_CLASSES, 
        num_stages=NUM_STAGES, proj_dim=PROJ_DIM, dropout=DROPOUT
    ).to(device)
    
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    rf = 2 ** (NUM_LAYERS + 1) - 1
    print(f"Total Trainable Params: {n_params:,} | TCN Receptive Field: {rf} frames")

    # 3. 日志配置
    dir_path = os.path.dirname(__file__)
    log_total_dir = os.path.join(dir_path, "logs", "Attn_TeCNO_CRF")
    exp_num = len(os.listdir(log_total_dir)) if os.path.exists(log_total_dir) else 0
    datetime_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = os.path.join(log_total_dir, f"exp{exp_num}_{datetime_str}")
    os.makedirs(log_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=log_dir)

    # 4. 优化器隔离 (让 CRF 的学习率保持 5 倍于主网络)
    crf_params = list(model.crf.parameters())
    base_params = (list(model.spatial_attention.parameters()) + 
                   list(model.stages.parameters()))
    
    optimizer = torch.optim.Adam([
        {'params': base_params, 'lr': LEARNING_RATE}, 
        {'params': crf_params, 'lr': LEARNING_RATE * 5.0} 
    ])
    
    scheduler = CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS)

    # 5. 训练主循环
    train_start = time.time()
    for epoch in range(NUM_EPOCHS):
        epoch_start = time.time()
        model.train()
        total_loss = 0
        
        for i, (sequences, labels, lengths) in enumerate(train_loader):
            sequences = sequences.to(device)
            labels = labels.to(device)
            
            optimizer.zero_grad()
            # 内部已实现 Multi-Stage Loss 求平均
            loss = model.compute_loss(sequences, labels, lengths)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
            
            total_loss += loss.item()

        scheduler.step()

        avg_loss = total_loss / len(train_loader)
        writer.add_scalar('Loss/MultiStage_CRF', avg_loss, epoch)
        writer.add_scalar('LR/Base', optimizer.param_groups[0]['lr'], epoch)
        
        elapsed = time.time() - train_start
        avg_per_epoch = elapsed / (epoch + 1)
        eta = avg_per_epoch * (NUM_EPOCHS - epoch - 1)
        eta_m, eta_s = divmod(int(eta), 60)
        eta_h, eta_m = divmod(eta_m, 60)
        
        current_lr = optimizer.param_groups[0]['lr']
        print(f'Epoch [{epoch+1}/{NUM_EPOCHS}] | Multi-Stage CRF Loss: {avg_loss:.4f} | '
              f'LR: {current_lr:.6f} | Time: {time.time()-epoch_start:.1f}s | '
              f'ETA: {eta_h:02d}:{eta_m:02d}:{eta_s:02d}')

    total_time = time.time() - train_start
    print(f"\n🎉 训练大功告成! 总耗时: {total_time/60:.1f} 分钟")
    writer.close()
    return model

if __name__ == "__main__":
    # 执行训练
    model = train_fusion_model()

    # 保存权重
    dir_path = os.path.dirname(__file__)
    save_dir = os.path.join(dir_path, "TeCNO_model")  
    os.makedirs(save_dir, exist_ok=True)
    
    # 命名以体现架构
    model_save_path = os.path.join(save_dir, "attn_tecno_crf_sequence_model.pth")
    save_model(model, model_save_path)