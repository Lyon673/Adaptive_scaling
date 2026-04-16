"""
Attn_TeCNO_seg_train.py
=======================
融合架构：Bimanual Spatial Cross-Attention (BDG) + Multi-Stage TCN (TeCNO)
用于机器人辅助手术的实时高精度阶段识别。

架构说明:
  前端 (Front-end): 双臂空间交叉注意力，动态评估并重加权左右臂特征的主导权。
  主干 (Backbone):  多阶段时序卷积网络 (MS-TCN)，利用因果膨胀卷积提取长程时序依赖。
  损失 (Loss):      基于 TeCNO 的多阶段加权交叉熵 + 状态转移惩罚 (Transition Penalty)。

输入特征: 16-dim (左臂8维 + 右臂8维)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
import numpy as np
from load_data import (
    load_train_state, load_train_label,
    load_test_state, load_test_label,
    load_demonstrations_state, _scale_demos,
)
from sklearn.metrics import classification_report, accuracy_score
import os
from torch.utils.tensorboard import SummaryWriter
import datetime
import time
from config import resample, without_quat
from torch.optim.lr_scheduler import CosineAnnealingLR

# ── 1. 超参数配置 (Hyperparameters) ──────────────────────────────────────
INPUT_SIZE = 16         # 原始物理特征总数 (16)
PROJ_DIM = 16           # 注意力前端的潜空间映射维度
HIDDEN_SIZE = 64        # TCN 通道宽度
NUM_LAYERS = 6          # 每个 Stage 的残差层数 (RF = 2^6-1 = 63 frames)
NUM_STAGES = 2          # 多阶段精炼的级数
NUM_CLASSES = 7         # 标签类别数
KERNEL_SIZE = 3
DROPOUT = 0.5           # TCN 内部 Dropout

BATCH_SIZE = 16
NUM_EPOCHS = 800
LEARNING_RATE = 1e-3

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# ── 2. 数据集与 DataLoader ───────────────────────────────────────────────

class KinematicDataset(Dataset):
    def __init__(self, states, labels):
        self.samples = []
        for s, l in zip(states, labels):
            if len(s) == 0 or len(l) == 0:
                continue
            st = torch.tensor(s, dtype=torch.float32)
            lt = torch.tensor(l, dtype=torch.long)
            if st.dim() == 1:
                st = st.unsqueeze(1)
            self.samples.append((st, lt))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]

def collate_fn(batch):
    sequences, labels = zip(*batch)
    lengths = [len(seq) for seq in sequences]
    padded_sequences = pad_sequence(sequences, batch_first=True, padding_value=0)
    padded_labels = pad_sequence(labels, batch_first=True, padding_value=-1)
    return padded_sequences, padded_labels, lengths

# ── 3. 模型核心组件 (Architecture) ───────────────────────────────────────

# 【新增】双臂空间交叉注意力前端
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
        """ x: shape (B, T, 16) """
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

# TCN 因果膨胀卷积基础块
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

# 融合主网络 (Attn + TeCNO)
class SequenceLabelingAttnTeCNO(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes,
                 num_stages=2, kernel_size=3, proj_dim=16, dropout=0.5):
        super().__init__()
        self.num_stages = num_stages
        self.num_classes = num_classes
        
        # 1. 空间注意力前端
        self.spatial_attention = BimanualSpatialAttention(input_dim=8, proj_dim=proj_dim)
        
        # 2. 多阶段 TCN 主干
        self.stages = nn.ModuleList()
        for s in range(num_stages):
            in_dim = input_size if s == 0 else num_classes
            self.stages.append(
                TCNStage(in_dim, hidden_size, num_classes, num_layers, kernel_size, dropout)
            )

    def forward_all_stages(self, x, lengths=None):
        """返回所有 Stage 的输出用于计算联合损失，同时返回注意力权重"""
        # 前端重加权
        tilde_X, alphas = self.spatial_attention(x)
        
        # 转入 TCN 需要的 (B, C, T) 格式
        h = tilde_X.permute(0, 2, 1)
        
        outputs = []
        for i, stage in enumerate(self.stages):
            h = stage(h)
            outputs.append(h.permute(0, 2, 1))  # 换回 (B, T, C)
            if i < self.num_stages - 1:
                h = F.softmax(h, dim=1)         # 阶段间传递概率
                
        return outputs, alphas

    def forward(self, x, lengths=None, return_alphas=False):
        """推理时调用：只返回最后一层的预测 (及其注意力权重)"""
        stage_outputs, alphas = self.forward_all_stages(x, lengths)
        final_logits = stage_outputs[-1]
        
        if return_alphas:
            return final_logits, alphas
        return final_logits

# ── 4. 多阶段损失函数 (Multi-Stage Loss) ─────────────────────────────────

class MultiStageLoss(nn.Module):
    """
    计算所有 Stage 的平均交叉熵，并附带状态转移平滑惩罚 (Transition Penalty)。
    """
    def __init__(self, num_classes=7, transition_weight=0.5, ignore_index=-1):
        super().__init__()
        self.ce = nn.CrossEntropyLoss(ignore_index=ignore_index)
        self.transition_weight = transition_weight
        self.ignore_index = ignore_index

        M = torch.zeros(num_classes, num_classes)
        for i in range(num_classes):
            for j in range(num_classes):
                d = abs(i - j)
                if d > 1:
                    M[i, j] = float(d - 1)
        self.register_buffer('penalty_matrix', M)

    def _transition_penalty(self, logits, labels):
        probs = F.softmax(logits, dim=-1)
        intermediate = torch.matmul(probs[:, :-1, :], self.penalty_matrix)
        step_pen = torch.sum(intermediate * probs[:, 1:, :], dim=-1)
        m1 = (labels[:, :-1] != self.ignore_index)
        m2 = (labels[:, 1:] != self.ignore_index)
        valid = m1 & m2
        if valid.sum() > 0:
            return (step_pen * valid).sum() / valid.sum()
        return torch.tensor(0.0, device=logits.device)

    def forward(self, stage_outputs, labels):
        total_ce = 0.0
        total_trans = 0.0
        M = len(stage_outputs)
        
        for logits in stage_outputs:
            total_ce += self.ce(logits.reshape(-1, logits.size(-1)), labels.reshape(-1))
            total_trans += self._transition_penalty(logits, labels)
            
        ce_avg = total_ce / M
        trans_avg = total_trans / M
        total = ce_avg + self.transition_weight * trans_avg
        return total, ce_avg, trans_avg

# ── 5. 工具函数 (Utils) ──────────────────────────────────────────────────

def _build_train_scalers(demo_id_list=None):
    excluded = [80, 81, 92, 109, 112, 117, 122, 144, 145]
    if demo_id_list is None:
        demo_id_list = np.delete(np.arange(148), excluded)
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
        'model_type': 'AttnTeCNO',
        'model_config': {
            'input_size': INPUT_SIZE,
            'hidden_size': HIDDEN_SIZE,
            'num_layers': NUM_LAYERS,
            'num_classes': NUM_CLASSES,
            'num_stages': NUM_STAGES,
            'proj_dim': PROJ_DIM,
            'kernel_size': KERNEL_SIZE,
            'dropout': DROPOUT,
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

# ── 6. 训练主引擎 (Training Engine) ──────────────────────────────────────

def train_Attn_TeCNO():
    demo_id_list = np.delete(np.arange(148), [80, 81, 92, 109, 112, 117, 122, 144, 145])

    train_states = load_train_state(without_quat=without_quat, resample=resample, demo_id_list=demo_id_list)
    train_labels = load_train_label(resample=resample, demo_id_list=demo_id_list)
    train_dataset = KinematicDataset(train_states, train_labels)
    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True,
        collate_fn=collate_fn, drop_last=False,
    )
    print(f"Training demos: {len(train_dataset)}")

    model = SequenceLabelingAttnTeCNO(
        INPUT_SIZE, HIDDEN_SIZE, NUM_LAYERS, NUM_CLASSES,
        num_stages=NUM_STAGES, kernel_size=KERNEL_SIZE, proj_dim=PROJ_DIM, dropout=DROPOUT,
    ).to(device)
    
    print(model)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    rf = 2 ** (NUM_LAYERS + 1) - 1
    print(f"Trainable params: {n_params:,}   Receptive field per stage: {rf} frames")

    criterion = MultiStageLoss(num_classes=NUM_CLASSES, transition_weight=0.5, ignore_index=-1).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS)

    # TensorBoard 日志设置
    dir_path = os.path.dirname(__file__)
    log_total_dir = os.path.join(dir_path, "logs", "Attn_TeCNO")
    exp_num = len(os.listdir(log_total_dir)) if os.path.exists(log_total_dir) else 0
    datetime_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = os.path.join(log_total_dir, f"exp{exp_num}_{datetime_str}")
    os.makedirs(log_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=log_dir)

    train_start = time.time()
    for epoch in range(NUM_EPOCHS):
        epoch_start = time.time()
        model.train()
        total_loss = 0.0

        for sequences, labels, lengths in train_loader:
            sequences = sequences.to(device)
            labels = labels.to(device)

            # 获取包含所有 Stage 的预测结果
            stage_outputs, _ = model.forward_all_stages(sequences, lengths)
            loss, ce_loss, trans_loss = criterion(stage_outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()

            total_loss += loss.item()

        scheduler.step()

        avg_loss = total_loss / len(train_loader)
        writer.add_scalar('loss', avg_loss, epoch)
        writer.add_scalar('ce_loss', ce_loss.item(), epoch)
        writer.add_scalar('trans_loss', trans_loss.item(), epoch)
        writer.add_scalar('lr', optimizer.param_groups[0]['lr'], epoch)

        elapsed = time.time() - train_start
        eta = (elapsed / (epoch + 1)) * (NUM_EPOCHS - epoch - 1)
        eta_m, eta_s = divmod(int(eta), 60)
        eta_h, eta_m = divmod(eta_m, 60)
        lr_now = optimizer.param_groups[0]['lr']
        
        print(f'Epoch [{epoch+1}/{NUM_EPOCHS}], Loss: {avg_loss:.4f}, '
              f'CE: {ce_loss.item():.4f}, Trans: {trans_loss.item():.4f}, '
              f'LR: {lr_now:.6f}, Time: {time.time()-epoch_start:.1f}s, '
              f'ETA: {eta_h:02d}:{eta_m:02d}:{eta_s:02d}')

    total_time = time.time() - train_start
    print(f"\n🎉 训练完成! 总耗时: {total_time/60:.1f} 分钟")
    writer.close()
    return model

# ── 7. 测试评估 (Evaluation) ─────────────────────────────────────────────

def evaluate_Attn_TeCNO(model):
    demo_id_list = np.delete(np.arange(148), [80, 81, 92, 109, 112, 117, 122, 144, 145])

    test_states = load_test_state(without_quat=without_quat, resample=resample, demo_id_list=demo_id_list)
    test_labels = load_test_label(resample=resample, demo_id_list=demo_id_list)
    test_dataset = KinematicDataset(test_states, test_labels)
    test_loader = DataLoader(
        test_dataset, batch_size=BATCH_SIZE, shuffle=False,
        collate_fn=collate_fn,
    )
    print(f"Test demos: {len(test_dataset)}")

    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for sequences, labels, lengths in test_loader:
            sequences = sequences.to(device)
            # 推理时只取最后一层的结果
            logits = model(sequences, lengths)          # (B, T, C)
            preds = logits.argmax(dim=-1).cpu()

            for b in range(len(lengths)):
                L = lengths[b]
                all_preds.append(preds[b, :L].numpy())
                all_labels.append(labels[b, :L].numpy())

    all_preds = np.concatenate(all_preds)
    all_labels = np.concatenate(all_labels)

    valid = all_labels >= 0
    all_preds = all_preds[valid]
    all_labels = all_labels[valid]

    acc = accuracy_score(all_labels, all_preds)
    print(f"\nTest Accuracy: {acc * 100:.2f}%")
    print(classification_report(
        all_labels, all_preds,
        target_names=[f'Phase {i}' for i in range(NUM_CLASSES)],
        digits=4, zero_division=0,
    ))
    return acc

# ── Main ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # 1. 执行训练
    model = train_Attn_TeCNO()

    # 2. 执行初步评估验证
    print("\n" + "=" * 60)
    print("Evaluating Attn-TeCNO on test set …")
    print("=" * 60)
    evaluate_Attn_TeCNO(model)

    # 3. 保存模型
    dir_path = os.path.dirname(__file__)
    save_dir = os.path.join(dir_path, "TeCNO_model")
    os.makedirs(save_dir, exist_ok=True)
    model_save_path = os.path.join(save_dir, "attn_tecno_sequence_model.pth")
    save_model(model, model_save_path)