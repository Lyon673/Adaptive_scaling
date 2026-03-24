"""
TeCNO_seg_train.py
==================
Multi-Stage Temporal Convolutional Network (MS-TCN) for surgical phase recognition.

Based on: Czempiel et al., "TeCNO: Surgical Phase Recognition with Multi-stage
Temporal Convolutional Networks", MICCAI 2020.

Architecture (no ResNet backbone — kinematic features directly as input):
  Stage 1:  1×1 conv → N causal dilated residual layers → 1×1 conv → logits
  Stage 2+: takes previous stage softmax → same structure → refined logits
  Loss:     weighted CE averaged across all stages + transition penalty

Input/Output interface identical to SequenceLabelingLSTM:
    forward(x, lengths) → logits (B, T, num_classes)

Feature vector: 16-dim (same as LSTM pipeline).
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

# ── Hyperparameters ──────────────────────────────────────────────────────
INPUT_SIZE = 16
HIDDEN_SIZE = 32        # compressed TCN feature channel width (small-data setting)
NUM_LAYERS = 8         # dilated residual layers per stage
NUM_STAGES = 2          # compressed: at most 2 stages
NUM_CLASSES = 7
KERNEL_SIZE = 3
DROPOUT = 0.2
BATCH_SIZE = 16
NUM_EPOCHS = 800
LEARNING_RATE = 5e-4
EMBED_DIM = 16
ATTN_DROPOUT = 0.2
LOCAL_WINDOW = 90

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# ── Dataset & DataLoader ─────────────────────────────────────────────────

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


# ── TeCNO Model ──────────────────────────────────────────────────────────

class CausalDilatedConv1d(nn.Module):
    """1D causal dilated convolution with left-padding (output length == input length)."""

    def __init__(self, in_ch, out_ch, kernel_size, dilation):
        super().__init__()
        self.pad_len = dilation * (kernel_size - 1)
        self.conv = nn.Conv1d(in_ch, out_ch, kernel_size, dilation=dilation)

    def forward(self, x):
        return self.conv(F.pad(x, (self.pad_len, 0)))


class DilatedResidualLayer(nn.Module):
    """
    D layer from TeCNO / MS-TCN:
        Z = ReLU(DilatedConv(D_{l-1}))
        D_l = D_{l-1} + Conv1x1(Z)
    """

    def __init__(self, hidden_dim, kernel_size, dilation, dropout):
        super().__init__()
        self.dconv = CausalDilatedConv1d(hidden_dim, hidden_dim, kernel_size, dilation)
        self.conv1x1 = nn.Conv1d(hidden_dim, hidden_dim, 1)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        z = self.drop(F.relu(self.dconv(x)))
        return x + self.conv1x1(z)


class TCNStage(nn.Module):
    """Single TCN prediction stage: 1×1 → N dilated residual layers → 1×1."""

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


class TeCNO(nn.Module):
    """
    Multi-Stage Temporal Convolutional Network for surgical phase recognition.

    Receptive field per stage = 2^(NUM_LAYERS+1) - 1 frames.
    With 10 layers: RF = 2047 frames per stage.

    Interface compatible with SequenceLabelingLSTM:
        forward(x, lengths) → (B, T, num_classes)
    """

    def __init__(self, input_size, hidden_size, num_layers, num_classes,
                 num_stages=2, kernel_size=3, dropout=0.5):
        super().__init__()
        assert input_size == 16, "This model expects 16-dim input (8 left + 8 right)."
        self.num_stages = min(num_stages, 2)
        self.num_classes = num_classes
        self.local_window = LOCAL_WINDOW

        # ── Dual-stream low-rank projection (8 -> 16) + LN ─────────────────
        self.left_proj = nn.Conv1d(8, EMBED_DIM, kernel_size=1)
        self.right_proj = nn.Conv1d(8, EMBED_DIM, kernel_size=1)
        self.left_ln = nn.LayerNorm(EMBED_DIM)
        self.right_ln = nn.LayerNorm(EMBED_DIM)

        # ── Heavily regularized cross-attention ─────────────────────────────
        self.left_from_right_attn = nn.MultiheadAttention(
            embed_dim=EMBED_DIM, num_heads=1, dropout=ATTN_DROPOUT, batch_first=True
        )
        self.right_from_left_attn = nn.MultiheadAttention(
            embed_dim=EMBED_DIM, num_heads=1, dropout=ATTN_DROPOUT, batch_first=True
        )
        self.gamma = nn.Parameter(torch.tensor(0.001))

        # ── Fuse (B,T,32) -> (B,16,T) for compressed MS-TCN ────────────────
        self.fusion = nn.Conv1d(EMBED_DIM * 2, EMBED_DIM, kernel_size=1)
        self.fusion_drop = nn.Dropout(dropout)

        # ── Compressed MS-TCN (<=2 stages, 32 channels) ────────────────────
        tcn_hidden = min(hidden_size, 32)
        self.stages = nn.ModuleList()
        for s in range(self.num_stages):
            in_dim = EMBED_DIM if s == 0 else num_classes
            self.stages.append(
                TCNStage(in_dim, tcn_hidden, num_classes, num_layers, kernel_size, dropout)
            )

    def _lengths_to_tensor(self, lengths, B, T, device):
        if lengths is None:
            return torch.full((B,), T, dtype=torch.long, device=device)
        if isinstance(lengths, torch.Tensor):
            return lengths.to(device=device, dtype=torch.long).clamp(min=1, max=T)
        return torch.tensor(lengths, dtype=torch.long, device=device).clamp(min=1, max=T)

    def _base_local_causal_mask(self, T, device):
        idx = torch.arange(T, device=device)
        q = idx[:, None]
        k = idx[None, :]
        causal_invalid = k > q
        local_invalid = (q - k) > self.local_window
        invalid = causal_invalid | local_invalid
        base = torch.zeros((T, T), device=device)
        base[invalid] = float("-inf")
        return base

    def _physical_bias(self, x):
        # x: (B,T,16), left xyz in [:, :, :3], right xyz in [:, :, 8:11]
        left_xyz = x[:, :, :3]
        right_xyz = x[:, :, 8:11]
        d_lr = torch.cdist(left_xyz, right_xyz, p=2)  # (B,T,T)
        d_rl = torch.cdist(right_xyz, left_xyz, p=2)  # (B,T,T)
        gamma = F.softplus(self.gamma)
        return -gamma * (d_lr ** 2), -gamma * (d_rl ** 2)

    def _cross_attend(self, q, kv, key_padding_mask, attn_additive_bias, attn_module, return_attn=False):
            B, T, _ = q.shape
            base = self._base_local_causal_mask(T, q.device).unsqueeze(0).expand(B, -1, -1)
            attn_mask = base + attn_additive_bias
            attn_mask = attn_mask.masked_fill(key_padding_mask[:, None, :], float("-inf"))

            all_inf = torch.isinf(attn_mask) & (attn_mask < 0)       
            row_all_inf = all_inf.all(dim=-1, keepdim=True)           
            safe = torch.zeros_like(attn_mask)
            safe[:, :, 0] = 0.0                                      
            attn_mask = torch.where(row_all_inf.expand_as(attn_mask), safe, attn_mask)

            # 【修改这里】：告诉 PyTorch 返回注意力权重
            out, attn_weights = attn_module(q, kv, kv, attn_mask=attn_mask, need_weights=return_attn)
            
            return out, attn_weights


    def _encode(self, x, lengths=None, return_attn=False):
        B, T, C = x.shape
        if C != 16:
            raise ValueError(f"Expected input feature dim 16, got {C}.")
        lens = self._lengths_to_tensor(lengths, B, T, x.device)
        valid = torch.arange(T, device=x.device)[None, :] < lens[:, None]  # (B,T)
        key_padding_mask = ~valid

        # split: left / right 8-dim streams
        left = x[:, :, :8].transpose(1, 2)   # (B,8,T)
        right = x[:, :, 8:].transpose(1, 2)   # (B,8,T)
        left = self.left_ln(self.left_proj(left).transpose(1, 2))   # (B,T,16)
        right = self.right_ln(self.right_proj(right).transpose(1, 2))  # (B,T,16)

        bias_lr, bias_rl = self._physical_bias(x)
        # 【修改这里】：接收返回的 weights
        left_ctx, left_attn = self._cross_attend(left, right, key_padding_mask, bias_lr, self.left_from_right_attn, return_attn)
        right_ctx, right_attn = self._cross_attend(right, left, key_padding_mask, bias_rl, self.right_from_left_attn, return_attn)

        q_mask = valid.unsqueeze(-1).to(left_ctx.dtype)
        left_ctx = left_ctx * q_mask
        right_ctx = right_ctx * q_mask

        # 【核心修改】：加上之前遗漏的残差连接！
        fused = torch.cat([left_ctx, right_ctx], dim=-1)      
        fused = self.fusion_drop(self.fusion(fused.transpose(1, 2)))  
        fused = x.permute(0, 2, 1) + fused  # <- 残差连接！

        # 【修改这里】：如果是可视化模式，把权重一起扔出去
        if return_attn:
            return fused, left_attn, right_attn
        return fused

    def forward(self, x, lengths=None, return_attn=False):
            if return_attn:
                h, left_attn, right_attn = self._encode(x, lengths, return_attn=True)
            else:
                h = self._encode(x, lengths)
                
            for i, stage in enumerate(self.stages):
                h = stage(h)
                if i < self.num_stages - 1:
                    h = F.softmax(h, dim=1)           
                    
            if return_attn:
                return h.permute(0, 2, 1), left_attn, right_attn
            return h.permute(0, 2, 1)

    def forward_all_stages(self, x, lengths=None):
        """Returns logits from ALL stages for multi-stage training loss."""
        h = self._encode(x, lengths)
        outputs = []
        for i, stage in enumerate(self.stages):
            h = stage(h)
            outputs.append(h.permute(0, 2, 1))    # (B, T, C)
            if i < self.num_stages - 1:
                h = F.softmax(h, dim=1)
        return outputs


# ── Multi-Stage Loss ─────────────────────────────────────────────────────

class MultiStageLoss(nn.Module):
    """
    CE loss averaged across all stages + transition penalty (same as
    SmoothPhaseLoss in LSTM pipeline, but aggregated over M stages).
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
        """
        stage_outputs: list of (B, T, C) from each stage
        labels:        (B, T)
        Returns: total_loss, avg_ce, avg_trans
        """
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


# ── Save / Load utilities ────────────────────────────────────────────────

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
        'model_type': 'TeCNO',
        'model_config': {
            'input_size': INPUT_SIZE,
            'hidden_size': HIDDEN_SIZE,
            'num_layers': NUM_LAYERS,
            'num_classes': NUM_CLASSES,
            'num_stages': NUM_STAGES,
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


def load_model(filepath, device_override=None):
    """Load a TeCNO checkpoint and return (model, config_dict)."""
    dev = device_override or device
    ckpt = torch.load(filepath, map_location=dev)
    cfg = ckpt['model_config']
    model = TeCNO(
        cfg['input_size'], cfg['hidden_size'], cfg['num_layers'], cfg['num_classes'],
        num_stages=cfg.get('num_stages', NUM_STAGES),
        kernel_size=cfg.get('kernel_size', KERNEL_SIZE),
        dropout=cfg.get('dropout', DROPOUT),
    )
    model.load_state_dict(ckpt['model_state_dict'])
    model.to(dev)
    model.eval()
    return model, ckpt


# ── Training ─────────────────────────────────────────────────────────────

def train_TeCNO():
    demo_id_list = np.delete(np.arange(148), [80, 81, 92, 109, 112, 117, 122, 144, 145])

    train_states = load_train_state(without_quat=without_quat, resample=resample, demo_id_list=demo_id_list)
    train_labels = load_train_label(resample=resample, demo_id_list=demo_id_list)
    train_dataset = KinematicDataset(train_states, train_labels)
    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True,
        collate_fn=collate_fn, drop_last=False,
    )
    print(f"Training demos: {len(train_dataset)}")

    model = TeCNO(
        INPUT_SIZE, HIDDEN_SIZE, NUM_LAYERS, NUM_CLASSES,
        num_stages=NUM_STAGES, kernel_size=KERNEL_SIZE, dropout=DROPOUT,
    ).to(device)
    print(model)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    rf = 2 ** (NUM_LAYERS + 1) - 1
    print(f"Trainable params: {n_params:,}   Receptive field per stage: {rf} frames")

    criterion = MultiStageLoss(
        num_classes=NUM_CLASSES, transition_weight=0.5, ignore_index=-1,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS)

    # TensorBoard
    dir_path = os.path.dirname(__file__)
    log_total_dir = os.path.join(dir_path, "logs", "TeCNO")
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

            stage_outputs = model.forward_all_stages(sequences, lengths)
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
    print(f"训练完成! 总耗时: {total_time/60:.1f} min")
    writer.close()
    return model


# ── Evaluation ───────────────────────────────────────────────────────────

def evaluate_TeCNO(model):
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
    model = train_TeCNO()

    print("\n" + "=" * 60)
    print("Evaluating on test set …")
    print("=" * 60)
    evaluate_TeCNO(model)

    dir_path = os.path.dirname(__file__)
    save_dir = os.path.join(dir_path, "TeCNO_model")
    os.makedirs(save_dir, exist_ok=True)
    model_save_path = os.path.join(save_dir, "tecno_sequence_model.pth")
    save_model(model, model_save_path)
