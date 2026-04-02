"""
compare_predictors.py
=====================
以 Dataset/state/2.txt 中的状态序列模拟实时帧输入，对比：
  - realtime_phase_predictor_tecno.py  （标准 TeCNO，确定性推理）
  - realtime_pre_dropout.py            （MC Dropout，不确定性感知）

可视化内容（4 行子图）：
  1. TeCNO        各阶段概率-时间帧曲线  +  Ground-Truth 阶段底色
  2. MC Dropout   各阶段概率-时间帧曲线  +  Ground-Truth 阶段底色
  3. 两种方法的香农信息熵对比（熵峰是否与阶段过渡边界对齐？）
  4. MC Dropout 输出的安全衰减系数 alpha

运行方式（在 Project 目录下）：
    python compare_predictors.py
"""

from __future__ import annotations

import os
import sys
import json
import importlib.util
import numpy as np
import matplotlib
matplotlib.use('Agg')          # 无显示环境下也能保存图片
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import to_rgba

# ── 路径配置 ──────────────────────────────────────────────────────────────────
PROJECT_DIR  = os.path.dirname(os.path.abspath(__file__))
STATE_FILE   = os.path.join(PROJECT_DIR, 'Dataset', 'state', '4.txt')
LABEL_FILE   = os.path.join(PROJECT_DIR, 'Dataset', 'label', '4_output_annotations.json')
MODEL_PATH   = os.path.join(PROJECT_DIR, 'Trajectory', 'TeCNO_model', 'tecno_sequence_model.pth')
OUTPUT_PNG   = os.path.join(PROJECT_DIR, 'predictor_comparison.png')

# ── 手术阶段名称 ──────────────────────────────────────────────────────────────
PHASE_NAMES = [
    'P0 Right Move',
    'P1 Pick Needle',
    'P2 Right Move2',
    'P3 Pass Needle',
    'P4 Left Move',
    'P5 Left Pick',
    'P6 Pull Thread',
]
NUM_CLASSES = len(PHASE_NAMES)

# 每个阶段对应的颜色（柔和色调）
PHASE_COLORS = [
    '#4e79a7', '#f28e2b', '#e15759', '#76b7b2',
    '#59a14f', '#edc948', '#b07aa1',
]


# ─────────────────────────────────────────────────────────────────────────────
# 1. 动态导入两个预测器模块（类名相同，需用 importlib 隔离）
# ─────────────────────────────────────────────────────────────────────────────

def _import_module(name: str, filepath: str):
    spec = importlib.util.spec_from_file_location(name, filepath)
    mod  = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod

tecno_mod   = _import_module('tecno_std',     os.path.join(PROJECT_DIR, 'realtime_phase_predictor_tecno.py'))
dropout_mod = _import_module('tecno_dropout', os.path.join(PROJECT_DIR, 'realtime_pre_dropout.py'))

TeCNOPredictor   = tecno_mod.TeCNOPhasePredictor
DropoutPredictor = dropout_mod.TeCNOPhasePredictor


# ─────────────────────────────────────────────────────────────────────────────
# 2. 加载状态序列与 Ground-Truth 标签
# ─────────────────────────────────────────────────────────────────────────────

def load_state_sequence(filepath: str) -> np.ndarray:
    """读取 state/*.txt，返回 (T, 16) 数组。"""
    rows = []
    with open(filepath) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            vals = [float(v) for v in line.split()]
            if len(vals) == 16:
                rows.append(vals)
    return np.array(rows, dtype=np.float32)


def load_gt_labels(filepath: str, num_frames: int) -> np.ndarray:
    """
    从 JSON 注释文件中读取 kine 帧级别的 Ground-Truth 标签。
    返回长度为 num_frames 的整数数组；未覆盖帧标记为 -1。
    """
    with open(filepath) as f:
        data = json.load(f)

    gt = np.full(num_frames, -1, dtype=int)
    for ann in data.get('annotations', []):
        ks = ann['kine_start']
        ke = ann['kine_end']
        lb = int(ann['label'])
        ks = max(0, ks - 1)          # JSON 使用 1-indexed，转为 0-indexed
        ke = min(num_frames - 1, ke - 1)
        gt[ks:ke + 1] = lb
    return gt


# ─────────────────────────────────────────────────────────────────────────────
# 3. 模拟实时推理
# ─────────────────────────────────────────────────────────────────────────────

def run_simulation(states: np.ndarray):
    """
    逐帧推送状态给两个预测器，收集概率、熵、alpha。
    """
    print("[Sim] 初始化 TeCNO 标准预测器 ...")
    pred_std = TeCNOPredictor(model_path=MODEL_PATH, min_frames=10, max_history=512)

    print("[Sim] 初始化 MC Dropout 预测器 ...")
    pred_mc  = DropoutPredictor(model_path=MODEL_PATH, min_frames=10, max_history=512,
                                mc_samples=20, entropy_lambda=2.0)

    T = len(states)
    probs_std = np.zeros((T, NUM_CLASSES), dtype=np.float32)
    probs_mc  = np.zeros((T, NUM_CLASSES), dtype=np.float32)
    alpha_mc  = np.zeros(T, dtype=np.float32)
    label_std = np.full(T, -1, dtype=int)
    label_mc  = np.full(T, -1, dtype=int)

    for i, row in enumerate(states):
        L_pos3    = row[0:3]
        L_gripper = float(row[7])
        R_pos3    = row[8:11]
        R_gripper = float(row[15])

        # 标准 TeCNO
        ph_s, pb_s = pred_std.update(L_pos3, R_pos3, L_gripper, R_gripper)
        label_std[i]  = ph_s
        probs_std[i]  = pb_s

        # MC Dropout
        ph_m, pb_m, al = pred_mc.update(L_pos3, R_pos3, L_gripper, R_gripper)
        label_mc[i]   = ph_m
        probs_mc[i]   = pb_m
        alpha_mc[i]   = al

        if (i + 1) % 50 == 0 or i == T - 1:
            print(f"  帧 {i+1:3d}/{T}  std_phase={ph_s}  mc_phase={ph_m}  alpha={al:.3f}")

    return probs_std, probs_mc, alpha_mc, label_std, label_mc


# ─────────────────────────────────────────────────────────────────────────────
# 4. 计算香农熵
# ─────────────────────────────────────────────────────────────────────────────

def shannon_entropy(probs: np.ndarray) -> np.ndarray:
    """输入 (T, C) 概率矩阵，返回 (T,) 熵向量（nats）。"""
    eps = 1e-10
    return -np.sum(probs * np.log(probs + eps), axis=1)


# ─────────────────────────────────────────────────────────────────────────────
# 5. 可视化
# ─────────────────────────────────────────────────────────────────────────────

def _shade_gt(ax, gt_labels: np.ndarray, alpha: float = 0.12):
    """在坐标轴上用阶段颜色背景色标注 Ground-Truth 区间。"""
    T = len(gt_labels)
    i = 0
    while i < T:
        lb = gt_labels[i]
        if lb < 0:
            i += 1
            continue
        j = i
        while j < T and gt_labels[j] == lb:
            j += 1
        color = PHASE_COLORS[lb % len(PHASE_COLORS)]
        ax.axvspan(i, j - 1, color=color, alpha=alpha, linewidth=0)
        i = j


def _draw_transitions(ax, gt_labels: np.ndarray):
    """在阶段切换处画竖虚线（Ground-Truth 过渡边界）。"""
    T = len(gt_labels)
    prev = gt_labels[0]
    for i in range(1, T):
        if gt_labels[i] != prev and gt_labels[i] >= 0 and prev >= 0:
            ax.axvline(x=i, color='grey', linewidth=0.9, linestyle='--', alpha=0.7)
        prev = gt_labels[i]


def plot_results(
    probs_std:  np.ndarray,
    probs_mc:   np.ndarray,
    alpha_mc:   np.ndarray,
    gt_labels:  np.ndarray,
    save_path:  str,
):
    T = len(gt_labels)
    frames = np.arange(T)
    max_entropy = np.log(NUM_CLASSES)

    entropy_std = shannon_entropy(probs_std)
    entropy_mc  = shannon_entropy(probs_mc)

    fig, axes = plt.subplots(
        4, 1, figsize=(16, 18),
        gridspec_kw={'height_ratios': [3, 3, 2, 1.5]},
        sharex=True
    )
    fig.suptitle(
        'Demo #2 — TeCNO vs MC-Dropout Realtime Phase Prediction\n'
        '(Shaded background = Ground-Truth phase; dashed lines = GT boundaries)',
        fontsize=13, y=0.99
    )

    # ── 子图 1：TeCNO 标准预测概率 ──────────────────────────────────────────
    ax = axes[0]
    _shade_gt(ax, gt_labels)
    _draw_transitions(ax, gt_labels)
    for c in range(NUM_CLASSES):
        ax.plot(frames, probs_std[:, c],
                color=PHASE_COLORS[c], linewidth=1.3,
                label=PHASE_NAMES[c])
    ax.set_ylim(-0.02, 1.05)
    ax.set_ylabel('Probability', fontsize=10)
    ax.set_title('① TeCNO (Deterministic) — Per-phase Probability', fontsize=11)
    ax.legend(loc='upper right', fontsize=7.5, ncol=4, framealpha=0.6)
    ax.grid(axis='y', linewidth=0.4, alpha=0.5)

    # ── 子图 2：MC Dropout 概率 ─────────────────────────────────────────────
    ax = axes[1]
    _shade_gt(ax, gt_labels)
    _draw_transitions(ax, gt_labels)
    for c in range(NUM_CLASSES):
        ax.plot(frames, probs_mc[:, c],
                color=PHASE_COLORS[c], linewidth=1.3,
                label=PHASE_NAMES[c])
    ax.set_ylim(-0.02, 1.05)
    ax.set_ylabel('Probability', fontsize=10)
    ax.set_title('② MC-Dropout — Per-phase Probability (mean over samples)', fontsize=11)
    ax.legend(loc='upper right', fontsize=7.5, ncol=4, framealpha=0.6)
    ax.grid(axis='y', linewidth=0.4, alpha=0.5)

    # ── 子图 3：香农熵对比 ──────────────────────────────────────────────────
    ax = axes[2]
    _shade_gt(ax, gt_labels, alpha=0.08)
    _draw_transitions(ax, gt_labels)
    ax.plot(frames, entropy_std, color='steelblue', linewidth=1.4,
            label='TeCNO entropy', alpha=0.85)
    ax.plot(frames, entropy_mc,  color='tomato',    linewidth=1.4,
            label='MC-Dropout entropy', alpha=0.85)
    ax.axhline(y=max_entropy, color='grey', linewidth=0.8, linestyle=':',
               label=f'Max entropy (ln {NUM_CLASSES} ≈ {max_entropy:.2f})')
    ax.set_ylim(-0.05, max_entropy * 1.15)
    ax.set_ylabel('Shannon Entropy (nats)', fontsize=10)
    ax.set_title('③ Prediction Entropy — does dropout reveal phase transition boundaries?',
                 fontsize=11)
    ax.legend(loc='upper right', fontsize=8, framealpha=0.7)
    ax.grid(axis='y', linewidth=0.4, alpha=0.5)

    # ── 子图 4：MC Dropout alpha 安全衰减系数 ──────────────────────────────
    ax = axes[3]
    _shade_gt(ax, gt_labels, alpha=0.10)
    _draw_transitions(ax, gt_labels)
    ax.fill_between(frames, alpha_mc, color='coral', alpha=0.45, linewidth=0)
    ax.plot(frames, alpha_mc, color='coral', linewidth=1.2,
            label='α (safety attenuation)')
    ax.axhline(y=1.0, color='grey', linewidth=0.7, linestyle=':')
    ax.set_ylim(-0.02, 1.08)
    ax.set_ylabel('α', fontsize=10)
    ax.set_xlabel('Frame Index', fontsize=10)
    ax.set_title('④ MC-Dropout Safety Coefficient α  (low α ↔ high uncertainty)', fontsize=11)
    ax.legend(loc='lower right', fontsize=8, framealpha=0.7)
    ax.grid(axis='y', linewidth=0.4, alpha=0.5)

    # ── 图例：GT 阶段颜色说明 ────────────────────────────────────────────────
    patches = [mpatches.Patch(color=PHASE_COLORS[i], alpha=0.4, label=PHASE_NAMES[i])
               for i in range(NUM_CLASSES)]
    fig.legend(handles=patches, title='GT Phase', loc='lower center',
               ncol=NUM_CLASSES, fontsize=8, framealpha=0.7,
               bbox_to_anchor=(0.5, 0.00))

    plt.tight_layout(rect=[0, 0.03, 1, 0.99])
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"\n[Plot] 图像已保存至: {save_path}")
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# 6. 统计输出
# ─────────────────────────────────────────────────────────────────────────────

def print_entropy_stats(
    entropy_std: np.ndarray,
    entropy_mc:  np.ndarray,
    gt_labels:   np.ndarray,
):
    """打印过渡边界处与稳定阶段中的熵均值对比。"""
    T = len(gt_labels)
    # 定义过渡区：GT 标签变化前后各 5 帧
    transition_mask = np.zeros(T, dtype=bool)
    for i in range(1, T):
        if gt_labels[i] != gt_labels[i - 1] and gt_labels[i] >= 0 and gt_labels[i - 1] >= 0:
            lo = max(0, i - 5)
            hi = min(T, i + 5)
            transition_mask[lo:hi] = True

    stable_mask = (~transition_mask) & (gt_labels >= 0)

    print("\n─── 信息熵统计 ───────────────────────────────────────")
    print(f"{'':20s}  {'TeCNO':>10s}  {'MC-Dropout':>12s}")
    if transition_mask.any():
        print(f"{'过渡区均熵':20s}  "
              f"{entropy_std[transition_mask].mean():10.4f}  "
              f"{entropy_mc[transition_mask].mean():12.4f}")
    if stable_mask.any():
        print(f"{'稳定区均熵':20s}  "
              f"{entropy_std[stable_mask].mean():10.4f}  "
              f"{entropy_mc[stable_mask].mean():12.4f}")
    print(f"{'全局最大熵':20s}  "
          f"{entropy_std.max():10.4f}  "
          f"{entropy_mc.max():12.4f}")
    print(f"{'全局均熵':20s}  "
          f"{entropy_std.mean():10.4f}  "
          f"{entropy_mc.mean():12.4f}")
    print("─────────────────────────────────────────────────────\n")

    if transition_mask.any() and stable_mask.any():
        ratio_std = entropy_std[transition_mask].mean() / (entropy_std[stable_mask].mean() + 1e-9)
        ratio_mc  = entropy_mc[transition_mask].mean()  / (entropy_mc[stable_mask].mean()  + 1e-9)
        print(f"过渡区/稳定区 熵比  TeCNO={ratio_std:.3f}  MC-Dropout={ratio_mc:.3f}")
        if ratio_mc > ratio_std:
            print("→ MC Dropout 的熵在过渡边界处相对更高，体现出更强的不确定性感知。")
        else:
            print("→ 两者过渡区熵比相近，MC Dropout 未显著增强边界处不确定性。")


# ─────────────────────────────────────────────────────────────────────────────
# main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print(" 实时预测对比：TeCNO vs MC-Dropout")
    print("=" * 60)

    # 加载数据
    print(f"\n[Data] 读取状态序列: {STATE_FILE}")
    states = load_state_sequence(STATE_FILE)
    T = len(states)
    print(f"[Data] 共 {T} 帧，特征维度 {states.shape[1]}")

    print(f"\n[Data] 读取 Ground-Truth 标签: {LABEL_FILE}")
    gt_labels = load_gt_labels(LABEL_FILE, T)
    phase_dist = {p: int((gt_labels == p).sum()) for p in range(NUM_CLASSES)}
    print(f"[Data] GT 阶段分布: {phase_dist}")

    # 运行推理
    print("\n[Sim] 开始逐帧模拟推理 ...")
    probs_std, probs_mc, alpha_mc, label_std, label_mc = run_simulation(states)

    # 统计
    entropy_std = shannon_entropy(probs_std)
    entropy_mc  = shannon_entropy(probs_mc)
    print_entropy_stats(entropy_std, entropy_mc, gt_labels)

    # 准确率（热身帧排除）
    valid = (label_std >= 0) & (gt_labels >= 0)
    if valid.any():
        acc_std = (label_std[valid] == gt_labels[valid]).mean()
        acc_mc  = (label_mc[valid]  == gt_labels[valid]).mean()
        print(f"Phase 预测准确率  TeCNO={acc_std*100:.1f}%  MC-Dropout={acc_mc*100:.1f}%")

    # 绘图
    print("\n[Plot] 正在绘制对比图 ...")
    plot_results(probs_std, probs_mc, alpha_mc, gt_labels, OUTPUT_PNG)


if __name__ == '__main__':
    main()
