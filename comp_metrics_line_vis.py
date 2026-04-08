import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.signal import medfilt
import matplotlib.patches as mpatches
import matplotlib.lines as mlines

# ==========================================
# 1. 实验组划分与配置
# ==========================================
DATA_ROOT = "data_Phase" 
OUTPUT_DIR = "pupil_big_figures_report"

GLOBAL_ADAPTIVE_DEMOS = list(range(110, 115))
PHASED_ADAPTIVE_DEMOS = list(range(115, 120))
TARGET_DEMOS = GLOBAL_ADAPTIVE_DEMOS + PHASED_ADAPTIVE_DEMOS

FINE_PHASES = [1, 3, 5]  # 需要高亮显示的精细阶段

if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

# ==========================================
# 2. 数据处理与特征提取函数
# ==========================================
def mad_outlier_detection(series, constant=4):
    median = np.nanmedian(series)
    mad = np.nanmedian(np.abs(series - median))
    if mad == 0: return np.zeros_like(series, dtype=bool)
    return np.abs(series - median) > (constant * mad)

def preprocess_pupil_signal(signal):
    """ 瞳孔信号预处理: 去跳变，插值，平滑 """
    s = pd.Series(signal).replace(0, np.nan)
    s[mad_outlier_detection(np.abs(s.diff()), constant=4)] = np.nan
    s = s.interpolate(method='linear', limit=10)
    if s.count() > 5:
        filled = s.ffill().bfill()
        return medfilt(filled, kernel_size=5)
    return s.values

def _find_segments(label_array, target_label):
    """ 提取连续的阶段时间段 (start, end) """
    in_phase = (label_array == target_label)
    if not np.any(in_phase): return []
    changes = np.diff(in_phase.astype(np.int8))
    starts = list(np.where(changes == 1)[0] + 1)
    ends = list(np.where(changes == -1)[0] + 1)
    if in_phase[0]: starts = [0] + starts
    if in_phase[-1]: ends = ends + [len(label_array)]
    return list(zip(starts, ends))

def load_and_process_demo(demo_idx):
    """ 加载单条演示数据，计算基线、矫正值并映射精细阶段区间 """
    matching_folders = [d for d in os.listdir(DATA_ROOT) 
                        if os.path.isdir(os.path.join(DATA_ROOT, d)) and d.startswith(f"{demo_idx}_")]
    if not matching_folders:
        raise FileNotFoundError(f"找不到 Demo {demo_idx}")
        
    data_folder = os.path.join(DATA_ROOT, matching_folders[0])
    gaze_data = np.load(os.path.join(data_folder, "gaze_data.npy"))
    phase_labels = np.load(os.path.join(data_folder, "phase_labels.npy"))

    pL_smooth = preprocess_pupil_signal(gaze_data[:, 6])
    pR_smooth = preprocess_pupil_signal(gaze_data[:, 7])
    pupil_abs = (pL_smooth + pR_smooth) / 2.0

    eye_frames = len(pupil_abs)
    kine_frames = len(phase_labels)
    
    # 核心：计算时域映射缩放系数
    scale_eye = eye_frames / kine_frames

    # 计算基线期 (热身期)
    valid_starts = np.where(phase_labels >= 0)[0]
    if len(valid_starts) > 0:
        baseline_end_eye = int(valid_starts[0] * scale_eye)
    else:
        baseline_end_eye = int(eye_frames * 0.05)
    if baseline_end_eye < 30: baseline_end_eye = 30

    baseline_val = np.nanmean(pupil_abs[:baseline_end_eye])
    if np.isnan(baseline_val) or baseline_val == 0:
        raise ValueError("基线无法计算")

    pupil_corrected = pupil_abs - baseline_val

    # 映射精细阶段的区间 (Fine Phases)
    fine_intervals = []
    for target_phase in FINE_PHASES:
        segments = _find_segments(phase_labels, target_phase)
        for s_kine, e_kine in segments:
            if e_kine - s_kine > 5: # 过滤极短的噪声抖动
                s_eye = int(s_kine * scale_eye)
                e_eye = int(e_kine * scale_eye)
                fine_intervals.append((s_eye, e_eye))

    return pupil_abs, pupil_corrected, baseline_val, baseline_end_eye, fine_intervals

# ==========================================
# 3. 绘制多子图大图
# ==========================================
def plot_big_comparison_figures():
    sns.set_theme(style="whitegrid")
    processed_data = {}
    
    print("开始加载并预处理 10 条序列的数据...")
    for demo_idx in TARGET_DEMOS:
        try:
            processed_data[demo_idx] = load_and_process_demo(demo_idx)
            print(f"  [成功] 加载 Demo {demo_idx}")
        except Exception as e:
            print(f"  [失败] Demo {demo_idx}: {e}")

    # 准备两块大画布
    fig_abs, axes_abs = plt.subplots(5, 2, figsize=(24, 28))
    fig_abs.suptitle("Absolute Pupil Diameter Across Operating Phases (10 Sequences)", fontsize=28, fontweight='bold', y=0.98)
    
    fig_corr, axes_corr = plt.subplots(5, 2, figsize=(24, 28))
    fig_corr.suptitle("Baseline-Corrected Pupil Dilation (Cognitive Load) Across Phases", fontsize=28, fontweight='bold', y=0.98)

    for i, demo_idx in enumerate(TARGET_DEMOS):
        row, col = i // 2, i % 2
        ax_abs = axes_abs[row, col]
        ax_corr = axes_corr[row, col]
        
        group_name = "Global Adaptive" if demo_idx in GLOBAL_ADAPTIVE_DEMOS else "Phased Adaptive"
        color = 'steelblue' if demo_idx in GLOBAL_ADAPTIVE_DEMOS else 'darkgreen'

        if demo_idx in processed_data:
            p_abs, p_corr, b_val, b_end, fine_intervals = processed_data[demo_idx]
            frames = np.arange(len(p_abs))

            # ---------------------------------------------
            # 大图 A：原始绝对直径子图
            # ---------------------------------------------
            l_abs, = ax_abs.plot(frames, p_abs, color=color, lw=1.5)
            ax_abs.axvspan(0, b_end, color='gray', alpha=0.2)
            l_base = ax_abs.axhline(b_val, color='darkorange', ls='--', lw=2)
            
            # 绘制精细阶段背景高亮
            for s, e in fine_intervals:
                ax_abs.axvspan(s, e, color='mediumseagreen', alpha=0.25)
                
            ax_abs.set_title(f"Demo {demo_idx} ({group_name})", fontsize=16, fontweight='bold')
            ax_abs.set_ylabel("Absolute Diameter", fontsize=13)
            ax_abs.margins(x=0)

            # 自定义图例
            patch_base = mpatches.Patch(color='gray', alpha=0.2, label='Baseline Window')
            patch_fine = mpatches.Patch(color='mediumseagreen', alpha=0.25, label='Fine Phases (1, 3, 5)')
            ax_abs.legend(handles=[l_abs, l_base, patch_base, patch_fine], 
                          labels=['Smoothed Pupil', f'Baseline ({b_val:.2f})', 'Baseline Window', 'Fine Phases (1, 3, 5)'],
                          loc='upper right', framealpha=0.9)

            # ---------------------------------------------
            # 大图 B：基线矫正相对值子图
            # ---------------------------------------------
            l_corr, = ax_corr.plot(frames, p_corr, color='crimson', lw=1.5)
            ax_corr.fill_between(frames, 0, p_corr, where=(p_corr > 0), color='salmon', alpha=0.3)
            ax_corr.fill_between(frames, 0, p_corr, where=(p_corr <= 0), color='lightblue', alpha=0.3)
            
            ax_corr.axvspan(0, b_end, color='gray', alpha=0.2)
            ax_corr.axhline(0, color='black', ls='--', lw=1.5, alpha=0.7)
            
            # 绘制精细阶段背景高亮
            for s, e in fine_intervals:
                ax_corr.axvspan(s, e, color='mediumseagreen', alpha=0.25)
                
            ax_corr.set_title(f"Demo {demo_idx} ({group_name})", fontsize=16, fontweight='bold')
            ax_corr.set_ylabel("Δ Dilation", fontsize=13)
            ax_corr.margins(x=0)
            
            # 自定义图例
            ax_corr.legend(handles=[l_corr, patch_fine], 
                           labels=['Corrected Dilation', 'Fine Phases (1, 3, 5)'],
                           loc='upper right', framealpha=0.9)

            if row == 4:
                ax_abs.set_xlabel("Time (Eye Tracking Frames)", fontsize=14)
                ax_corr.set_xlabel("Time (Eye Tracking Frames)", fontsize=14)
        else:
            ax_abs.text(0.5, 0.5, f"Data Missing", ha='center', va='center', fontsize=14, color='red')
            ax_corr.text(0.5, 0.5, f"Data Missing", ha='center', va='center', fontsize=14, color='red')

    # 布局优化与保存
    fig_abs.tight_layout(rect=[0, 0, 1, 0.97])
    fig_corr.tight_layout(rect=[0, 0, 1, 0.97])

    path_abs = os.path.join(OUTPUT_DIR, "A_absolute_pupil_10demos.png")
    path_corr = os.path.join(OUTPUT_DIR, "B_corrected_pupil_10demos.png")
    
    fig_abs.savefig(path_abs, dpi=150)
    fig_corr.savefig(path_corr, dpi=150)
    
    plt.close(fig_abs)
    plt.close(fig_corr)
    
    print(f"\n>>> 生成完毕！图片已保存至目录: {OUTPUT_DIR}")

if __name__ == "__main__":
    plot_big_comparison_figures()