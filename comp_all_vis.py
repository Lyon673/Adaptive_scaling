"""
综合评估指标提取与高级可视化脚本 (Comprehensive Metrics Visualization)
整合了:
- 运动学与空间特征 (Smoothness, SampEn, SPARC, Gracefulness, 95% Ellipsoid Vol, Path Length)
- 视线与认知负荷特征 (Static/Transition Gaze Entropy, Pupil Diameter, IPA)
- [排版重构] 均值±标准差优雅地融入 X 轴组别标签，P 值标准置于横线上方
- [保留] 严格按照原始方式还原片段级散点数量，序列级自动去重
- [保留] 同步 comp_metricsGS.py 的 Ellipsoid_Volume_95 组内离群点剔除逻辑
- [极简版] 仅保留 Demo 级别的全局平均 (Overall) 可视化与统计检验，彻底杜绝伪重复
- [新增] 分组独立输出各项核心指标最大/最小的前 5 个 Demo ID，用于极值溯源
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.spatial.distance import pdist
from scipy.signal import medfilt
from scipy.stats import ttest_ind

# 尝试导入 IPA 计算库
try:
    from IPA_code.ipa import Pupil, ipa_cal
except ImportError:
    print("警告: 无法导入 ipa.py，IPA 指标将记录为 NaN。")
    ipa_cal = None

# ==========================================
# 1. 实验组划分与全局配置
# ==========================================

# GLOBAL_ADAPTIVE_DEMOS1 = list(range(0,5))+[10,12,13,14]+[22,23,25]+[31,32,33,35]+[41,42,43]+[47,48,49]+[53,54,55]+[59,60,61]+[65,66,67,68,69,70]
# PHASED_ADAPTIVE_DEMOS1 = [5,6,9]+[15,20] +[26,27,29]+[37,38,40]+[44,45]+[50,51,52]+[56,57,58]+[62,63,64]+[74,75,76]
GLOBAL_ADAPTIVE_DEMOS1 = list(range(0,5))+[10,12,13,14]+[22,23,25]+[31,32,33,35]+[41,42,43]+[47,48,49]+[53,54,55]+[59,60,61]+[65,66,67,68,69,70]
PHASED_ADAPTIVE_DEMOS1 = list(range(5,10))+[15,20] +[26,27,29]+[37,40]+[44,45,46]+[50,51,52]+[56,57,58]+[62,63,64]+[71,72,74,75,76]


GLOBAL_ADAPTIVE_DEMOS2 = list(range(0,5))+[10,12,13,14]+[22,23,25]+[32,35]+[41,42,43]+[47,48,49]
PHASED_ADAPTIVE_DEMOS2 = list(range(5,10))+[15,20] +[26,27,29]+[36,37,38,40]+[44,45,46]+[51,52]

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_ROOT  = os.path.join(SCRIPT_DIR, "data_Phase")
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "Essay_image_results")

if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

# ==========================================
# 2. 运动学与空间指标计算核心函数
# ==========================================
def calc_ellipsoid_volume_95(positions):
    if positions is None or len(positions) < 4:
        return np.nan
    cov_matrix = np.cov(positions, rowvar=False)
    det = np.linalg.det(cov_matrix)
    if det <= 0: return 0.0
    chi2_95_3d = 7.8147
    volume = (4.0 / 3.0) * np.pi * (chi2_95_3d ** 1.5) * np.sqrt(det)
    return volume

def calc_sample_entropy(positions, m=2, r_coeff=0.2):
    if positions is None or len(positions) < m + 2: return np.nan
    centroid = np.mean(positions, axis=0)
    U = np.linalg.norm(positions - centroid, axis=1)
    N = len(U)
    r = r_coeff * np.std(U)
    if r == 0: return 0.0 
    def _phi(m_len):
        try:
            x = np.array([U[i:i+m_len] for i in range(N - m_len + 1)])
            dists = pdist(x, metric='chebyshev')
            return np.sum(dists <= r) * 2 
        except: return np.nan
    N1, N2 = _phi(m), _phi(m + 1)
    if np.isnan(N1) or np.isnan(N2) or N1 == 0 or N2 == 0: return np.nan
    return -np.log(N2 / N1)

def calculate_sparc_from_data(data, padlevel=4, fc=10.0, amp_th=0.05):
    if data is None or len(data) < 5: return np.nan
    positions, time = data[:, :3], data[:, 3]
    dt_array = np.diff(time)
    valid_dt = dt_array[dt_array > 0]
    if len(valid_dt) == 0: return np.nan
    fs = 1.0 / np.mean(valid_dt) 
    velocities = np.gradient(positions, time, axis=0)
    speed = np.linalg.norm(velocities, axis=1) 
    movement = speed
    nfft = int(pow(2, np.ceil(np.log2(len(movement))) + padlevel))
    Mf = np.abs(np.fft.fft(movement, nfft))
    mf_max = Mf.max()
    if mf_max == 0: return np.nan
    Mf = Mf / mf_max 
    f = np.fft.fftfreq(nfft, 1.0/fs)
    idx = np.where((f >= 0) & (f <= fc))[0]
    Mf, f = Mf[idx], f[idx]
    idx_amp = np.where(Mf >= amp_th)[0]
    if len(idx_amp) == 0: return np.nan
    fc_idx = idx_amp[-1]
    Mf, f = Mf[:fc_idx+1], f[:fc_idx+1]
    df = f[1] - f[0]
    dM = np.diff(Mf)
    return -np.sum(np.sqrt(df**2 + dM**2))

def calc_all_kinematic_metrics(data):
    if data is None or len(data) < 5: 
        return (np.nan,) * 6
    
    positions, time = data[:, :3], data[:, 3]
    velocities = np.gradient(positions, time, axis=0)
    accelerations = np.gradient(velocities, time, axis=0)
    third_deriv = np.gradient(accelerations, time, axis=0)
    
    jerk_squared = np.sum(np.square(third_deriv), axis=1)
    integral = np.trapz(jerk_squared, x=time)
    duration = time[-1] - time[0]
    peak_velocity = np.max(np.linalg.norm(velocities, axis=1))
    
    if peak_velocity < 1e-10:
        S_scalar = np.nan
    else:
        phi = (np.power(duration, 5) / np.square(peak_velocity)) * integral
        S_scalar = np.log10(phi + 1e-10)
        
    path_length = np.sum(np.linalg.norm(np.diff(positions, axis=0), axis=1)) if len(positions) > 1 else 0.0
    
    cross_products = np.cross(velocities, accelerations)
    numerator = np.linalg.norm(cross_products, axis=1)
    denominator = np.power(np.linalg.norm(velocities, axis=1), 3)
    denominator = np.where(denominator == 0, np.inf, denominator)
    curvature = numerator / denominator
    gracefulness = np.median(np.log10(curvature + 1e-10))
        
    sparc = calculate_sparc_from_data(data)
    vol95 = calc_ellipsoid_volume_95(positions)
    sampen = calc_sample_entropy(positions)
    
    return S_scalar, sampen, vol95, sparc, gracefulness, path_length

# ==========================================
# 3. 眼动特征提取
# ==========================================
def mad_outlier_detection(series, constant=4):
    median = np.nanmedian(series)
    mad = np.nanmedian(np.abs(series - median))
    if mad == 0: return np.zeros_like(series, dtype=bool)
    return np.abs(series - median) > (constant * mad)

def preprocess_pupil_signal(signal):
    s = pd.Series(signal).replace(0, np.nan)
    dilation_speed = np.abs(s.diff())
    s[mad_outlier_detection(dilation_speed, constant=4)] = np.nan
    s = s.interpolate(method='linear', limit=10)
    if s.count() > 5:
        filled = s.ffill().bfill()
        return medfilt(filled, kernel_size=5)
    return s.values

def calculate_entropy(gaze_points, bins=16):
    mask = (gaze_points[:, 0] > 0) & (gaze_points[:, 1] > 0)
    valid_points = gaze_points[mask]
    if len(valid_points) < 10: return np.nan
    hist, _, _ = np.histogram2d(
        valid_points[:, 0], valid_points[:, 1], bins=bins,
        range=[[valid_points[:, 0].min(), valid_points[:, 0].max()],
               [valid_points[:, 1].min(), valid_points[:, 1].max()]]
    )
    probs = hist.flatten() / np.sum(hist)
    probs = probs[probs > 0]
    return -np.sum(probs * np.log2(probs))

def calculate_gte(points, bins=8):
    mask = (points[:, 0] > 0) & (points[:, 1] > 0)
    valid = points[mask]
    if len(valid) < 20: return np.nan
    xmin, xmax = valid[:, 0].min(), valid[:, 0].max()
    ymin, ymax = valid[:, 1].min(), valid[:, 1].max()
    if xmax == xmin or ymax == ymin: return 0.0
    cols = np.clip(((valid[:, 0] - xmin) / (xmax - xmin) * (bins - 1)).astype(int), 0, bins - 1)
    rows = np.clip(((valid[:, 1] - ymin) / (ymax - ymin) * (bins - 1)).astype(int), 0, bins - 1)
    states = rows * bins + cols
    num_states = bins * bins
    p_i = np.bincount(states, minlength=num_states) / len(states)
    tm = np.zeros((num_states, num_states))
    for t in range(len(states) - 1):
        tm[states[t], states[t + 1]] += 1
    gte = 0.0
    for i in range(num_states):
        if p_i[i] > 0 and np.sum(tm[i, :]) > 0:
            p_ij = tm[i, :] / np.sum(tm[i, :])
            p_ij_nz = p_ij[p_ij > 0]
            gte += p_i[i] * (-np.sum(p_ij_nz * np.log2(p_ij_nz)))
    return gte

# ==========================================
# 4. 融合数据收集引擎
# ==========================================
def extract_all_metrics(global_demos, phased_demos, target_phases):
    if not os.path.exists(DATA_ROOT): return pd.DataFrame()
    demo_folders = sorted(
        [d for d in os.listdir(DATA_ROOT) if os.path.isdir(os.path.join(DATA_ROOT, d)) and d.split('_')[0].isdigit()],
        key=lambda x: int(x.split('_')[0])
    )
    
    results = []
    
    for folder_name in demo_folders:
        demo_idx = int(folder_name.split('_')[0])
        data_folder = os.path.join(DATA_ROOT, folder_name)

        if demo_idx in global_demos: group = "Global Adaptive"
        elif demo_idx in phased_demos: group = "Phased Adaptive"
        else: continue

        demo_segments = []
        total_path = 0.0
        
        try:
            phase_labels = np.load(os.path.join(data_folder, "phase_labels.npy"))
            def _load_pos(folder, arm):
                for fname in (f"{arm}psm_position.npy", f"{arm}psm_pose.npy"):
                    p = os.path.join(folder, fname)
                    if os.path.exists(p):
                        arr = np.load(p)
                        if arr.ndim == 2 and arr.shape[1] >= 4:
                            t = np.maximum.accumulate(arr[:, 3])
                            arr[:, 3] = t - t[0]
                            return arr[:, :4]
                return None

            L_pos, R_pos = _load_pos(data_folder, "L"), _load_pos(data_folder, "R")
            
            # 提取指定阶段
            for label in target_phases:
                in_phase = (phase_labels == label)
                if not np.any(in_phase): continue
                changes = np.diff(in_phase.astype(np.int8))
                starts = list(np.where(changes == 1)[0] + 1)
                ends   = list(np.where(changes == -1)[0] + 1)
                if in_phase[0]: starts = [0] + starts
                if in_phase[-1]: ends = ends + [len(phase_labels)]
                
                for s_kine, e_kine in zip(starts, ends):
                    if e_kine - s_kine < 5: continue
                    target_arm = "R" if label in [1, 3] else "L"
                    data_seg = R_pos[s_kine:e_kine] if target_arm == "R" else L_pos[s_kine:e_kine]
                    
                    S, sampen, vol, sparc, grace, path_len = calc_all_kinematic_metrics(data_seg)
                    total_path += path_len
                    if not np.isnan(S):
                        demo_segments.append({
                            "Smoothness": S,
                            "Kinematic_SampEn": sampen,
                            "Ellipsoid_Volume_95": vol * 1e6, # mm³
                            "SPARC": sparc,
                            "Gracefulness": grace
                        })
        except: pass

        stat_ent, trans_ent, pupil_mean, ipa_val = np.nan, np.nan, np.nan, np.nan
        try:
            gaze_pos = np.load(os.path.join(data_folder, "gazepoint_position_data.npy"))
            if gaze_pos.ndim == 3: gaze_pos = gaze_pos.reshape(gaze_pos.shape[0], -1)
            if gaze_pos.ndim == 1: gaze_pos = gaze_pos.reshape(-1, 2)
            stat_ent, trans_ent = calculate_entropy(gaze_pos), calculate_gte(gaze_pos)
        except: pass
        try:
            pupilL_data = np.load(os.path.join(data_folder, "pupilL_data.npy"))
            pupilR_data = np.load(os.path.join(data_folder, "pupilR_data.npy"))
            if pupilL_data.ndim == 2 and pupilL_data.shape[1] >= 2 and pupilL_data.shape[0] > 0:
                pL = preprocess_pupil_signal(pupilL_data[:, 0])
                pR = preprocess_pupil_signal(pupilR_data[:, 0])
                pupil_mean = np.nanmean((pL + pR) / 2.0)

                if ipa_cal is not None:
                    eye_times = pupilL_data[:, 1] - pupilL_data[0, 1]
                    valid_mask = ~np.isnan(pL) & ~np.isnan(pR)
                    if np.sum(valid_mask) > 64:
                        pupils_L = [Pupil(val, t) for val, t in zip(pL[valid_mask], eye_times[valid_mask])]
                        pupils_R = [Pupil(val, t) for val, t in zip(pR[valid_mask], eye_times[valid_mask])]
                        res_L, res_R = ipa_cal(pupils_L, None), ipa_cal(pupils_R, None)
                        ipa_val = np.nanmean([res_L[0] if res_L else np.nan, res_R[0] if res_R else np.nan])
        except: pass

        base_dict = {
            "Demo_ID": demo_idx,
            "Group": group,
            "Path_Length": total_path if total_path > 0 else np.nan,
            "Static_Gaze_Entropy": stat_ent,
            "Trans_Gaze_Entropy": trans_ent,
            "Pupil_Diameter": pupil_mean,
            "IPA": ipa_val
        }

        if len(demo_segments) > 0:
            for seg in demo_segments:
                row = base_dict.copy()
                row.update(seg)
                results.append(row)
        else:
            row = base_dict.copy()
            row.update({
                "Smoothness": np.nan, "Kinematic_SampEn": np.nan,
                "Ellipsoid_Volume_95": np.nan, "SPARC": np.nan, "Gracefulness": np.nan
            })
            results.append(row)

    return pd.DataFrame(results)

# ==========================================
# 5. 绘图引擎与美化排版
# ==========================================
def _format_num(val):
    if pd.isna(val): return "NaN"
    if abs(val) >= 10000 or (abs(val) < 0.01 and val != 0):
        return f"{val:.2e}"
    elif abs(val) >= 100:
        return f"{val:.1f}"
    else:
        return f"{val:.3f}"

def draw_styled_boxplot(ax, df, x_col, y_col, title, is_seq_level=False, has_scatter=True, has_statistic=True):
    # 离群点剔除逻辑
    if y_col == "Ellipsoid_Volume_95":
        df_filtered_list = []
        for grp in ['Global Adaptive', 'Phased Adaptive']:
            df_grp = df[df['Group'] == grp]
            if not df_grp.empty:
                Q1 = df_grp[y_col].quantile(0.25)
                Q3 = df_grp[y_col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                df_filtered = df_grp[(df_grp[y_col] >= lower_bound) & (df_grp[y_col] <= upper_bound)]
                df_filtered_list.append(df_filtered)
        if df_filtered_list:
            df = pd.concat(df_filtered_list)
        
    order = ['Global Adaptive', 'Phased Adaptive']
    palette = ['#C96566', '#3F6FA8']

    sns.boxplot(
        data=df, x=x_col, y=y_col, order=order, ax=ax,
        showfliers=False, width=0.4,
        boxprops={'facecolor': 'none', 'linewidth': 1.5},
        medianprops={'linewidth': 2.5},
        whiskerprops={'linewidth': 1.5},
        capprops={'linewidth': 1.5}
    )

    for i, patch in enumerate(ax.patches):
        c = palette[i % len(palette)]
        patch.set_edgecolor(c)
        patch.set_facecolor('none')
    
    for i, line in enumerate(ax.lines):
        box_idx = i // 5
        if box_idx < len(palette):
            line.set_color(palette[box_idx])

    if has_scatter:
        sns.stripplot(
            data=df, x=x_col, y=y_col, order=order, ax=ax,
            hue=x_col, palette=palette, legend=False, 
            size=4.0, alpha=0.5, jitter=0.15, zorder=1
        )

    label_g1 = "Global Adaptive"
    label_g2 = "Phased Adaptive"

    data_g1 = df[df[x_col] == order[0]][y_col].dropna()
    data_g2 = df[df[x_col] == order[1]][y_col].dropna()

    if len(data_g1) >= 2 and len(data_g2) >= 2:
        stat, p_val = ttest_ind(data_g1, data_g2, equal_var=False)
        
        mean1, std1 = data_g1.mean(), data_g1.std()
        mean2, std2 = data_g2.mean(), data_g2.std()
        
        label_g1 = f"Global Adaptive\n({_format_num(mean1)} ± {_format_num(std1)})"
        label_g2 = f"Phased Adaptive\n({_format_num(mean2)} ± {_format_num(std2)})"

        if np.isnan(p_val):
            sig_text = "p=NaN"
        else:
            if p_val < 0.001: sig = '***'
            elif p_val < 0.01: sig = '**'
            elif p_val < 0.05: sig = '*'
            else: sig = ''
            
            if p_val < 0.001 and p_val > 0:
                p_str = f"{p_val:.2e}"
            else:
                p_str = f"{p_val:.3f}"
            
            if p_val < 0.1:
                sig_text = f"p={p_str} {sig}"
            else:
                sig_text = f"p>0.05"
        
        y_max = df[y_col].max()
        y_min = df[y_col].min()
        y_range = y_max - y_min if y_max != y_min else 1.0
        h = y_range * 0.05
        
        if has_statistic:
            y_bracket_bottom = y_max + h * 0.8
            y_bracket_top = y_max + h * 1.5
            ax.plot([0, 0, 1, 1], [y_bracket_bottom, y_bracket_top, y_bracket_top, y_bracket_bottom], 
                    lw=1.2, color='#333333')
            
            ax.text(0.5, y_bracket_top + h * 0.2, sig_text, ha='center', va='bottom', 
                    color='#222222', fontsize=12, fontweight='bold')
        
            ax.set_ylim(y_min - y_range * 0.1, y_bracket_top + h * 2.0)

    ax.set_title(title, fontsize=15, fontweight='bold', pad=12, color='#222222')
    ax.set_xlabel('')
    ax.set_ylabel('')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color('#BBBBBB')
    ax.spines['bottom'].set_color('#BBBBBB')
    ax.grid(axis='y', linestyle=':', linewidth=1, alpha=0.5, color='#CCCCCC')
    
    ax.set_xticks([0, 1])
    ax.set_xticklabels([label_g1, label_g2], fontsize=12)
    
    if df[y_col].max() > 1000:
        ax.ticklabel_format(axis='y', style='sci', scilimits=(0,0))


def plot_overall_figures(df1, df2):
    """
    绘制极简的 3x2 全局平均 (Overall) 对比图。
    """
    fig, axes = plt.subplots(3, 2, figsize=(12, 14))
    
    draw_styled_boxplot(axes[0, 0], df1, 'Group', 'Smoothness', 'Median Log Jerk↓', is_seq_level=False)
    draw_styled_boxplot(axes[0, 1], df1, 'Group', 'Kinematic_SampEn', 'Kinematic Sample Entropy↓', is_seq_level=False)
    
    draw_styled_boxplot(axes[1, 0], df1, 'Group', 'Path_Length', 'Total Path Length↓', is_seq_level=False)
    draw_styled_boxplot(axes[1, 1], df1, 'Group', 'Ellipsoid_Volume_95', '95% Confidence Ellipsoid Vol↓', is_seq_level=False)
    
    draw_styled_boxplot(axes[2, 0], df1, 'Group', 'SPARC', 'SPARC of PSM Speed↑', is_seq_level=False)
    draw_styled_boxplot(axes[2, 1], df2, 'Group', 'IPA', 'Index of Pupillary Activity↓', is_seq_level=False)
    
    plt.tight_layout(pad=2.0)
    
    save_path = os.path.join(OUTPUT_DIR, f"Kinematics&Eyes_Overall_Boxplots.pdf")
    plt.savefig(save_path, format="pdf", dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"✅ 全局平均对比图 (Overall) 已保存至: {save_path}")


# ==========================================
# 6. 分组极值输出模块 (独立输出各组的Top5和Bottom5)
# ==========================================
def print_extreme_demos(df, metric_col, n=5):
    """提取并在控制台分组输出指定指标的最大与最小的 5 个样本"""
    if metric_col not in df.columns: return
    df_clean = df.dropna(subset=[metric_col])
    if df_clean.empty: return
    
    print(f"\n[ 核心指标: {metric_col} ]")
    
    for group_name in ['Global Adaptive', 'Phased Adaptive']:
        df_grp = df_clean[df_clean['Group'] == group_name]
        if df_grp.empty: continue
        
        sorted_df = df_grp.sort_values(by=metric_col, ascending=True)
        
        print(f"\n  ■ {group_name}")
        print(f"    ▼ 数值最小的 {n} 个 Demo (Bottom {n}):")
        for _, row in sorted_df.head(n).iterrows():
            print(f"      - Demo {int(row['Demo_ID']):>3d}: {row[metric_col]:.4e}")
            
        print(f"    ▲ 数值最大的 {n} 个 Demo (Top {n}):")
        for _, row in sorted_df.tail(n)[::-1].iterrows():
            print(f"      - Demo {int(row['Demo_ID']):>3d}: {row[metric_col]:.4e}")


if __name__ == "__main__":
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "axes.labelsize": 11,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
    })

    print(">>>> 开始提取全阶段数据...")
    df1_all = extract_all_metrics(GLOBAL_ADAPTIVE_DEMOS1, PHASED_ADAPTIVE_DEMOS1, target_phases=[1, 3, 5])
    df2_all = extract_all_metrics(GLOBAL_ADAPTIVE_DEMOS2, PHASED_ADAPTIVE_DEMOS2, target_phases=[1, 3, 5])
    
    print("\n>>>> 正在对所有提取片段执行 Demo 级别的降维均值聚合...")
    # 强制将同一个手术视频内的多次操作片段坍缩为一个统一的数据点
    df1_avg = df1_all.groupby(['Demo_ID', 'Group'], as_index=False).mean(numeric_only=True)
    df2_avg = df2_all.groupby(['Demo_ID', 'Group'], as_index=False).mean(numeric_only=True)
        
    print("\n>>>> 聚合完成，正在生成全局对比图表...")
    plot_overall_figures(df1_avg, df2_avg)
    
    print("\n" + "="*50)
    print(">>>> 执行极值分析: 分组独立提取各项指标 Top 5 与 Bottom 5")
    print("="*50)
    
    metrics_to_check_df1 = ['Smoothness', 'Kinematic_SampEn', 'Path_Length', 'Ellipsoid_Volume_95', 'SPARC']
    for metric in metrics_to_check_df1:
        print_extreme_demos(df1_avg, metric)
        
    metrics_to_check_df2 = ['IPA']
    for metric in metrics_to_check_df2:
        print_extreme_demos(df2_avg, metric)
        
    print("\n✅ 所有分析与排查完毕。")