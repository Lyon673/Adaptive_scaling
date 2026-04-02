import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.signal import medfilt

# ==========================================
# 1. 实验组划分配置 (请根据实际采集记录填写)
# ==========================================
# TODO: 请填写属于“全局自适应缩放”方案的 demo_idx 列表
GLOBAL_ADAPTIVE_DEMOS = [1, 2, 3, 4, 5] 

# TODO: 请填写属于“分阶段自适应缩放”方案的 demo_idx 列表
PHASED_ADAPTIVE_DEMOS = [6, 7, 8, 9, 10]

# ==========================================
# 参数与路径配置
# ==========================================
VIDEO_FPS = 30
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
LABEL_DIR = os.path.join(SCRIPT_DIR, "phase_label")
DATA_ROOT = os.path.join(SCRIPT_DIR, "data")
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "scaling_comparison_report")

if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

# ==========================================
# 2. 核心评估指标计算函数
# ==========================================

def calculate_jerk(velocity_series, dt):
    """
    计算 Mean Squared Jerk (MSJ)。
    基于速度求一阶导数得到加速度，再求一阶导数得到加加速度(Jerk)。
    """
    if len(velocity_series) < 3 or dt <= 0: return np.nan
    acc = np.diff(velocity_series) / dt
    jerk = np.diff(acc) / dt
    return np.nanmean(jerk**2)

def calculate_sparc(velocity_series, fs, padlevel=4, fc=10.0, amp_th=0.05):
    """
    计算速度谱弧长 (Spectral Arc Length, SPARC)，用于评估运动平滑度。
    值越小(越接近负无穷)表示越不平滑，值越大(越接近0)表示越平滑。
    """
    movement = np.array(velocity_series)
    if len(movement) < 10: return np.nan
    
    nfft = int(pow(2, np.ceil(np.log2(len(movement))) + padlevel))
    Mf = np.abs(np.fft.fft(movement, nfft))
    Mf = Mf / Mf.max() # 归一化
    
    f = np.fft.fftfreq(nfft, 1.0/fs)
    idx = np.where((f >= 0) & (f <= fc))[0]
    Mf, f = Mf[idx], f[idx]
    
    idx_amp = np.where(Mf >= amp_th)[0]
    if len(idx_amp) == 0: return np.nan
    fc_idx = idx_amp[-1]
    
    Mf, f = Mf[:fc_idx+1], f[:fc_idx+1]
    
    df = f[1] - f[0]
    dM = np.diff(Mf)
    arc_length = -np.sum(np.sqrt(df**2 + dM**2))
    return arc_length

def mad_outlier_detection(series, constant=4):
    median = np.nanmedian(series)
    mad = np.nanmedian(np.abs(series - median))
    if mad == 0: return np.zeros_like(series, dtype=bool)
    return np.abs(series - median) > (constant * mad)

def preprocess_pupil_signal(signal):
    """ 瞳孔信号预处理 """
    s = pd.Series(signal).replace(0, np.nan)
    dilation_speed = np.abs(s.diff()) 
    s[mad_outlier_detection(dilation_speed, constant=4)] = np.nan
    s = s.interpolate(method='linear', limit=10)
    if s.count() > 5:
        filled = s.ffill().bfill()
        return medfilt(filled, kernel_size=5)
    return s.values

def calculate_entropy(gaze_points, bins=16):
    """ 计算静态熵 """
    mask = (gaze_points[:, 0] > 0) & (gaze_points[:, 1] > 0)
    valid_points = gaze_points[mask]
    if len(valid_points) < 10: return np.nan

    hist, _, _ = np.histogram2d(
        valid_points[:, 0], valid_points[:, 1], bins=bins, 
        range=[[valid_points[:,0].min(), valid_points[:,0].max()], 
               [valid_points[:,1].min(), valid_points[:,1].max()]]
    )
    probs = hist.flatten() / np.sum(hist)
    probs = probs[probs > 0]
    return -np.sum(probs * np.log2(probs))

def calculate_gte(points, bins=8):
    """ 计算转移熵 (GTE) """
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
        tm[states[t], states[t+1]] += 1
        
    gte = 0.0
    for i in range(num_states):
        if p_i[i] > 0 and np.sum(tm[i, :]) > 0:
            p_ij = tm[i, :] / np.sum(tm[i, :])
            p_ij_nz = p_ij[p_ij > 0]
            gte += p_i[i] * (-np.sum(p_ij_nz * np.log2(p_ij_nz)))
    return gte

def get_cohens_d(g1, g2):
    n1, n2 = len(g1), len(g2)
    v1, v2 = np.var(g1, ddof=1), np.var(g2, ddof=1)
    pooled_std = np.sqrt(((n1 - 1) * v1 + (n2 - 1) * v2) / (n1 + n2 - 2))
    return (np.mean(g1) - np.mean(g2)) / pooled_std if pooled_std != 0 else 0

# ==========================================
# 3. 数据加载与指标提取流水线
# ==========================================

def find_data_folder(demo_idx):
    if not os.path.exists(DATA_ROOT): return None
    folders = [f for f in os.listdir(DATA_ROOT) if f.startswith(f"{demo_idx}_data")]
    return os.path.join(DATA_ROOT, folders[0]) if folders else None

def extract_fine_phase_metrics():
    results = []
    json_files = [f for f in os.listdir(LABEL_DIR) if f.endswith('.json')]
    print(f"开始提取精细阶段指标，共发现 {len(json_files)} 个标注文件...")

    for j_file in json_files:
        try:
            with open(os.path.join(LABEL_DIR, j_file), 'r') as f:
                meta = json.load(f)
            
            demo_idx = meta['demo_idx']
            
            # 分组判断
            if demo_idx in GLOBAL_ADAPTIVE_DEMOS:
                group = "Global_Adaptive"
            elif demo_idx in PHASED_ADAPTIVE_DEMOS:
                group = "Phased_Adaptive"
            else:
                continue # 既不属于全局也不属于分阶段，跳过该Demo

            data_folder = find_data_folder(demo_idx)
            if not data_folder: continue

            # 载入数据
            L_vel = np.load(os.path.join(data_folder, "Lpsm_velocity_filtered.npy"))
            R_vel = np.load(os.path.join(data_folder, "Rpsm_velocity_filtered.npy"))
            gaze_pos = np.load(os.path.join(data_folder, "gazepoint_position_data.npy"))
            gaze_data = np.load(os.path.join(data_folder, "gaze_data.npy"))
            
            # 自动修复 gaze_pos 形状
            if gaze_pos.ndim == 3: gaze_pos = gaze_pos.reshape(gaze_pos.shape[0], -1)
            if gaze_pos.ndim == 1 and len(gaze_pos) % 2 == 0: gaze_pos = gaze_pos.reshape(-1, 2)

            kine_frames = meta['kine_frames']
            kine_duration = kine_frames / VIDEO_FPS
            kine_hz = kine_frames / kine_duration # 近似采集频率
            eye_hz = len(gaze_data) / kine_duration
            
            scale_eye = len(gaze_data) / kine_frames

            # 预处理瞳孔数据
            pL = preprocess_pupil_signal(gaze_data[:, 6])
            pR = preprocess_pupil_signal(gaze_data[:, 7])
            pupil_avg = (pL + pR) / 2.0 # 使用双眼平均瞳孔直径

            for ann in meta['annotations']:
                label = int(ann['label'])
                # 【核心】：只提取奇数标签（即 Fine 阶段）
                if label % 2 == 0: 
                    continue
                
                # 运动学切片 (Kine)
                s_kine, e_kine = int(ann['kine_start']), int(ann['kine_end'])
                if e_kine <= s_kine: continue
                seg_L_vel = L_vel[s_kine:e_kine]
                seg_R_vel = R_vel[s_kine:e_kine]

                # 眼动切片 (Eye)
                s_eye, e_eye = int(s_kine * scale_eye), int(e_kine * scale_eye)
                if e_eye <= s_eye: continue
                seg_gaze_pos = gaze_pos[s_eye:e_eye]
                seg_pupil = pupil_avg[s_eye:e_eye]

                # ---------------- 计算各项指标 ----------------
                # 1. 运动平滑度：双臂 Jerk 平均值
                jerk_L = calculate_jerk(seg_L_vel, 1.0/kine_hz)
                jerk_R = calculate_jerk(seg_R_vel, 1.0/kine_hz)
                mean_jerk = np.nanmean([jerk_L, jerk_R])

                # 2. 运动平滑度：双臂 SPARC 平均值
                sparc_L = calculate_sparc(seg_L_vel, kine_hz)
                sparc_R = calculate_sparc(seg_R_vel, kine_hz)
                mean_sparc = np.nanmean([sparc_L, sparc_R])

                # 3. 认知负荷：瞳孔直径变异率 (Coefficient of Variation)
                # 使用标准差除以均值作为变异率，消除绝对大小的个体差异
                pupil_cv = np.nanstd(seg_pupil) / np.nanmean(seg_pupil) if np.nanmean(seg_pupil) != 0 else np.nan

                # 4. 认知负荷：注视静态熵 (Stationary Entropy)
                stat_entropy = calculate_entropy(seg_gaze_pos)

                # 5. 认知负荷：注视转移熵 (Transition Entropy)
                trans_entropy = calculate_gte(seg_gaze_pos)

                # 将结果记录到列表
                results.append({
                    "Demo_ID": demo_idx,
                    "Group": group,
                    "Phase_Label": label,
                    "Mean_Jerk": mean_jerk,
                    "SPARC": mean_sparc,
                    "Pupil_Variation_Rate": pupil_cv,
                    "Static_Entropy": stat_entropy,
                    "Transition_Entropy": trans_entropy
                })

        except Exception as e:
            print(f"处理文件 {j_file} 时出错: {e}")

    return pd.DataFrame(results)

# ==========================================
# 4. 统计检验与结果可视化
# ==========================================

def run_statistical_analysis(df):
    metrics = {
        "Mean_Jerk": ("Mean Squared Jerk (Lower is smoother)", "运动平滑度(加加速度)"),
        "SPARC": ("Spectral Arc Length (Higher is smoother)", "运动平滑度(速度谱弧长)"),
        "Pupil_Variation_Rate": ("Pupil Coefficient of Variation", "认知负荷(瞳孔变异率)"),
        "Static_Entropy": ("Stationary Gaze Entropy (bits)", "认知负荷(静态熵)"),
        "Transition_Entropy": ("Gaze Transition Entropy (bits)", "认知负荷(转移熵)")
    }

    report_text = "全局自适应缩放 vs 分阶段自适应缩放 (精细阶段对比报告)\n"
    report_text += "=" * 65 + "\n\n"

    plt.figure(figsize=(15, 10))
    sns.set_theme(style="whitegrid")

    for i, (col, (y_label, title)) in enumerate(metrics.items(), 1):
        df_valid = df.dropna(subset=[col])
        group_g = df_valid[df_valid['Group'] == 'Global_Adaptive'][col]
        group_p = df_valid[df_valid['Group'] == 'Phased_Adaptive'][col]

        if len(group_g) < 3 or len(group_p) < 3:
            report_text += f"{title} ({col}) 数据不足，跳过统计。\n\n"
            continue

        # 正态性检验与显著性检验选取
        _, p_norm_g = stats.shapiro(group_g)
        _, p_norm_p = stats.shapiro(group_p)
        
        if p_norm_g > 0.05 and p_norm_p > 0.05:
            _, p_lev = stats.levene(group_g, group_p)
            t_stat, p_val = stats.ttest_ind(group_g, group_p, equal_var=(p_lev > 0.05))
            test_method = "T-Test (Parametric)"
        else:
            _, p_val = stats.mannwhitneyu(group_g, group_p, alternative='two-sided')
            test_method = "Mann-Whitney U (Non-parametric)"

        d = get_cohens_d(group_p, group_g) # Phased 相较于 Global 的效应量
        
        mean_g, std_g = np.mean(group_g), np.std(group_g)
        mean_p, std_p = np.mean(group_p), np.std(group_p)

        # 报告生成
        sig_mark = "【显著差异】" if p_val < 0.05 else "【无显著差异】"
        report_text += f"指标: {title} ({col})\n"
        report_text += f"- 全局自适应 (Mean±SD): {mean_g:.4f} ± {std_g:.4f}\n"
        report_text += f"- 分阶段自适应 (Mean±SD): {mean_p:.4f} ± {std_p:.4f}\n"
        report_text += f"- 检验方法: {test_method} | p-value: {p_val:.4e}\n"
        report_text += f"- 效应量 (Cohen's d): {d:.4f}\n"
        report_text += f"-> 结论: {sig_mark} (p={p_val:.3f})\n"
        report_text += "-" * 50 + "\n"

        # 图表绘制
        plt.subplot(2, 3, i)
        ax = sns.boxplot(x="Group", y=col, data=df_valid, palette="Set2", width=0.5, order=["Global_Adaptive", "Phased_Adaptive"])
        sns.stripplot(x="Group", y=col, data=df_valid, color=".3", alpha=0.4, order=["Global_Adaptive", "Phased_Adaptive"])
        
        plt.title(f"{title}\np={p_val:.3f}, d={d:.2f}", fontsize=11)
        plt.ylabel(y_label)
        plt.xlabel("")
        ax.set_xticklabels(["Global\nAdaptive", "Phased\nAdaptive"])

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "fine_phase_metrics_comparison.png"), dpi=300)
    plt.close()

    # 导出结果
    report_path = os.path.join(OUTPUT_DIR, "statistical_report.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report_text)
    
    df.to_csv(os.path.join(OUTPUT_DIR, "fine_phase_metrics_raw.csv"), index=False)

    print(report_text)
    print(f"\n>>> 统计完成！结果与图表已保存至: {OUTPUT_DIR}")

if __name__ == "__main__":
    df_metrics = extract_fine_phase_metrics()
    if not df_metrics.empty:
        run_statistical_analysis(df_metrics)
    else:
        print("未能成功提取数据，请检查数据集路径和列表配置。")