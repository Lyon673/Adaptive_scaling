import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.signal import medfilt

# # 将当前目录加入路径以导入 ipa.py
# _SCRIPT_DIR_EARLY = os.path.dirname(os.path.abspath(__file__))
# if _SCRIPT_DIR_EARLY not in sys.path:
#     sys.path.insert(0, _SCRIPT_DIR_EARLY)

try:
    from IPA_code.ipa import Pupil, ipa_cal
except ImportError:
    print("警告: 无法导入 ipa.py，请确保它与此脚本在同一目录下。")

# ==========================================
# 1. 实验组划分配置
# ==========================================
# GLOBAL_ADAPTIVE_DEMOS = list(range(0, 5)) + list(range(10, 15)) + list(range(21, 26)) + list(range(31, 36))
# PHASED_ADAPTIVE_DEMOS = list(range(5, 10)) + list(range(15, 21)) + list(range(26, 31)) + list(range(36, 41))
# GLOBAL_ADAPTIVE_DEMOS = list(range(0,5))+[10,12,13,14]+[22,23,25]+[31,32,33,35]+[41,42,43]+[47,48,49]+[53,54,55]+[59,60,61]
# PHASED_ADAPTIVE_DEMOS = list(range(5,10))+[15,20] +[26,27,29]+[36,37,38,40]+[44,45,46]+[50,51,52]+[56,57,58]+[62,63,64]
GLOBAL_ADAPTIVE_DEMOS = list(range(0,5))+[10,12,13,14]+[22,23,25]+[31,32,33,35]+[41,42,43]+[47,48,49]+[53,54,55]+[59,60,61]+[65,66,67,68,69,70]
PHASED_ADAPTIVE_DEMOS = list(range(5,10))+[15,20] +[26,27,29]+[36,37,38,40]+[44,45,46]+[50,51,52]+[56,57,58]+[62,63,64]+[71,72,73,74,75,76]

# ==========================================
# 参数与路径配置
# ==========================================
VIDEO_FPS  = 30
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_ROOT  = os.path.join(SCRIPT_DIR, "data_Phase")
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "phase_metrics_comparison_report")

if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

# ==========================================
# 2. 眼动指标计算函数
# ==========================================

def mad_outlier_detection(series, constant=4):
    median = np.nanmedian(series)
    mad = np.nanmedian(np.abs(series - median))
    if mad == 0:
        return np.zeros_like(series, dtype=bool)
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
    if len(valid_points) < 10:
        return np.nan

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
    if len(valid) < 20:
        return np.nan

    xmin, xmax = valid[:, 0].min(), valid[:, 0].max()
    ymin, ymax = valid[:, 1].min(), valid[:, 1].max()
    if xmax == xmin or ymax == ymin:
        return 0.0

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


def get_cohens_d(g1, g2):
    n1, n2 = len(g1), len(g2)
    v1, v2 = np.var(g1, ddof=1), np.var(g2, ddof=1)
    pooled_std = np.sqrt(((n1 - 1) * v1 + (n2 - 1) * v2) / (n1 + n2 - 2))
    return (np.mean(g1) - np.mean(g2)) / pooled_std if pooled_std != 0 else 0

# ==========================================
# 3. 数据加载与指标提取（整体序列）
# ==========================================

def extract_eye_metrics():
    results = []

    if not os.path.exists(DATA_ROOT):
        print(f"数据根目录不存在: {DATA_ROOT}")
        return pd.DataFrame(results)

    demo_folders = sorted(
        [d for d in os.listdir(DATA_ROOT)
         if os.path.isdir(os.path.join(DATA_ROOT, d)) and d.split('_')[0].isdigit()],
        key=lambda x: int(x.split('_')[0])
    )
    print(f"开始提取眼动指标，共发现 {len(demo_folders)} 个数据目录...\n")

    for folder_name in demo_folders:
        demo_idx = int(folder_name.split('_')[0])
        data_folder = os.path.join(DATA_ROOT, folder_name)

        try:
            if demo_idx in GLOBAL_ADAPTIVE_DEMOS:
                group = "Global_Adaptive"
            elif demo_idx in PHASED_ADAPTIVE_DEMOS:
                group = "Phased_Adaptive"
            else:
                continue

            gaze_pos_path  = os.path.join(data_folder, "gazepoint_position_data.npy")
            gaze_data_path = os.path.join(data_folder, "gaze_data.npy")

            if not os.path.exists(gaze_pos_path) or not os.path.exists(gaze_data_path):
                print(f"  [跳过] demo {demo_idx}: 找不到眼动数据文件")
                continue

            gaze_pos  = np.load(gaze_pos_path)
            gaze_data = np.load(gaze_data_path)

            if gaze_pos.ndim == 3:
                gaze_pos = gaze_pos.reshape(gaze_pos.shape[0], -1)
            if gaze_pos.ndim == 1 and len(gaze_pos) % 2 == 0:
                gaze_pos = gaze_pos.reshape(-1, 2)

            pL = preprocess_pupil_signal(gaze_data[:, 6])
            pR = preprocess_pupil_signal(gaze_data[:, 7])
            pupil_avg = (pL + pR) / 2.0

            pm = np.nanmean(pupil_avg)
            pupil_cv = np.nanstd(pupil_avg) / pm if pm != 0 else np.nan

            stat_entropy  = calculate_entropy(gaze_pos)
            trans_entropy = calculate_gte(gaze_pos)

            # ==============================================================
            # 新增：利用整段样本眼动序列及真实时间戳计算 IPA
            # ==============================================================
            ipa_mean = np.nan
            
            # 第 9 列是眼动仪微秒级时间戳，转为秒并使用相对时间
            eye_times = gaze_data[:, 8] * 1e-6
            eye_times = eye_times - eye_times[0] 
            
            # 过滤掉预处理未能填充的少量 NaN 边界值
            valid_mask = ~np.isnan(pL) & ~np.isnan(pR)
            
            # 小波变换对长度有一定要求，避免极短残缺序列报错
            if np.sum(valid_mask) > 64:
                valid_pL = pL[valid_mask]
                valid_pR = pR[valid_mask]
                valid_times = eye_times[valid_mask]
                
                # 封装为 ipa.py 所需的 Pupil 对象列表
                pupils_L = [Pupil(val, t) for val, t in zip(valid_pL, valid_times)]
                pupils_R = [Pupil(val, t) for val, t in zip(valid_pR, valid_times)]
                
                # timestamp=None 代表计算整段序列的全时 IPA
                res_L = ipa_cal(pupils_L, timestamp=None)
                res_R = ipa_cal(pupils_R, timestamp=None)
                
                # ipa_cal 返回 (IPA, position, threshold)
                ipa_L = res_L[0] if res_L is not None else np.nan
                ipa_R = res_R[0] if res_R is not None else np.nan
                
                ipa_mean = float(np.nanmean([ipa_L, ipa_R]))
            # ==============================================================

            results.append({
                "Demo_ID":              demo_idx,
                "Group":                group,
                "Pupil_Variation_Rate": pupil_cv,
                "Static_Entropy":       stat_entropy,
                "Transition_Entropy":   trans_entropy,
                "IPA_Mean":             ipa_mean,
            })
            
            # 由于 ipa_cal 会打印 threshold，这里调整打印使其更清晰
            print(f"  -> demo {demo_idx} ({group}) | PupilCV={pupil_cv:.4f}, "
                  f"StatEnt={stat_entropy:.4f}, TransEnt={trans_entropy:.4f}, "
                  f"IPA_Mean={ipa_mean:.4f}\n")

        except Exception as e:
            print(f"处理 demo {demo_idx} ({folder_name}) 时出错: {e}")

    df = pd.DataFrame(results)
    if not df.empty:
        print(f"\n提取完成，共 {len(df)} 条记录：")
        print(df.groupby('Group').size())
    return df

# ==========================================
# 4. 统计检验与可视化
# ==========================================

def run_statistical_analysis(df):
    metrics = {
        "Pupil_Variation_Rate": ("Pupil Coefficient of Variation", "Pupil Variation Rate"),
        "Static_Entropy":       ("Stationary Gaze Entropy (bits)", "Static Entropy"),
        "Transition_Entropy":   ("Gaze Transition Entropy (bits)", "Transition Entropy"),
        "IPA_Mean":             ("Mean IPA (Bimanual Interaction)", "IPA Mean"),
    }

    report_text  = "Eye Metrics Comparison (Global Adaptive vs Phased Adaptive)\n"
    report_text += "=" * 65 + "\n\n"

    n_metrics = len(metrics)
    fig, axes = plt.subplots(1, n_metrics, figsize=(5 * n_metrics, 6), squeeze=False)
    axes = axes[0]
    sns.set_theme(style="whitegrid")

    for ax, (col, (y_label, title)) in zip(axes, metrics.items()):
        df_valid  = df.dropna(subset=[col])
        group_g   = df_valid[df_valid['Group'] == 'Global_Adaptive'][col].values
        group_p   = df_valid[df_valid['Group'] == 'Phased_Adaptive'][col].values

        if len(group_g) < 3 or len(group_p) < 3:
            report_text += f"{title} ({col}) 数据不足，跳过统计。\n\n"
            continue

        _, p_norm_g = stats.shapiro(group_g)
        _, p_norm_p = stats.shapiro(group_p)

        if p_norm_g > 0.05 and p_norm_p > 0.05:
            _, p_lev = stats.levene(group_g, group_p)
            _, p_val = stats.ttest_ind(group_g, group_p, equal_var=(p_lev > 0.05))
            test_method = "T-Test (Parametric)"
        else:
            _, p_val = stats.mannwhitneyu(group_g, group_p, alternative='two-sided')
            test_method = "Mann-Whitney U (Non-parametric)"

        d = get_cohens_d(group_p, group_g)

        mean_g, std_g = np.mean(group_g), np.std(group_g)
        mean_p, std_p = np.mean(group_p), np.std(group_p)

        sig_mark = "【显著差异】" if p_val < 0.05 else "【无显著差异】"
        report_text += f"指标: {title} ({col})\n"
        report_text += f"- 全局自适应  (Mean±SD): {mean_g:.4f} ± {std_g:.4f}  (N={len(group_g)})\n"
        report_text += f"- 分阶段自适应 (Mean±SD): {mean_p:.4f} ± {std_p:.4f}  (N={len(group_p)})\n"
        report_text += f"- 检验方法: {test_method} | p-value: {p_val:.4e}\n"
        report_text += f"- 效应量 (Cohen's d): {d:.4f}\n"
        report_text += f"-> 结论: {sig_mark} (p={p_val:.3f})\n"
        report_text += "-" * 50 + "\n"

        sns.boxplot(x="Group", y=col, data=df_valid, palette="Set2", width=0.5,
                    order=["Global_Adaptive", "Phased_Adaptive"], ax=ax)
        sns.stripplot(x="Group", y=col, data=df_valid, color=".3", alpha=0.5,
                      order=["Global_Adaptive", "Phased_Adaptive"], ax=ax)

        ax.set_title(f"{title}\np={p_val:.3f}, d={d:.2f}", fontsize=12)
        ax.set_ylabel(y_label)
        ax.set_xlabel("")
        ax.set_xticklabels(["Global\nAdaptive", "Phased\nAdaptive"])

    plt.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, "eye_metrics_comparison.png"), dpi=300)
    plt.close(fig)

    # report_path = os.path.join(OUTPUT_DIR, "statistical_report.txt")
    # with open(report_path, "w", encoding="utf-8") as f:
    #     f.write(report_text)

    # df.to_csv(os.path.join(OUTPUT_DIR, "eye_metrics_raw.csv"), index=False)

    print(report_text)
    print(f"\n>>> 统计完成！结果与图表已保存至: {OUTPUT_DIR}")


if __name__ == "__main__":
    df_metrics = extract_eye_metrics()
    if not df_metrics.empty:
        run_statistical_analysis(df_metrics)
    else:
        print("未能成功提取数据，请检查数据集路径和列表配置。")