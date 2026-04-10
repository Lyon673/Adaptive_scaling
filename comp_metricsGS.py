import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.spatial.distance import pdist, squareform
import matplotlib.lines as mlines

_SCRIPT_DIR_EARLY = os.path.dirname(os.path.abspath(__file__))
if _SCRIPT_DIR_EARLY not in sys.path:
    sys.path.insert(0, _SCRIPT_DIR_EARLY)
from gracefulness import calculate_smoothness  # 仅保留 smoothness

# ==========================================
# 1. 实验组划分与被试者配置
# ==========================================
# GLOBAL_ADAPTIVE_DEMOS = list(range(0,5))+[10,12,13,14]+[22,23,25]+[31,32,33,35]
# PHASED_ADAPTIVE_DEMOS = list(range(5,10))+[15,20] +[26,27,29]+[36,37,38,40]
GLOBAL_ADAPTIVE_DEMOS = list(range(0,5))+[10,12,13,14]+[22,23,25]+[31,32,33,35]+[41,42,43]+[47,48,49]+[53,54,55]+[59,60,61]
PHASED_ADAPTIVE_DEMOS = list(range(5,10))+[15,20] +[26,27,29]+[36,37,38,40]+[44,45,46]+[50,51,52]+[56,57,58]+[62,63,64]

PARTICIPANT_RANGES = {
    "P1": range(0, 10),
    "P2": range(10, 21),
    "P3": range(21, 31),
    "P4": range(31, 41),
    "P5": range(41, 47),
    "P6": range(47, 53),
    "P7": range(53, 59),
    "P8": range(59, 65)
}

BASE_PALETTES = {
    "P1": "Blues",
    "P2": "Greens",
    "P3": "Reds",
    "P4": "Purples",
    "P5": "Oranges",
    "P6": "Greys",
    "P7": "Blues",
    "P8": "Greens"
}

# ==========================================
# 参数与路径配置
# ==========================================
DATA_ROOT  = os.path.join(_SCRIPT_DIR_EARLY, "data_Phase")
OUTPUT_DIR = os.path.join(_SCRIPT_DIR_EARLY, "kinematic_spatial_metrics_report")
FINE_PHASES = [1, 3, 5]

if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

# ==========================================
# 2. 新增空间离散程度计算函数
# ==========================================

def calc_ellipsoid_volume(positions):
    """
    计算三维坐标的 95% 置信协方差椭球体积。
    基于 3 自由度卡方分布，95% 置信度的 chi2 阈值约为 7.815。
    V = (4/3) * pi * (chi2_val)^1.5 * sqrt(det(Cov))
    越小说明位置越集中。
    """
    if positions is None or len(positions) < 4:
        return np.nan
    # 计算 3x3 协方差矩阵
    cov_matrix = np.cov(positions, rowvar=False)
    det = np.linalg.det(cov_matrix)
    if det <= 0:
        return 0.0
    chi2_95_3d = 7.8147
    volume = (4.0 / 3.0) * np.pi * (chi2_95_3d ** 1.5) * np.sqrt(det)
    return volume

def calc_sample_entropy(positions, m=2, r_coeff=0.2):
    """
    计算运动学样本熵 (Kinematic Sample Entropy)。
    将三维坐标转化为偏离中心点的距离标量序列 d(t)，衡量操作空间轨迹的复杂度和无序性。
    越小说明运动越有规律（抖动少）。
    """
    if positions is None or len(positions) < m + 2:
        return np.nan
    
    # 提取相对于质心的距离序列
    centroid = np.mean(positions, axis=0)
    U = np.linalg.norm(positions - centroid, axis=1)
    
    N = len(U)
    r = r_coeff * np.std(U)
    
    if r == 0: return 0.0 # 完全静止
    
    def _phi(m_len):
        try:
            x = np.array([U[i:i+m_len] for i in range(N - m_len + 1)])
            # 使用 Chebyshev 距离计算无穷范数下的最大距离
            dists = pdist(x, metric='chebyshev')
            C = np.sum(dists <= r) * 2  # 上三角重构全矩阵对应数量
            return C
        except MemoryError:
            return np.nan
            
    N1 = _phi(m)
    N2 = _phi(m + 1)
    
    if np.isnan(N1) or np.isnan(N2): return np.nan
    if N1 == 0 or N2 == 0: return np.nan
    
    return -np.log(N2 / N1)

def calculate_sparc_from_data(data, padlevel=4, fc=10.0, amp_th=0.05):
    """ 计算速度谱弧长 (SPARC) """
    if data is None: return np.nan
    positions = data[:, :3]
    time = data[:, 3]
    
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
    arc_length = -np.sum(np.sqrt(df**2 + dM**2))
    return arc_length

def _build_pos_time(pos_npy):
    pos = np.asarray(pos_npy, dtype=float)
    if pos.ndim == 1 or pos.shape[1] < 4: return None
    xyz, t_col = pos[:, :3], pos[:, 3]
    t_col = np.maximum.accumulate(t_col)
    t = t_col - t_col[0]
    if t[-1] == 0: return None
    return np.column_stack([xyz, t])

def _calc_segment_arrays(pos_npy_seg):
    if pos_npy_seg is None or len(pos_npy_seg) < 5:
        return np.nan, np.nan, np.nan, np.nan
        
    data = _build_pos_time(pos_npy_seg)
    if data is None or len(data) < 5:
        return np.nan, np.nan, np.nan, np.nan
        
    positions, time = data[:, :3], data[:, 3]
    
    # 1. 计算标量平滑度 S (文献公式)
    velocities = np.gradient(positions, time, axis=0)
    accelerations = np.gradient(velocities, time, axis=0)
    third_deriv = np.gradient(accelerations, time, axis=0)
    
    jerk_squared = np.sum(np.square(third_deriv), axis=1)
    integral = np.trapz(jerk_squared, x=time)
    duration = time[-1] - time[0]
    peak_velocity = np.max(np.linalg.norm(velocities, axis=1))
    
    if peak_velocity < 1e-10:
        S_scalar = np.log10(1e-10)
    else:
        phi = (np.power(duration, 5) / np.square(peak_velocity)) * integral
        S_scalar = np.log10(phi + 1e-10)
        
    # 2. 计算速度谱弧长 SPARC
    SPARC_scalar = calculate_sparc_from_data(data)
    
    # 3. 计算 95% 协方差椭球体积
    vol_scalar = calc_ellipsoid_volume(positions)
    
    # 4. 计算运动学样本熵
    sampen_scalar = calc_sample_entropy(positions)
        
    return S_scalar, SPARC_scalar, vol_scalar, sampen_scalar

def _find_segments(label_array, target_label):
    in_phase = (label_array == target_label)
    if not np.any(in_phase): return []
    changes = np.diff(in_phase.astype(np.int8))
    starts  = list(np.where(changes == 1)[0] + 1)
    ends    = list(np.where(changes == -1)[0] + 1)
    if in_phase[0]: starts = [0] + starts
    if in_phase[-1]: ends = ends + [len(label_array)]
    return list(zip(starts, ends))

def get_cohens_d(g1, g2):
    n1, n2 = len(g1), len(g2)
    v1, v2 = np.var(g1, ddof=1), np.var(g2, ddof=1)
    pooled_std = np.sqrt(((n1 - 1) * v1 + (n2 - 1) * v2) / (n1 + n2 - 2))
    return (np.mean(g1) - np.mean(g2)) / pooled_std if pooled_std != 0 else 0

# ==========================================
# 3. 核心指标提取流水线
# ==========================================
def extract_fine_phase_metrics():
    results = []
    if not os.path.exists(DATA_ROOT): return pd.DataFrame(results)

    demo_folders = sorted(
        [d for d in os.listdir(DATA_ROOT) if os.path.isdir(os.path.join(DATA_ROOT, d)) and d.split('_')[0].isdigit()],
        key=lambda x: int(x.split('_')[0])
    )
    print(f"开始提取混合粒度指标 (S/SPARC/Vol/SampEn为片段级, Path为序列级聚合)...\n{'='*85}")

    for folder_name in demo_folders:
        demo_idx = int(folder_name.split('_')[0])
        data_folder = os.path.join(DATA_ROOT, folder_name)

        if demo_idx in GLOBAL_ADAPTIVE_DEMOS: group = "Global_Adaptive"
        elif demo_idx in PHASED_ADAPTIVE_DEMOS: group = "Phased_Adaptive"
        else: continue

        try:
            label_path = os.path.join(data_folder, "phase_labels.npy")
            if not os.path.exists(label_path): continue
            phase_labels = np.load(label_path)

            def _load_pos(folder, arm):
                for fname in (f"{arm}psm_position.npy", f"{arm}psm_pose.npy"):
                    p = os.path.join(folder, fname)
                    if os.path.exists(p):
                        arr = np.load(p)
                        return arr[:, :4] if arr.ndim == 2 and arr.shape[1] >= 4 else None
                return None

            L_pos, R_pos = _load_pos(data_folder, "L"), _load_pos(data_folder, "R")
            if L_pos is None and R_pos is None: continue

            printed_demo_header = False
            
            demo_segments_cache = []
            demo_total_path_length = 0.0

            for label in FINE_PHASES:
                segments = _find_segments(phase_labels, label)
                for s_kine, e_kine in segments:
                    if e_kine - s_kine < 5: continue

                    if not printed_demo_header:
                        print(f"► 正在处理 Demo {demo_idx:>3d} ({group})")
                        printed_demo_header = True
                    
                    target_arm = "R" if label in [1, 3] else "L"
                    
                    if target_arm == "R":
                        seg_S, seg_SPARC, seg_vol, seg_sampen = _calc_segment_arrays(R_pos[s_kine:e_kine] if R_pos is not None else None)
                        seg_pos = R_pos[s_kine:e_kine, :3] if R_pos is not None else np.zeros((0,3))
                    else:
                        seg_S, seg_SPARC, seg_vol, seg_sampen = _calc_segment_arrays(L_pos[s_kine:e_kine] if L_pos is not None else None)
                        seg_pos = L_pos[s_kine:e_kine, :3] if L_pos is not None else np.zeros((0,3))
                        
                    seg_path_len = np.sum(np.linalg.norm(np.diff(seg_pos, axis=0), axis=1)) if len(seg_pos) > 1 else np.nan

                    if not np.isnan(seg_path_len):
                        demo_total_path_length += seg_path_len

                    # 为了打印整齐，将体积乘 1e6 展示 (mm^3)
                    print(f"    ├─ Phase {label} (臂:{target_arm}): [{s_kine:>4d}->{e_kine:>4d}] | S={seg_S:5.2f}, SPARC={seg_SPARC:5.2f}, Vol={seg_vol*1e6:6.2f}mm³, SampEn={seg_sampen:5.2f}, Path={seg_path_len:5.3f}m")

                    if not np.isnan(seg_S):
                        demo_segments_cache.append({
                            "Demo_ID": demo_idx,
                            "Group": group,
                            "Phase_Label": label,
                            "Target_Arm": target_arm,
                            "Smoothness": seg_S,
                            "SPARC": seg_SPARC,
                            "Ellipsoid_Volume_95": seg_vol,
                            "Kinematic_SampEn": seg_sampen
                        })

            for seg in demo_segments_cache:
                seg["Total_Path_Length"] = demo_total_path_length
                results.append(seg)

            if printed_demo_header:
                print(f"    └─ 该序列精细操作目标从手总路径: {demo_total_path_length:.3f}m")
                print("-" * 85)
                
        except Exception as e:
            print(f"处理 demo {demo_idx} 时出错: {e}")

    return pd.DataFrame(results)

# ==========================================
# 4. 色彩映射与统计绘图
# ==========================================
def generate_color_mapping():
    color_dict = {}
    for p_name, p_range in PARTICIPANT_RANGES.items():
        p_global = sorted([d for d in GLOBAL_ADAPTIVE_DEMOS if d in p_range])
        p_phased = sorted([d for d in PHASED_ADAPTIVE_DEMOS if d in p_range])
        
        max_shades = max(len(p_global), len(p_phased))
        shades = sns.color_palette(BASE_PALETTES[p_name], n_colors=max_shades + 3)[3:] 
        
        for i, d in enumerate(p_global):
            color_dict[d] = shades[i]
        for i, d in enumerate(p_phased):
            color_dict[d] = shades[i]
            
    return color_dict

def run_statistical_analysis(df):
    metrics = {
        "Smoothness":          ("Smoothness S (Lower is smoother)", "Smoothness (S)\n(Per Segment)"),
        "SPARC":               ("SPARC (Higher is smoother)", "Spectral Arc Length (SPARC)\n(Per Segment)"),
        "Ellipsoid_Volume_95": ("Ellipsoid Volume (m³) (Lower is denser)", "95% Covariance Ellipsoid Vol\n(Per Segment)"),
        "Kinematic_SampEn":    ("Sample Entropy (Lower is more regular)", "Kinematic Sample Entropy\n(Per Segment)"),
        "Total_Path_Length":   ("Path Length (m) (Lower is better)", "Total Path Length\n(Per Sequence)")
    }
    
    color_dict = generate_color_mapping()

    report_text = "Mixed-Level Kinematic & Spatial Metrics Comparison\n"
    report_text += "=" * 65 + "\n\n"

    fig, axes = plt.subplots(1, 5, figsize=(30, 6))
    sns.set_theme(style="whitegrid")

    for i, (col, (y_label, title)) in enumerate(metrics.items()):
        ax = axes[i]
        
        # 确定统计粒度与基础数据
        if col == "Total_Path_Length":
            df_valid = df.drop_duplicates(subset=["Demo_ID"]).dropna(subset=[col])
            unit = "序列(Demo)"
        else:
            df_valid = df.dropna(subset=[col])
            unit = "片段(Segment)"
            
        # ---------------------------------------------------------
        # 【新增逻辑】：针对 95% 协方差椭球体积进行基于 IQR 的组内离群点剔除
        # ---------------------------------------------------------
        outlier_msg = ""
        if col == "Ellipsoid_Volume_95":
            df_filtered_list = []
            outlier_info = []
            for grp in ['Global_Adaptive', 'Phased_Adaptive']:
                df_grp = df_valid[df_valid['Group'] == grp]
                
                # 计算四分位数与 IQR
                Q1 = df_grp[col].quantile(0.25)
                Q3 = df_grp[col].quantile(0.75)
                IQR = Q3 - Q1
                
                # 定义 1.5*IQR 边界
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                
                # 过滤并记录
                df_filtered = df_grp[(df_grp[col] >= lower_bound) & (df_grp[col] <= upper_bound)]
                df_filtered_list.append(df_filtered)
                
                removed_count = len(df_grp) - len(df_filtered)
                outlier_info.append(f"{grp}: {removed_count}个")
                
            # 重组剔除离群点后的 DataFrame
            df_valid = pd.concat(df_filtered_list)
            outlier_msg = f"- 离群点剔除 (1.5*IQR): {', '.join(outlier_info)}\n"
        # ---------------------------------------------------------
            
        group_g = df_valid[df_valid['Group'] == 'Global_Adaptive'][col]
        group_p = df_valid[df_valid['Group'] == 'Phased_Adaptive'][col]

        if len(group_g) < 3 or len(group_p) < 3: continue

        # 统计检验
        _, p_norm_g = stats.shapiro(group_g)
        _, p_norm_p = stats.shapiro(group_p)
        
        if p_norm_g > 0.05 and p_norm_p > 0.05:
            _, p_lev = stats.levene(group_g, group_p)
            t_stat, p_val = stats.ttest_ind(group_g, group_p, equal_var=(p_lev > 0.05))
            test_method = "T-Test (Parametric)"
        else:
            _, p_val = stats.mannwhitneyu(group_g, group_p, alternative='two-sided')
            test_method = "Mann-Whitney U"

        d = get_cohens_d(group_p, group_g) 
        mean_g, std_g = np.mean(group_g), np.std(group_g)
        mean_p, std_p = np.mean(group_p), np.std(group_p)

        report_text += f"指标: {title.replace(chr(10), ' ')}\n"
        report_text += f"- 统计单位: {unit} (N_global={len(group_g)}, N_phased={len(group_p)})\n"
        if outlier_msg: report_text += outlier_msg
        report_text += f"- Global (Mean±SD): {mean_g:.4e} ± {std_g:.4e}\n"
        report_text += f"- Phased (Mean±SD): {mean_p:.4e} ± {std_p:.4e}\n"
        report_text += f"- Method: {test_method} | p-value: {p_val:.4e} | Cohen's d: {d:.4f}\n"
        report_text += "-" * 50 + "\n"

        sns.boxplot(
            x="Group", y=col, data=df_valid, 
            order=["Global_Adaptive", "Phased_Adaptive"],
            ax=ax, width=0.5, showfliers=False,
            boxprops={'facecolor': 'none', 'edgecolor': 'gray', 'zorder': 1},
            whiskerprops={'color': 'gray'},
            capprops={'color': 'gray'},
            medianprops={'color': 'darkred', 'linewidth': 2}
        )
        
        sns.stripplot(
            x="Group", y=col, hue="Demo_ID", data=df_valid,
            order=["Global_Adaptive", "Phased_Adaptive"],
            palette=color_dict, size=7, alpha=0.8, jitter=0.2, 
            dodge=False, ax=ax, zorder=2
        )
        
        if ax.get_legend() is not None:
            ax.get_legend().remove()
        
        ax.set_title(f"{title}\np-value={p_val:.4f}, d={d:.2f}", fontsize=14, fontweight='bold')
        ax.set_ylabel(y_label, fontsize=12)
        ax.set_xlabel("")
        ax.set_xticklabels(["Global Adaptive", "Phased Adaptive"], fontsize=12)
        
        if col in ["Ellipsoid_Volume_95"]:
            ax.ticklabel_format(axis='y', style='sci', scilimits=(0,0))

    # 被试者图例
    all_demos_in_data = set(df['Demo_ID'].unique())
    legend_elements = []
    
    for p_name, p_range in PARTICIPANT_RANGES.items():
        p_demos = sorted([d for d in list(p_range) if d in color_dict and d in all_demos_in_data])
        if not p_demos: continue

        legend_elements.append(
            mlines.Line2D([0], [0], linestyle='none', marker='', color='none', label=f'▌ {p_name}')
        )
        for d in p_demos:
            tag = 'G' if d in GLOBAL_ADAPTIVE_DEMOS else 'P'
            legend_elements.append(
                mlines.Line2D([0], [0], marker='o', linestyle='none',
                              color='w', markerfacecolor=color_dict[d],
                              markersize=9, label=f'  Demo {d:>3d}  [{tag}]')
            )

    fig.legend(
        handles=legend_elements,
        title='Demo Index\n[G=Global / P=Phased]',
        loc='center left',
        bbox_to_anchor=(1.01, 0.5),
        fontsize=10,
        title_fontsize=10,
        framealpha=0.92,
        edgecolor='#bbbbbb',
        handlelength=1.0,
    )

    plt.tight_layout(rect=[0, 0, 0.90, 1])
    plt.savefig(os.path.join(OUTPUT_DIR, "kinematic_spatial_metrics_comparison.png"), dpi=300, bbox_inches='tight')
    plt.close()

    with open(os.path.join(OUTPUT_DIR, "statistical_report.txt"), "w", encoding="utf-8") as f:
        f.write(report_text)
    
    df.to_csv(os.path.join(OUTPUT_DIR, "kinematic_spatial_metrics_raw.csv"), index=False)

    print(report_text)
    print(f"\n>>> 分析与绘图已完成！图表与数据保存至: {OUTPUT_DIR}")

if __name__ == "__main__":
    df_metrics = extract_fine_phase_metrics()
    if not df_metrics.empty:
        run_statistical_analysis(df_metrics)
    else:
        print("未提取到有效数据，请检查数据源及配置。")