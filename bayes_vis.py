import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from scipy.interpolate import make_interp_spline

def visualize_bo_unified_smooth(json_files, output_path="BO_Unified_Results_Smooth.png"):
    """
    绘制符合 IEEE RA-L 顶级审美的一体化贝叶斯优化结果图。
    使用样条插值对均值与标准差带进行平滑处理，且严格经过原始节点。
    """
    
    # ==========================================
    # 1. 严格的学术级样式设置
    # ==========================================
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "axes.labelsize": 13,
        "axes.titlesize": 14,
        "font.size": 11,
        "legend.fontsize": 11,
        "xtick.labelsize": 11,
        "ytick.labelsize": 11,
        "axes.linewidth": 1.2
    })
    
    all_best_so_far = []
    subject_data = []
    
    # 严格读取您在原代码中设置的受试者标记
    subject_names = [1, 2, 3, 4, 5, 6, 7, 8]
    subject_colors = sns.color_palette("pastel", len(json_files))
    
    # ==========================================
    # 2. 数据读取与解析
    # ==========================================
    for file_name in json_files:
        file_path = os.path.join(os.path.dirname(__file__), "BayesianLog", file_name)
        if not os.path.exists(file_path):
            file_path = file_name
            
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
        except Exception as e:
            print(f"无法读取文件 {file_path}: {e}")
            continue
            
        t_scores = [item.get('total_score', 0) for item in data]
        subject_data.append(t_scores)
        
        best_so_far = np.maximum.accumulate(t_scores)
        all_best_so_far.append(best_so_far)
        
    if not all_best_so_far:
        print("没有成功加载任何有效数据！")
        return

    # ==========================================
    # 3. 收敛范围数据对齐与计算
    # ==========================================
    max_iters = max(len(arr) for arr in all_best_so_far)
    padded_best_so_far = []
    for arr in all_best_so_far:
        padded = np.pad(arr, (0, max_iters - len(arr)), mode='edge')
        padded_best_so_far.append(padded)
        
    padded_best_so_far = np.array(padded_best_so_far)
    iterations_raw = np.arange(1, max_iters + 1)
    
    mean_scores_raw = np.mean(padded_best_so_far, axis=0)
    std_scores_raw = np.std(padded_best_so_far, axis=0)
    
    lower_bound_raw = mean_scores_raw - std_scores_raw
    upper_bound_raw = mean_scores_raw + std_scores_raw

    # ==========================================
    # 4. 样条插值平滑处理 (Spline Interpolation)
    # ==========================================
    # 生成 300 个高密度采样点用于绘制平滑曲线
    iterations_smooth = np.linspace(iterations_raw.min(), iterations_raw.max(), 300)
    
    # k=3 表示使用 3 阶 B 样条，确保曲线二次可导且足够平滑
    # 如果 max_iters 太少（小于 4），可以降低 k 值
    k_order = 3 if max_iters > 3 else 1 
    
    spline_mean = make_interp_spline(iterations_raw, mean_scores_raw, k=k_order)
    spline_lower = make_interp_spline(iterations_raw, lower_bound_raw, k=k_order)
    spline_upper = make_interp_spline(iterations_raw, upper_bound_raw, k=k_order)
    
    mean_scores_smooth = spline_mean(iterations_smooth)
    lower_bound_smooth = spline_lower(iterations_smooth)
    upper_bound_smooth = spline_upper(iterations_smooth)

    # ==========================================
    # 5. 单图高密度融合绘图
    # ==========================================
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # --- 背景层：原始评估散点 ---
    for i, t_scores in enumerate(subject_data):
        iters = np.arange(1, len(t_scores) + 1)
        ax.plot(iters, t_scores, marker='o', markersize=4, linewidth=1.5, 
                alpha=0.4, color=subject_colors[i], label=f'Subject {subject_names[i]}')

    # --- 前景层：平滑的均值收敛曲线与标准差阴影 ---
    trend_color = "#1E8FC9" 
    
    # 绘制平滑的标准差阴影
    ax.fill_between(iterations_smooth, lower_bound_smooth, upper_bound_smooth, 
                    color=trend_color, alpha=0.2, edgecolor='none', 
                    label=r'Convergence Std Dev')
    
    # 绘制平滑的历史最优均值曲线
    ax.plot(iterations_smooth, mean_scores_smooth, color=trend_color, linewidth=3.0, 
            label='Mean Trend')
            
    # # 【学术规范】：在平滑曲线上点出真实的离散采样节点
    # ax.scatter(iterations_raw, mean_scores_raw, color=trend_color, s=30, zorder=5, 
    #            label='Actual Iteration Nodes')
    
    # --- 图表装饰 ---
    ax.set_title('Bayesian Optimization Iteration', fontweight='bold', pad=15)
    ax.set_xlabel('Iteration Number')
    ax.set_ylabel('Score')

    ax.set_xticks(np.arange(1, max_iters + 1))
    
    ax.set_xlim(0.8, max_iters + 0.2)
    
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(axis='y', linestyle='--', linewidth=0.8, alpha=0.4, color='gray')
    
    ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left', frameon=False, fontsize=10)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"一体化平滑优化图表已成功生成并保存至: {output_path}")

if __name__ == "__main__":
    json_files_list = [
        "zmsscores.json",
        "zymscores.json",
        "ztsscores.json",
        "ljsscores.json",
        "lyfscores.json",
        "scores0118ljs.json",
        "scores1201.json",
        "scores.json"
    ]
    
    visualize_bo_unified_smooth(json_files_list, output_path="BO_Unified_Results_Smooth.png")