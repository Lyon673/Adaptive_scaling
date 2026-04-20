import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
import os

def draw_smooth_parallel_line(ax, x_coords, y_coords, color, label, alpha=0.65):
    """
    绘制平滑的平行坐标曲线 (Sigmoid/Hermite 插值)。
    确保曲线在经过垂直坐标轴时保持水平，从而极大减少交叉时的视觉混乱。
    """
    # 绘制节点圆点
    ax.plot(x_coords, y_coords, 'o', color=color, markersize=6, alpha=0.9, zorder=5)
    
    # 逐段绘制平滑曲线
    for i in range(len(x_coords) - 1):
        x0, x1 = x_coords[i], x_coords[i+1]
        y0, y1 = y_coords[i], y_coords[i+1]
        
        # 生成 50 个插值点
        t = np.linspace(0, 1, 50)
        x_smooth = x0 + t * (x1 - x0)
        # 使用 Smoothstep 公式: 3t^2 - 2t^3，保证头尾导数为 0
        smooth_factor = 3 * t**2 - 2 * t**3
        y_smooth = y0 + smooth_factor * (y1 - y0)
        
        # 只在第一段添加 label，避免图例重复
        if i == 0:
            ax.plot(x_smooth, y_smooth, color=color, alpha=alpha, linewidth=2.0, label=label, zorder=4)
        else:
            ax.plot(x_smooth, y_smooth, color=color, alpha=alpha, linewidth=2.0, zorder=4)

def visualize_optimal_parameters(optimal_data, output_path="Optimal_Parameters_Distribution.png"):
    """
    绘制贝叶斯优化最优参数组合的差异可视化图（Z-score热力图 + 平滑的Z-score平行坐标图）。
    """
    # ==========================================
    # 1. 学术级样式设置
    # ==========================================
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "axes.labelsize": 12,
        "axes.titlesize": 14,
        "font.size": 11,
        "legend.fontsize": 10,
        "xtick.labelsize": 11,
        "ytick.labelsize": 11,
    })
    
    # 转换为 DataFrame
    df = pd.DataFrame.from_dict(optimal_data, orient='index')
    subjects = df.index.tolist()
    params = df.columns.tolist()
    
    
    # ==========================================
    # 2. 数据标准化 (Z-score Standardization)
    # ==========================================
    scaler = StandardScaler()
    df_normalized = pd.DataFrame(scaler.fit_transform(df), columns=params, index=subjects)
    
    
    # ==========================================
    # 3. 组合画布布局 (1行2列)
    # ==========================================
    fig = plt.figure(figsize=(16, 6.5))
    gs = fig.add_gridspec(1, 2, width_ratios=[1, 1.2], wspace=0.15)
    
    ax_heatmap = fig.add_subplot(gs[0, 0])
    ax_parallel = fig.add_subplot(gs[0, 1])

    # 统一提取 LaTeX 格式的标签
    latex_labels = [
        r'$C_{\mathrm{base}}$',
        r'$K_g$',
        r'$A_{\mathrm{gp}}$',
        r'$A_{\theta}$',
        r'$A_v$',
        r'$B_{\mathrm{safety}}$'
    ]
    
    # ---------------------------------------------------------
    # 图 (a): Z-score 标准化参数指纹热力图 (使用原始数值标注)
    # ---------------------------------------------------------
    # 【优化 1】：将 fmt 改为 ".2f"，限制小数点后两位，大幅减轻网格内的视觉拥挤感
    sns.heatmap(df_normalized, cmap="vlag", center=0, annot=df, fmt=".2f", 
                annot_kws={"size": 11, "weight": "bold"}, 
                linewidths=1.5, linecolor='white', 
                cbar_kws={'label': 'Z-score'}, 
                ax=ax_heatmap)
    
    ax_heatmap.set_title('(a) Optimal Parameter Matrix', fontweight='bold', pad=15)
    ax_heatmap.set_ylabel('Subjects')
    ax_heatmap.set_xlabel('Hyperparameters')
    
    # 将热力图的 X 轴标签替换为 LaTeX 格式
    ax_heatmap.set_xticklabels(latex_labels)
    ax_heatmap.tick_params(axis='x', rotation=25)
    
    # ---------------------------------------------------------
    # 图 (b): Z-score 平行坐标图 (Smooth Z-score Parallel Coordinates)
    # ---------------------------------------------------------
    df_parallel = df_normalized.reset_index().rename(columns={'index': 'Subject'})
    
    # 选用对比度更高的调色板
    colors = sns.color_palette("Set2", len(subjects))
    x_positions = np.arange(len(params))
    
    # 【优化 2】：使用平滑曲线代替锯齿直线绘制平行坐标
    for i, subject in enumerate(subjects):
        row = df_parallel[df_parallel['Subject'] == subject].iloc[0]
        y_values = [row[p] for p in params]
        draw_smooth_parallel_line(ax_parallel, x_positions, y_values, color=colors[i], label=f'Subject {subject}')
        
    # 装饰平行坐标图
    ax_parallel.set_title('(b) Smooth Parallel Coordinates of Z-score Hyperparameters', fontweight='bold', pad=15)
    ax_parallel.set_ylabel('Z-score')
    
    # 弱化水平网格线
    ax_parallel.grid(axis='y', linestyle=':', alpha=0.3)
    
    # 中心基准线
    ax_parallel.axhline(0, color='black', linestyle='--', linewidth=1.2, alpha=0.5, zorder=1)
    
    # 【优化 3】：强化垂直参数轴，构建视觉“支柱”
    for x_pos in x_positions:
        ax_parallel.axvline(x_pos, color='#95a5a6', linestyle='-', alpha=0.3, linewidth=1.0, zorder=0)
        
    # 设定 X 轴刻度和 LaTeX 标签
    ax_parallel.set_xticks(x_positions)
    ax_parallel.set_xticklabels(latex_labels)
    
    # 去除外边框
    ax_parallel.spines['top'].set_visible(False)
    ax_parallel.spines['right'].set_visible(False)
    ax_parallel.spines['bottom'].set_visible(False)
    ax_parallel.tick_params(axis='x', rotation=25, length=0) # 隐藏 x 轴的小短线刻度
    
    # 图例
    ax_parallel.legend(bbox_to_anchor=(1.02, 1), loc='upper left', frameon=False, title="Participants")
    
    # ==========================================
    # 4. 保存与输出
    # ==========================================
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"params fig saved to: {output_path}")

if __name__ == "__main__":
    # ==========================================
    # 真实最优参数数据
    # ==========================================
    optimal_params_mock = {
        1: {"C_base": 13.731003053878773, "K_g": 4.004938974633493, "A_gp": 2.252163132973055, "A_theta": 4.628285795238272, "A_v": 4.424680868208455, "B_safety": 3.828077489162437},
        2: {"C_base": 11.729176675963613, "K_g": 6.112117725918013,"A_gp": 4.0, "A_theta": 3.0, "A_v": 4.36028976481758, "B_safety": 4.0},
        3: {"C_base": 10.978811801615047, "K_g": 10.808192797812778,"A_gp": 2.2949116589467016, "A_theta": 4.553815277583111, "A_v": 4.1983396644297235, "B_safety": 3.923855971837864},
        4: {"C_base": 13.584593293918232, "K_g": 6.649830376727672,"A_gp": 2.3357273246141084, "A_theta": 4.678460849731229, "A_v": 4.411627055269589, "B_safety": 3.8912488509488146},
        5: {"C_base": 11.729176675963613, "K_g": 6.112117725918013,"A_gp": 3.4578946545631365, "A_theta": 4.329001414529318, "A_v": 4.202443320194236, "B_safety": 3.952259315416258},
        6: {"C_base": 10.740768116529518, "K_g": 8.0, "A_gp": 3.9591029250180663, "A_theta": 1.8608319789973347, "A_v": 3.7504741683031346, "B_safety": 4.0},
        7: {"C_base": 11.523280921852646, "K_g": 9.814210465008008,"A_gp": 2.9110343838669928, "A_theta": 6.525454707504819, "A_v": 5.031658854725127, "B_safety": 3.9581074272378065},
        8: {"C_base": 13.55295301146677, "K_g": 6.6705415539749,"A_gp": 3.791310167440371, "A_theta": 4.446112447908829, "A_v": 4.7505116240125576, "B_safety": 3.877888321780633},
    }
    output_path = os.path.join(os.path.dirname(__file__), "Essay_image_results", "Optimal_Parameters_Distribution.png")
    visualize_optimal_parameters(optimal_params_mock, output_path=output_path)