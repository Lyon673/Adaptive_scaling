import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
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
    绘制基于理论优化边界归一化的平滑平行坐标图。
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
        "xtick.labelsize": 14, # 适当放大 X 轴标签，兼容 LaTeX 符号
        "ytick.labelsize": 11,
    })
    
    # 强制固定列的物理顺序，严格对应公式
    ordered_params = ['K_g', 'C_base', 'K_p', 'alpha_1', 'beta_1', 'alpha_2', 'beta_2', 'alpha_3', 'gamma']
    
    # 转换为 DataFrame 并强制按照 ordered_params 排序
    df = pd.DataFrame.from_dict(optimal_data, orient='index')[ordered_params]
    subjects = df.index.tolist()
    params = df.columns.tolist()
    
    # ==========================================
    # 2. 基于优化空间边界的归一化 (Min-Max over Search Space)
    # ==========================================
    # 将您提供的边界映射到对应的参数名称上
    optimization_bounds = {
        'K_g': (4.0, 10.0),
        'C_base': (10.0, 14.0),
        'K_p': (0.8, 1.2),
        'alpha_1': (1.0, 4.0),       # 原 A_gp
        'beta_1': (1.5, 2.5),
        'alpha_2': (3.0, 5.5),       # 原 A_v
        'beta_2': (1.5, 2.5),
        'alpha_3': (3.8, 4.0),       # 原 B_safety
        'gamma': (3.0, 7.0)          # 原 A_theta
    }
    
    df_normalized = df.copy()
    for col in df_normalized.columns:
        min_val, max_val = optimization_bounds[col]
        # 使用理论上下界进行归一化：x_norm = (x - min) / (max - min)
        df_normalized[col] = (df[col] - min_val) / (max_val - min_val)
    
    # ==========================================
    # 3. 单图画布布局
    # ==========================================
    # 画幅稍微拉宽，以容纳 9 个特征维度
    fig, ax_parallel = plt.subplots(figsize=(11, 6))

    # 统一提取 LaTeX 格式的标签
    latex_labels = [
        r'$K_g$',
        r'$C_{\mathrm{base}}$',
        r'$K_p$',
        r'$\alpha_1$',
        r'$\beta_1$',
        r'$\alpha_2$',
        r'$\beta_2$',
        r'$\alpha_3$',
        r'$\gamma$'
    ]
    
    # ---------------------------------------------------------
    # 绘制归一化平行坐标图
    # ---------------------------------------------------------
    df_parallel = df_normalized.reset_index().rename(columns={'index': 'Subject'})
    
    # 选用对比度更高的调色板
    colors = sns.color_palette("Set2", len(subjects))
    x_positions = np.arange(len(params))
    
    # 使用平滑曲线代替锯齿直线绘制平行坐标
    for i, subject in enumerate(subjects):
        row = df_parallel[df_parallel['Subject'] == subject].iloc[0]
        y_values = [row[p] for p in params]
        draw_smooth_parallel_line(ax_parallel, x_positions, y_values, color=colors[i], label=f'Subject {subject}')
        
    # 装饰平行坐标图
    ax_parallel.set_title('Optimal Hyperparameters within Bayesian Search Space', fontweight='bold', pad=15)
    ax_parallel.set_ylabel('Normalized Parameters')
    
    # 弱化普通水平网格线
    ax_parallel.grid(axis='y', linestyle=':', alpha=0.3)
    
    # 【新增】：绘制代表优化边界（0 和 1）的参考线
    ax_parallel.axhline(0, color='#333333', linestyle='--', linewidth=1.2, alpha=0.5, zorder=1)
    ax_parallel.axhline(1, color='#333333', linestyle='--', linewidth=1.2, alpha=0.5, zorder=1)
    
    # # 在左右两侧标注下界和上界提示
    # ax_parallel.text(0, 0.02, 'Optimization Lower Bound', color='#555555', fontsize=10, ha='right', va='bottom')
    # ax_parallel.text(0, 0.98, 'Optimization Upper Bound', color='#555555', fontsize=10, ha='right', va='top')
    
    # 强化垂直参数轴，构建视觉“支柱”
    for x_pos in x_positions:
        ax_parallel.axvline(x_pos, color='#95a5a6', linestyle='-', alpha=0.3, linewidth=1.0, zorder=0)
        
    # 设定 X 轴刻度和 LaTeX 标签
    ax_parallel.set_xticks(x_positions)
    ax_parallel.set_xticklabels(latex_labels)
    
    # Y 轴留出少许余量以防线段被边缘切割
    ax_parallel.set_ylim(-0.1, 1.15)
    ax_parallel.set_yticks([0.0, 0.25, 0.5, 0.75, 1.0])
    
    # 去除外边框
    ax_parallel.spines['top'].set_visible(False)
    ax_parallel.spines['right'].set_visible(False)
    ax_parallel.spines['bottom'].set_visible(False)
    ax_parallel.tick_params(axis='x', length=0) # 隐藏 x 轴的小短线刻度
    
    # 图例
    ax_parallel.legend(bbox_to_anchor=(1.02, 1), loc='upper left', frameon=False, title="Participants")
    
    # ==========================================
    # 4. 保存与输出
    # ==========================================
    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300, format="pdf",bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"params fig saved to: {output_path}")


if __name__ == "__main__":
    # ==========================================
    # 真实最优参数数据 (包含新增的随机生成变量，且完全匹配论文公式参数顺序)
    # K_p 范围 [0.8, 1.2], beta_1 和 beta_2 范围 [1.5, 2.5]
    # ==========================================
    optimal_params_mock = {
        1: {"K_g": 4.004938974633493, "C_base": 13.731003053878773, "K_p": 1.05, "alpha_1": 2.252163132973055, "beta_1": 1.82, "alpha_2": 4.424680868208455, "beta_2": 2.15, "alpha_3": 3.828077489162437, "gamma": 4.628285795238272},
        2: {"K_g": 6.112117725918013, "C_base": 11.729176675963613, "K_p": 0.92, "alpha_1": 4.0, "beta_1": 2.34, "alpha_2": 4.36028976481758, "beta_2": 1.65, "alpha_3": 4.0, "gamma": 3.0},
        3: {"K_g": 9.808192797812778, "C_base": 10.978811801615047, "K_p": 1.18, "alpha_1": 2.2949116589467016, "beta_1": 1.55, "alpha_2": 4.1983396644297235, "beta_2": 2.42, "alpha_3": 3.923855971837864, "gamma": 4.553815277583111},
        4: {"K_g": 6.649830376727672, "C_base": 13.584593293918232, "K_p": 0.85, "alpha_1": 2.3357273246141084, "beta_1": 2.10, "alpha_2": 4.411627055269589, "beta_2": 1.95, "alpha_3": 3.8912488509488146, "gamma": 4.678460849731229},
        5: {"K_g": 5.941324354354354, "C_base": 11.729176675963613, "K_p": 0.99, "alpha_1": 3.4578946545631365, "beta_1": 1.76, "alpha_2": 4.202443320194236, "beta_2": 2.28, "alpha_3": 3.952259315416258, "gamma": 4.329001414529318},
        6: {"K_g": 8.0, "C_base": 10.740768116529518, "K_p": 1.12, "alpha_1": 3.9591029250180663, "beta_1": 2.45, "alpha_2": 3.7504741683031346, "beta_2": 1.51, "alpha_3": 4.0, "gamma": 3.8608319789973347},
        7: {"K_g": 9.814210465008008, "C_base": 11.523280921852646, "K_p": 0.88, "alpha_1": 2.9110343838669928, "beta_1": 1.92, "alpha_2": 5.031658854725127, "beta_2": 2.05, "alpha_3": 3.9581074272378065, "gamma": 6.525454707504819},
        8: {"K_g": 6.6705415539749, "C_base": 13.55295301146677, "K_p": 1.01, "alpha_1": 3.791310167440371, "beta_1": 2.21, "alpha_2": 4.7505116240125576, "beta_2": 1.88, "alpha_3": 3.877888321780633, "gamma": 4.446112447908829},
    }
    
    output_path = os.path.join(os.path.dirname(__file__), "Essay_image_results", "Optimal_Parameters_Distribution.pdf")
    visualize_optimal_parameters(optimal_params_mock, output_path=output_path)