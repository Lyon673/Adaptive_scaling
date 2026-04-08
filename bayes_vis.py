import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

def visualize_bo_comprehensive(json_files, output_path="BO_Comprehensive_Results.png"):
    """
    绘制符合 IEEE RA-L 顶级审美的贝叶斯优化综合结果图。
    包含 4 个子图：
    (a) 平均收敛趋势与分布范围 (Shaded Range)
    (b) 总评分随迭代的散点折线图
    (c) 主观评分随迭代的散点折线图
    (d) 客观评分随迭代的散点折线图
    
    参数:
    json_files (list): JSON 文件名称列表。
    output_path (str): 输出图表的路径，保存为 .png 格式。
    """
    
    # ==========================================
    # 1. 严格的学术级样式设置
    # ==========================================
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "axes.labelsize": 12,
        "axes.titlesize": 13,
        "font.size": 11,
        "legend.fontsize": 10,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "axes.linewidth": 1.2
    })
    
    # 准备数据容器
    all_best_so_far = []
    subject_data = []
    
    # 获取受试者标记及对应颜色
    subject_names = [f.replace('scores.json', '').upper() for f in json_files]
    subject_colors = sns.color_palette("tab10", len(json_files))
    
    # ==========================================
    # 2. 数据读取与解析
    # ==========================================
    for file_name in json_files:
        # 注意: 依照您之前的代码逻辑，这里保留了 BayesianLog 路径。如需调整请自行修改。
        file_path = os.path.join(os.path.dirname(__file__), "BayesianLog", file_name)
        
        # 兼容性处理：如果 BayesianLog 文件夹不存在，尝试直接读取当前目录
        if not os.path.exists(file_path):
            file_path = file_name
            
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
        except Exception as e:
            print(f"无法读取文件 {file_path}: {e}")
            continue
            
        # 提取评分特征
        t_scores = []
        s_scores = []
        o_scores = []
        
        for item in data:
            t_score = item.get('total_score', 0)
            s_score = item.get('subscore', 0)
            # 客观评分 = 各客观指标得分之和
            o_score = (item.get('gracefulness_score', 0) + 
                       item.get('smoothness_score', 0) + 
                       item.get('clutch_times_score', 0) + 
                       item.get('total_distance_score', 0) + 
                       item.get('total_time_score', 0))
            
            t_scores.append(t_score)
            s_scores.append(s_score)
            o_scores.append(o_score)
            
        subject_data.append({
            'total': t_scores,
            'subj': s_scores,
            'obj': o_scores
        })
        
        # 计算“历史最优分数”
        best_so_far = np.maximum.accumulate(t_scores)
        all_best_so_far.append(best_so_far)
        
    if not all_best_so_far:
        print("没有成功加载任何有效数据！")
        return

    # ==========================================
    # 3. 收敛范围数据对齐 (处理不同受试者迭代次数不一致的问题)
    # ==========================================
    max_iters = max(len(arr) for arr in all_best_so_far)
    padded_best_so_far = []
    for arr in all_best_so_far:
        padded = np.pad(arr, (0, max_iters - len(arr)), mode='edge')
        padded_best_so_far.append(padded)
        
    padded_best_so_far = np.array(padded_best_so_far)
    
    iterations_conv = np.arange(1, max_iters + 1)
    mean_scores = np.mean(padded_best_so_far, axis=0)
    min_scores = np.min(padded_best_so_far, axis=0)
    max_scores = np.max(padded_best_so_far, axis=0)

    # ==========================================
    # 4. 2x2 组合网格绘图
    # ==========================================
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    ax_conv = axes[0, 0]
    ax_tot  = axes[0, 1]
    ax_sub  = axes[1, 0]
    ax_obj  = axes[1, 1]
    
    # 统一函数：用于装饰子图
    def decorate_ax(ax, title, xlabel, ylabel):
        ax.set_title(title, fontweight='bold', pad=10)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.grid(axis='y', linestyle='--', linewidth=0.8, alpha=0.4, color='gray')
        
    # ---------------------------------------------------------
    # 图 (a): BO 收敛趋势与极值阴影范围
    # ---------------------------------------------------------
    line_color = "#005b96"
    fill_color = "#b3cde0"
    ax_conv.fill_between(iterations_conv, min_scores, max_scores, color=fill_color, alpha=0.5, edgecolor='none', label='Performance Range (Min-Max)')
    ax_conv.plot(iterations_conv, mean_scores, color=line_color, linewidth=2.5, label='Mean Best-so-far Score')
    
    decorate_ax(ax_conv, '(a) Bayesian Optimization Convergence', 'Iteration / Trial Number', 'Best-so-far Total Score')
    ax_conv.set_xlim(1, max_iters)
    ax_conv.legend(loc='lower right', frameon=False)
    
    # ---------------------------------------------------------
    # 图 (b), (c), (d): 散点折线图展示各评分分量
    # ---------------------------------------------------------
    for i, data_dict in enumerate(subject_data):
        iters = np.arange(1, len(data_dict['total']) + 1)
        sub_name = subject_names[i]
        color = subject_colors[i]
        
        # 参数设定: marker size, line width, alpha
        plot_kwargs = {'marker': 'o', 'markersize': 5, 'linewidth': 1.5, 'alpha': 0.8, 'color': color, 'label': f'Subject {sub_name}'}
        
        ax_tot.plot(iters, data_dict['total'], **plot_kwargs)
        ax_sub.plot(iters, data_dict['subj'], **plot_kwargs)
        ax_obj.plot(iters, data_dict['obj'], **plot_kwargs)
        
    decorate_ax(ax_tot, '(b) Total Score Across Iterations', 'Iteration / Trial Number', 'Evaluated Total Score')
    decorate_ax(ax_sub, '(c) Subjective Score Across Iterations', 'Iteration / Trial Number', 'Evaluated Subjective Score')
    decorate_ax(ax_obj, '(d) Objective Score Across Iterations', 'Iteration / Trial Number', 'Evaluated Objective Score')
    
    # 为散点折线图统一添加图例 (放在右上角的图 b 中即可)
    ax_tot.legend(title='Participants', bbox_to_anchor=(1.02, 1), loc='upper left', frameon=False)

    # 动态调整各子图的 X 轴范围
    for ax in [ax_tot, ax_sub, ax_obj]:
        ax.set_xlim(0.5, max_iters + 0.5)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"学术图表已成功生成并保存至: {output_path}")

if __name__ == "__main__":
    # 评分 JSON 文件列表
    json_files_list = [
        "zmsscores.json",
        "zymscores.json",
        "ztsscores.json",
        "ljsscores.json",
        "lyfscores.json"
    ]
    
    visualize_bo_comprehensive(json_files_list, output_path="BO_Comprehensive_Results.png")