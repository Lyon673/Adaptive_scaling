"""
vis_participant_bars.py
=======================
基于被试者 (Participant) 维度的 2x2 统计检验柱状图可视化。
特性：
 1. X 轴为被试者，每个被试者内部紧凑展示三种模式的柱状图（带标准差误差棒）。
 2. 自动在柱状图上方执行组内 (同被试不同模式间) 的 T-Test 并精准标注显著性括号。
 3. [已同步] Y 轴刻度和留白范围与 analyze_results.py 的全局统计图完全一致。
"""

import json
import os
import argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import ttest_ind

# ── 学术级样式设置 ──────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "axes.labelsize": 14,
    "axes.titlesize": 16,
    "font.size": 12,
    "legend.fontsize": 12,
    "xtick.labelsize": 14,
    "ytick.labelsize": 12,
})

_PALETTE = ['#5681B9', '#5DA076', '#E9A263']

# ── 数据加载模块 ──────────────────────────────────────────────────────────────
def load_data(json_path: str) -> pd.DataFrame:
    with open(json_path) as f:
        raw = json.load(f)

    rows = []
    for entry in raw:
        folder = entry['objective_metrics'].get('data_folder', '0_unknown')
        demo_id = int(folder.split('_')[0])

        row = {'demo_id': demo_id}
        # 提取被试并大写
        row['participant'] = entry.get('participant', 'unknown').upper()
        
        # 规范化模式名称
        note = entry.get('note', 'unknown').lower()
        if 'fixed' in note: row['group'] = 'Fixed Mode'
        elif 'global' in note: row['group'] = 'Global Adaptive'
        elif 'phased' in note: row['group'] = 'Phased Adaptive'
        else: row['group'] = note
        
        row.update(entry['ratings'])
        row['subjective_score'] = entry['subjective_score']
        row.update(entry['objective_metrics'])
        
        clutch_l = entry['objective_metrics'].get('clutch_times_L', 0)
        clutch_r = entry['objective_metrics'].get('clutch_times_R', 0)
        row['clutch_times'] = clutch_l + clutch_r
        
        row['total_score'] = entry['total_score']
        rows.append(row)

    df = pd.DataFrame(rows).sort_values('demo_id').reset_index(drop=True)
    return df

# ── 柱状图悬浮统计检验标注引擎 ──────────────────────────────────────────────────
def add_grouped_stat_annotation(ax, df, x_col, y_col, hue_col, x_order, hue_order, y_lim_max, step):
    """
    针对 grouped barplot 的定制化标注算法：
    使用传入的全局 y_lim_max 自动计算安全的步长，防止越界。
    """
    n_hues = len(hue_order)
    width = 0.8 
    offsets = [(h - (n_hues - 1) / 2) * (width / n_hues) for h in range(n_hues)]
    
    # 根据全局上限和数据最大值的差值，计算安全的堆叠步长

    
    for i, p_name in enumerate(x_order):
        p_data = df[df[x_col] == p_name]
        if p_data.empty: continue

        # local_data_max = p_data[y_col].max()
        # local_avail = y_lim_max - local_data_max

        # step = local_avail / 3 if local_avail > 0 else local_data_max * 0.1
        print(f"p{i}: step={step}")
        
        # 寻找该被试当前数据的最高柱顶（均值 + 标准差）
        means = p_data.groupby(hue_col)[y_col].mean()
        stds = p_data.groupby(hue_col)[y_col].std().fillna(0)
        
        local_max = (means + stds).max()
        if np.isnan(local_max): local_max = p_data[y_col].max()
        
        current_y = local_max + step * 0.7
        
        pairs = [(0, 1), (1, 2), (0, 2)]
        for (h1_idx, h2_idx) in pairs:
            h1_name = hue_order[h1_idx]
            h2_name = hue_order[h2_idx]
            
            data1 = p_data[p_data[hue_col] == h1_name][y_col].dropna()
            data2 = p_data[p_data[hue_col] == h2_name][y_col].dropna()
            
            if len(data1) < 2 or len(data2) < 2: continue
            
            stat, p_val = ttest_ind(data1, data2)
            if np.isnan(p_val): continue
            
            # 只标注具有显著性的配对
            if p_val < 0.001: sig, color, weight = '***', '#222222', 'bold'
            elif p_val < 0.01: sig, color, weight = '**', '#222222', 'bold'
            elif p_val < 0.05: sig, color, weight = '*', '#222222', 'bold'
            else: continue 
            
            x1 = i + offsets[h1_idx]
            x2 = i + offsets[h2_idx]
            h = step * 0.15
            
            # 绘制统计括号
            ax.plot([x1, x1, x2, x2], [current_y, current_y+h, current_y+h, current_y], lw=1.2, color='#666666')
            ax.text((x1+x2)/2, current_y+h*1.1, sig, ha='center', va='bottom', color=color, fontsize=12, fontweight=weight)
            
            current_y += step

# ── 主绘图函数 ────────────────────────────────────────────────────────────────
def plot_participant_grouped_bars(df: pd.DataFrame, save_dir: str):
    fig, axes = plt.subplots(2, 2, figsize=(18, 13)) 
    axes = axes.flatten()
    
    # 格式: (字段名, 标题, 标签显示刻度, 留白下限, 留白上限) —— 与 analyze_results.py 完全一致
    metrics_to_plot = [
        ('clutch_times',     'Total Clutch Times ↓',           [0, 2, 4, 6, 8, 10],      -0.5, 11),
        ('total_time',       'Total Completion Time (s) ↓',    [0, 10, 20, 30, 40, 50],  -3.0, 55),
        ('subjective_score', 'Subjective Score ↑', [0, 20, 40, 60, 80, 100], -5.0, 110),
        ('total_score',      'Total Score ↑',         [0, 20, 40, 60, 80, 100], -5.0, 110)
    ]
    
    # 在这里手动选择与排布您需要展示的参与者
    x_order = ["LYF", "ZYM", "LJS"]
    hue_order = ['Fixed Mode', 'Global Adaptive', 'Phased Adaptive']
    
    handles, labels = [], [] 

    step_list = [0.7,5,5,7]

    for i, (col, title, y_ticks, y_lim_min, y_lim_max) in enumerate(metrics_to_plot):
        ax = axes[i]
        
        # 绘制主柱状图
        barplot = sns.barplot(
            data=df, x='participant', y=col, hue='group',
            order=x_order, hue_order=hue_order, ax=ax,
            palette=_PALETTE, ci='sd', capsize=0.08, 
            edgecolor='#333333', linewidth=1.2, alpha=0.85,
            errwidth=1.5, errcolor='#333333'
        )
        
        if i == 0:
            handles, labels = ax.get_legend_handles_labels()
        
        if ax.get_legend() is not None:
            ax.get_legend().remove()
            
        # 挂载统计检验标注（传入 y_lim_max）
        add_grouped_stat_annotation(ax, df, 'participant', col, 'group', x_order, hue_order, y_lim_max, step_list[i])
        
        # 【应用与 analyze_results.py 相同的刻度限制】
        ax.set_ylim(y_lim_min, y_lim_max)
        ax.set_yticks(y_ticks)
        display_labels = ['Participant 1', 'Participant 2', 'Participant 3']  # 你想显示的名字

        ax.set_xticklabels(display_labels, fontsize=15)
        
        ax.set_title(title, fontsize=17, fontweight='bold', pad=15)
        ax.set_xlabel('Participants' if i >= 5 else '') # 下排显示X轴标签
        ax.set_ylabel('')
        ax.grid(axis='y', linestyle=':', linewidth=1.2, alpha=0.5, color='#A0A0A0')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_linewidth(1.2)
        ax.spines['bottom'].set_linewidth(1.2)

    # 在整个图表最下方添加全局图例
    fig.legend(handles, labels, loc='lower center', bbox_to_anchor=(0.5, 0.02),
               ncol=3, title='Motion Scaling Modes', title_fontsize=15, 
               fontsize=15, frameon=False)
            
    plt.tight_layout(pad=3.5)
    plt.subplots_adjust(bottom=0.15) 

    path = os.path.join(save_dir, 'Participant_Grouped_Bars_2x2.png')
    plt.savefig(path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f'  → 图表已生成: {path}')

# ── 启动入口 ──────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--json', default=None, help='JSON 文件路径')
    args = parser.parse_args()

    script_dir = os.path.dirname(os.path.abspath(__file__))
    json_path = os.path.join(script_dir, 'comp_3method_data.json')
    
    save_dir = os.path.join(script_dir, os.pardir, "Essay_image_results")
    os.makedirs(save_dir, exist_ok=True)

    print(f'[Analyze] 读取数据源: {json_path}')
    df = load_data(json_path)
    
    print('── 正在生成被试者聚类柱状图 (Participant-level Analysis) ──')
    plot_participant_grouped_bars(df, save_dir)
    print('\n[Analyze] 可视化流程完毕。')

if __name__ == '__main__':
    main()