"""
analyze_results.py
==================
对 comp_3method_data.json 按记录中的 note 字段分组进行统计分析并生成高水平学术图表。
包含：
 1. 2x2 布局箱体图，包含两两 T-Test 统计检验与 P 值标注 (独立三色配色，刻度解耦带来完美上下留白)。
 2. 主客观多维雷达图 (1x2 并排子图) (保留原 7 色莫兰迪配色)。
"""

import json
import argparse
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.gridspec import GridSpec
from scipy.stats import ttest_ind
from itertools import combinations

# ── 学术级样式设置 ──────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "axes.labelsize": 12,
    "axes.titlesize": 14,
    "font.size": 11,
    "legend.fontsize": 11,
    "xtick.labelsize": 12,
    "ytick.labelsize": 11,
})

# 【严格保留您设定的专属调色板】
_BOX_PALETTE = ['#5681B9', '#5DA076', '#E9A263' ]
_RADAR_PALETTE = ['#5681B9', '#5DA076', '#E9A263' ]

GROUP_ORDER  = []
BOX_COLORS   = {}
RADAR_COLORS = {}
GROUP_SHORT  = {}

def _init_groups(df: pd.DataFrame):
    global GROUP_ORDER, BOX_COLORS, RADAR_COLORS, GROUP_SHORT
    seen: list = []
    for g in df['group']:
        if g not in seen:
            seen.append(g)
    GROUP_ORDER  = seen
    BOX_COLORS   = {g: _BOX_PALETTE[i % len(_BOX_PALETTE)] for i, g in enumerate(GROUP_ORDER)}
    RADAR_COLORS = {g: _RADAR_PALETTE[i % len(_RADAR_PALETTE)] for i, g in enumerate(GROUP_ORDER)}
    GROUP_SHORT  = {g: g for g in GROUP_ORDER}

# ── 关注的指标列 ───────────────────────────────────────────────────────────────
SUBJECTIVE_ITEMS = [
    'physical_demand', 'temporal_demand', 'controllability',
    'performance', 'mental_demand', 'effort', 'frustration',
]
OBJECTIVE_SCORES = [
    'gracefulness_score', 'smoothness_score', 'clutch_times_score',
    'total_distance_score', 'total_time_score', 'objective_total',
]
_SCORE_COLS_RAW = ['total_score', 'subjective_score', 'objective_total']

# ── 工具函数 ───────────────────────────────────────────────────────────────────

def load_data(json_path: str) -> pd.DataFrame:
    with open(json_path) as f:
        raw = json.load(f)

    rows = []
    for entry in raw:
        folder = entry['objective_metrics']['data_folder']
        demo_id = int(folder.split('_')[0])

        row = {'demo_id': demo_id}
        row['participant'] = entry.get('participant', 'unknown')
        row['group'] = entry.get('note', 'unknown')
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

def normalize_per_participant(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for col in _SCORE_COLS_RAW:
        norm_col = col + '_norm'
        df[norm_col] = np.nan
        for p, idx in df.groupby('participant').groups.items():
            vals = df.loc[idx, col]
            lo, hi = vals.min(), vals.max()
            rng = hi - lo
            if rng > 0:
                df.loc[idx, norm_col] = (vals - lo) / rng * 100.0
            else:
                df.loc[idx, norm_col] = 50.0
    return df

# ── 统计学检验括号绘制 (利用专属空白段引擎) ───────────────────────────────────

def add_stat_annotation(ax, df, x_col, y_col, order, y_lim_max):
    if len(order) < 2: return
    
    data_max = df[y_col].max()
    
    # 巧妙利用我们拓展出的绝对空白区 (y_lim_max - data_max)
    avail_space = y_lim_max - data_max
    
    # 让第一个括号的起点脱离数据最高点
    start_y = data_max + avail_space * 0.15
    
    # 在留白区内均匀划分出 3 层的空间
    step = (y_lim_max - start_y) / 3.3
    current_y = start_y
    h = step * 0.15 # 下垂的小边长
    
    if len(order) == 3:
        pairs = [(order[0], order[1]), (order[1], order[2]), (order[0], order[2])]
    else:
        pairs = list(combinations(order, 2))
        
    for i, (g1, g2) in enumerate(pairs):
        data1 = df[df[x_col] == g1][y_col].dropna()
        data2 = df[df[x_col] == g2][y_col].dropna()
        
        if len(data1) < 2 or len(data2) < 2: continue
        
        stat, p_val = ttest_ind(data1, data2)
        
        x1, x2 = order.index(g1), order.index(g2)
        ax.plot([x1, x1, x2, x2], [current_y, current_y+h, current_y+h, current_y], lw=1.2, color='#222222')
        
        if np.isnan(p_val):
            text = "p=NaN"
        else:
            sig = ''
            if p_val < 0.001: sig = '***'
            elif p_val < 0.01: sig = '**'
            elif p_val < 0.05: sig = '*'
            text = f"p={p_val:.3f} {sig}" if sig != '' else f"p={p_val:.3f} "
            
        ax.text((x1+x2)/2, current_y+h*1.1, text, ha='center', va='bottom', color='#222222', fontsize=12, fontweight='bold')
        
        current_y += step

# ── 绘图 1: 2x2 统计检验箱线图 ────────────────────────────────────────────────

def plot_2x2_boxplots_with_stats(df: pd.DataFrame, save_dir: str):
    fig, axes = plt.subplots(2, 2, figsize=(13, 11))
    axes = axes.flatten()
    
    # 【核心逻辑重构】：解耦可视范围(y_lim)和刻度显示(y_ticks)
    # 格式: (字段名, 标题, 标签显示范围, 留白下限, 留白上限)
    metrics_to_plot = [
        ('clutch_times',     'Total Clutch Times',  [0, 2, 4, 6, 8, 10],      -0.5, 12.5),
        ('total_time',       'Total Completion Time (s)', [0, 10, 20, 30, 40, 50, 60],  -3.0, 75.0),
        ('subjective_score', 'Subjective Score',          [0, 20, 40, 60, 80, 100], -5.0, 125.0),
        ('total_score',      'Total Score',               [0, 20, 40, 60, 80, 100], -5.0, 125.0)
    ]

    for i, (col, title, y_ticks, y_lim_min, y_lim_max) in enumerate(metrics_to_plot):
        ax = axes[i]
        
        sns.boxplot(
            data=df, x='group', y=col, ax=ax, order=GROUP_ORDER,
            palette=BOX_COLORS, showfliers=True, width=0.45,
            boxprops=dict(alpha=0.85, edgecolor='#222222', linewidth=1.5),
            medianprops=dict(color='#111111', linewidth=2.2),
            whiskerprops=dict(color='#222222', linewidth=1.5, linestyle='solid'),
            capprops=dict(color='#222222', linewidth=1.5),
            flierprops=dict(marker='o', markerfacecolor='#999999', markersize=6, alpha=0.6, markeredgecolor='none')
        )

        # 传入绝对的图表上限，利用上限留白绘制 P 值
        add_stat_annotation(ax, df, 'group', col, GROUP_ORDER, y_lim_max)

        # 关键设定：图表画幅伸展，但数字刻度锁定
        ax.set_ylim(y_lim_min, y_lim_max)
        ax.set_yticks(y_ticks)

        ax.set_title(title, fontsize=15, fontweight='bold', pad=15)
        ax.set_xlabel('')
        ax.set_ylabel('')
        ax.set_xticklabels([GROUP_SHORT[g] for g in GROUP_ORDER], fontsize=13)

        ax.grid(axis='y', linestyle=':', linewidth=1.2, alpha=0.5, color='#A0A0A0')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_linewidth(1.2)
        ax.spines['bottom'].set_linewidth(1.2)
        ax.spines['left'].set_color('#222222')
        ax.spines['bottom'].set_color('#222222')
        ax.tick_params(colors='#222222', width=1.2)

    plt.tight_layout(pad=2.0)
    path = os.path.join(save_dir, 'statistical_boxplots_2x2.png')
    plt.savefig(path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f'  → 图 1 已生成: {path}')

# ── 绘图 2: 雷达图组合 (主观 + 客观) ──────────────────────────────────────────

def plot_radar_charts(df: pd.DataFrame, save_dir: str):
    fig = plt.figure(figsize=(15, 7))
    gs = GridSpec(1, 2, wspace=0.3)

    def draw_radar(ax, angles, data_dict, labels, title, ymax, yticks, yticklabels, draw_raw_text=False, raw_dict=None):
        ax.set_theta_offset(np.pi / 2)
        ax.set_theta_direction(-1)
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(labels, fontsize=15)
        # ax.tick_params(axis='x', pad=25)
        for label in ax.get_xticklabels()[1:]:
            label.set_y(label.get_position()[1] - 0.12)
        for gname in GROUP_ORDER:
            vals = data_dict[gname].tolist()
            vals += vals[:1]
            
            color = RADAR_COLORS[gname]
            ax.plot(angles, vals, 'o-', linewidth=2.5, color=color, label=GROUP_SHORT[gname])
            ax.fill(angles, vals, alpha=0.15, color=color)
            
            if draw_raw_text and raw_dict is not None:
                raw_vals = raw_dict[gname]
                for ang, nv, rv in zip(angles[:-1], vals[:-1], raw_vals):
                    ax.annotate(f'{rv:.1f}', xy=(ang, nv), xytext=(ang, nv + 0.1),
                                fontsize=12, ha='center', va='center', color=color, weight='bold')

        ax.set_ylim(0, ymax)
        ax.set_yticks(yticks)
        ax.set_yticklabels(yticklabels, color='grey', size=12)
        ax.set_title(title, fontweight='bold', pad=30)
        ax.grid(True, linestyle='--', alpha=0.5)

    # 子图 (a): 主观 TLX 雷达图
    ax1 = fig.add_subplot(gs[0, 0], polar=True)
    subj_angles = np.linspace(0, 2 * np.pi, len(SUBJECTIVE_ITEMS), endpoint=False).tolist()
    subj_angles += subj_angles[:1]
    subj_labels = [
    'Physical Demand', 'Temporal\nDemand', 'Controllability',
    'Performance', 'Mental\nDemand', 'Effort', 'Frustration',
    ]
    
    subj_data = {}
    for gname in GROUP_ORDER:
        subj_data[gname] = np.array([10 - df[df['group'] == gname][it].mean() for it in SUBJECTIVE_ITEMS])
        
    draw_radar(ax1, subj_angles, subj_data, subj_labels, 
               '(a) Subjective TLX Score', 
               ymax=10, yticks=[2, 4, 6, 8], yticklabels=['2', '4', '6', '8'])

    # 子图 (b): 客观指标雷达图
    ax2 = fig.add_subplot(gs[0, 1], polar=True)
    obj_items = [c for c in OBJECTIVE_SCORES if c != 'objective_total']
    obj_labels = ['Log Curvature', 'Log Jerk', 'Clutch Times', 'PSM Movement\nDistance', 'Total Time']
    obj_angles = np.linspace(0, 2 * np.pi, len(obj_items), endpoint=False).tolist()
    obj_angles += obj_angles[:1]
    
    raw_means = {g: np.array([df[df['group'] == g][it].mean() for it in obj_items]) for g in GROUP_ORDER}
    all_raw_vals = np.stack(list(raw_means.values()))
    v_min, v_max = all_raw_vals.min(axis=0), all_raw_vals.max(axis=0)
    v_range = np.where(v_max - v_min > 0, v_max - v_min, 1.0)
    
    obj_normed = {g: (raw_means[g] - v_min) / v_range for g in GROUP_ORDER}
    
    draw_radar(ax2, obj_angles, obj_normed, obj_labels, 
               '(b) Normalized Objective Score', 
               ymax=1.2, yticks=[0.25, 0.5, 0.75, 1.0], yticklabels=['0.25', '0.5', '0.75', '1.0'],
               draw_raw_text=True, raw_dict=raw_means)

    handles, labels = ax1.get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', bbox_to_anchor=(0.5, -0.02),
               ncol=len(GROUP_ORDER), title="Motion Scaling Modes", frameon=False, fontsize=13, title_fontsize=13)

    plt.tight_layout()
    path = os.path.join(save_dir, 'comprehensive_radar_charts.png')
    plt.savefig(path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f'  → 图 2 已生成: {path}')

# ── 主程序 ────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--json', default=None, help='JSON 文件路径')
    args = parser.parse_args()

    script_dir = os.path.dirname(os.path.abspath(__file__))
    json_path = os.path.join(script_dir, 'comp_3method_data.json')
    
    save_dir = os.path.join(script_dir, os.pardir, "Essay_image_results")
    os.makedirs(save_dir, exist_ok=True)

    print(f'[analyze] 读取: {json_path}')
    
    df = load_data(json_path)
    df = normalize_per_participant(df)
    _init_groups(df)

    print(f'[analyze] 发现分组: {GROUP_ORDER}\n')
    print('── 生成学术集成图表 ──────────────────────────────────────────────')
    
    plot_2x2_boxplots_with_stats(df, save_dir)
    plot_radar_charts(df, save_dir)
    
    print('\n[analyze] 全部可视化完成。')

if __name__ == '__main__':
    main()