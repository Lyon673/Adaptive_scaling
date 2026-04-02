"""
analyze_results.py
==================
对 subjective_*.json 按记录中的 note 字段分组进行统计分析并可视化。

分组由 JSON 每条记录的 "note" 字段自动推断，无需硬编码。

用法：
    python analyze_results.py
    python analyze_results.py --json path/to/other.json
"""

import json
import argparse
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy import stats
from itertools import combinations

# ── 分组元数据（运行时由数据动态填充）──────────────────────────────────────────
_COLOR_PALETTE = ['#4C72B0', '#55A868', '#C44E52', '#DD8452', '#8172B2', '#937860', '#64B5CD']

GROUP_ORDER  = []   # 按首次出现顺序排列的分组名列表
GROUP_COLORS = {}   # group_name -> color
GROUP_SHORT  = {}   # group_name -> 短标签（用于图表显示）


def _init_groups(df: pd.DataFrame):
    """从 DataFrame 的 group 列推断分组顺序及颜色映射。"""
    global GROUP_ORDER, GROUP_COLORS, GROUP_SHORT
    seen: list = []
    for g in df['group']:
        if g not in seen:
            seen.append(g)
    GROUP_ORDER  = seen
    GROUP_COLORS = {g: _COLOR_PALETTE[i % len(_COLOR_PALETTE)] for i, g in enumerate(GROUP_ORDER)}
    GROUP_SHORT  = {g: g for g in GROUP_ORDER}

# ── 关注的指标列 ───────────────────────────────────────────────────────────────
SUBJECTIVE_ITEMS = [
    'physical_demand', 'temporal_demand', 'controllability',
    'performance', 'mental_demand', 'effort', 'frustration',
]
OBJECTIVE_RAW = [
    'gracefulness', 'smoothness',
    'clutch_times_L', 'clutch_times_R',
    'total_distance', 'total_time',
]
OBJECTIVE_SCORES = [
    'gracefulness_score', 'smoothness_score', 'clutch_times_score',
    'total_distance_score', 'total_time_score', 'objective_total',
]
# 归一化前的原始评分列
_SCORE_COLS_RAW = ['subjective_score', 'objective_total', 'total_score']
# 经过按参与者 min-max 归一化后的列，用于汇总分析
SUMMARY_COLS = ['subjective_score_norm', 'objective_total_norm', 'total_score_norm']


# ── 工具函数 ───────────────────────────────────────────────────────────────────

def load_data(json_path: str) -> pd.DataFrame:
    with open(json_path) as f:
        raw = json.load(f)

    rows = []
    for entry in raw:
        folder = entry['objective_metrics']['data_folder']  # e.g. "10_data_03-22"
        demo_id = int(folder.split('_')[0])

        row = {'demo_id': demo_id}
        row['participant'] = entry.get('participant', 'unknown')
        row['group'] = entry.get('note', 'unknown')
        row.update(entry['ratings'])
        row['subjective_score'] = entry['subjective_score']
        row.update(entry['objective_metrics'])
        row['total_score'] = entry['total_score']
        rows.append(row)

    df = pd.DataFrame(rows).sort_values('demo_id').reset_index(drop=True)
    return df


def normalize_per_participant(df: pd.DataFrame) -> pd.DataFrame:
    """
    对每个参与者的 subjective_score、objective_total、total_score
    独立做 min-max 归一化到 [0, 100]，结果存入对应的 *_norm 列。

    归一化公式：norm = (x - min_p) / (max_p - min_p) * 100
    若某参与者所有值完全相同（极差为 0），则归一化结果统一为 50.0。
    """
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


def desc_stats(series: pd.Series) -> dict:
    return {
        'mean': series.mean(),
        'std':  series.std(ddof=1),
        'min':  series.min(),
        'max':  series.max(),
        'median': series.median(),
        'n':    len(series),
    }


def group_stats(df: pd.DataFrame, col: str) -> pd.DataFrame:
    rows = []
    for gname in GROUP_ORDER:
        sub = df[df['group'] == gname][col]
        d = desc_stats(sub)
        d['group'] = GROUP_SHORT[gname]
        rows.append(d)
    return pd.DataFrame(rows).set_index('group')


def kruskal_pairwise(df: pd.DataFrame, col: str):
    """Kruskal-Wallis 总体检验 + 两两 Mann-Whitney U 检验（Bonferroni 校正）"""
    groups = [df[df['group'] == g][col].values for g in GROUP_ORDER]
    n_groups = len(groups)

    # 总体 KW 检验
    H, p_kw = stats.kruskal(*groups)

    # 两两 MWU（Bonferroni 校正）
    pairs = list(combinations(range(n_groups), 2))
    n_comparisons = len(pairs)
    pairwise = []
    for i, j in pairs:
        _, p_mwu = stats.mannwhitneyu(groups[i], groups[j], alternative='two-sided')
        p_adj = min(p_mwu * n_comparisons, 1.0)   # Bonferroni
        pairwise.append({
            'group_1': GROUP_SHORT[GROUP_ORDER[i]],
            'group_2': GROUP_SHORT[GROUP_ORDER[j]],
            'p_mwu':   p_mwu,
            'p_adj':   p_adj,
            'sig':     '***' if p_adj < 0.001 else ('**' if p_adj < 0.01 else ('*' if p_adj < 0.05 else 'ns')),
        })
    return H, p_kw, pd.DataFrame(pairwise)


# ── 绘图函数 ───────────────────────────────────────────────────────────────────

def _bar_with_error(ax, df: pd.DataFrame, col: str, title: str, ylabel: str):
    """按组绘制带误差棒的柱状图（均值 ± 1 std）。"""
    gnames = GROUP_ORDER
    x = np.arange(len(gnames))
    means = [df[df['group'] == g][col].mean() for g in gnames]
    stds  = [df[df['group'] == g][col].std(ddof=1) for g in gnames]
    colors = [GROUP_COLORS[g] for g in gnames]

    bars = ax.bar(x, means, yerr=stds, color=colors, alpha=0.85,
                  error_kw=dict(ecolor='black', capsize=4, linewidth=1.2), width=0.55)
    ax.set_xticks(x)
    ax.set_xticklabels([GROUP_SHORT[g] for g in gnames], fontsize=8)
    ax.set_title(title, fontsize=9, fontweight='bold')
    ax.set_ylabel(ylabel, fontsize=8)
    ax.grid(axis='y', alpha=0.3)

    # 在柱上标注均值
    for bar, m in zip(bars, means):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + max(stds) * 0.05,
                f'{m:.2f}', ha='center', va='bottom', fontsize=7)


def _boxplot(ax, df: pd.DataFrame, col: str, title: str, ylabel: str):
    gnames = GROUP_ORDER
    data   = [df[df['group'] == g][col].values for g in gnames]
    colors = [GROUP_COLORS[g] for g in gnames]

    bp = ax.boxplot(data, patch_artist=True, widths=0.5,
                    medianprops=dict(color='black', linewidth=2))
    for patch, c in zip(bp['boxes'], colors):
        patch.set_facecolor(c)
        patch.set_alpha(0.75)

    ax.set_xticks(range(1, len(gnames) + 1))
    ax.set_xticklabels([GROUP_SHORT[g] for g in gnames], fontsize=8)
    ax.set_title(title, fontsize=9, fontweight='bold')
    ax.set_ylabel(ylabel, fontsize=8)
    ax.grid(axis='y', alpha=0.3)


def plot_summary(df: pd.DataFrame, save_dir: str):
    """综合评分总览：subjective / objective / total 三个指标的柱图+箱图（参与者内归一化）。"""
    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    fig.suptitle('Experiment Group Comparison — Summary Scores\n'
                 '(per-participant min-max normalized to [0, 100])',
                 fontsize=12, fontweight='bold')

    nice_names = {
        'subjective_score_norm': 'Subjective Score',
        'objective_total_norm':  'Objective Total',
        'total_score_norm':      'Total Score',
    }
    for col_idx, col in enumerate(SUMMARY_COLS):
        nice = nice_names.get(col, col.replace('_', ' ').title())
        _bar_with_error(axes[0, col_idx], df, col, f'{nice} (Mean ± Std)', 'Norm. Score [0-100]')
        _boxplot(axes[1, col_idx], df, col, f'{nice} (Box)', 'Norm. Score [0-100]')

    plt.tight_layout()
    path = os.path.join(save_dir, 'summary_scores.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'  → {path}')


def plot_subjective_items(df: pd.DataFrame, save_dir: str):
    """TLX 各子项分组柱图。"""
    n = len(SUBJECTIVE_ITEMS)
    ncols = 4
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(16, 4 * nrows))
    axes = axes.flatten()
    fig.suptitle('TLX Subjective Sub-Items by Group', fontsize=13, fontweight='bold')

    for i, col in enumerate(SUBJECTIVE_ITEMS):
        _bar_with_error(axes[i], df, col, col.replace('_', ' ').title(), 'Rating')

    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    plt.tight_layout()
    path = os.path.join(save_dir, 'subjective_items.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'  → {path}')


def plot_objective_raw(df: pd.DataFrame, save_dir: str):
    """客观指标原始值分组箱图。"""
    n = len(OBJECTIVE_RAW)
    ncols = 3
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(14, 4 * nrows))
    axes = axes.flatten()
    fig.suptitle('Objective Raw Metrics by Group', fontsize=13, fontweight='bold')

    for i, col in enumerate(OBJECTIVE_RAW):
        _boxplot(axes[i], df, col, col.replace('_', ' ').title(), 'Value')

    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    plt.tight_layout()
    path = os.path.join(save_dir, 'objective_raw.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'  → {path}')


def plot_objective_scores(df: pd.DataFrame, save_dir: str):
    """客观评分分量分组柱图。"""
    n = len(OBJECTIVE_SCORES)
    ncols = 3
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(14, 4 * nrows))
    axes = axes.flatten()
    fig.suptitle('Objective Score Components by Group', fontsize=13, fontweight='bold')

    for i, col in enumerate(OBJECTIVE_SCORES):
        _bar_with_error(axes[i], df, col, col.replace('_', ' ').title(), 'Score')

    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    plt.tight_layout()
    path = os.path.join(save_dir, 'objective_scores.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'  → {path}')


def plot_radar(df: pd.DataFrame, save_dir: str):
    """雷达图：各组在 TLX 子项上的均值轮廓。"""
    items   = SUBJECTIVE_ITEMS
    N       = len(items)
    angles  = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(7, 7), subplot_kw=dict(polar=True))
    ax.set_title('TLX Sub-Items Radar by Group', fontsize=12, fontweight='bold', pad=20)

    for gname in GROUP_ORDER:
        means = [10-df[df['group'] == gname][it].mean() for it in items]
        means += means[:1]
        ax.plot(angles, means, 'o-', linewidth=2,
                color=GROUP_COLORS[gname], label=GROUP_SHORT[gname])
        ax.fill(angles, means, alpha=0.1, color=GROUP_COLORS[gname])

    ax.set_thetagrids(np.degrees(angles[:-1]), [it.replace('_', '\n') for it in items], fontsize=9)
    ax.set_ylim(0, 10)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.15), fontsize=9)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(save_dir, 'radar_tlx.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'  → {path}')


def plot_radar_objective(df: pd.DataFrame, save_dir: str):
    """
    雷达图：各组在客观评分各分量上的均值轮廓。

    各轴独立做 min-max 归一化（0-1），使量纲差异较大的指标可以放在同一图上
    比较形状。轴刻度标注原始均值，方便读数。
    """
    items = [c for c in OBJECTIVE_SCORES if c != 'objective_total']
    labels = [
        'Gracefulness',
        'Smoothness',
        'Clutch\nTimes',
        'Total\nDistance',
        'Total\nTime',
    ]
    N      = len(items)
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    angles += angles[:1]

    # 各指标组均值
    group_means = {
        g: np.array([df[df['group'] == g][it].mean() for it in items])
        for g in GROUP_ORDER
    }

    # min-max 归一化（基于各组均值的范围）
    all_vals = np.stack(list(group_means.values()))
    v_min = all_vals.min(axis=0)
    v_max = all_vals.max(axis=0)
    v_range = np.where(v_max - v_min > 0, v_max - v_min, 1.0)

    def normalize(v):
        return (v - v_min) / v_range

    fig, ax = plt.subplots(figsize=(7, 7), subplot_kw=dict(polar=True))
    ax.set_title('Objective Score Components Radar by Group\n'
                 '(axes normalized to [0, 1] per metric)',
                 fontsize=11, fontweight='bold', pad=22)

    for gname in GROUP_ORDER:
        raw   = group_means[gname]
        normed = normalize(raw).tolist()
        normed += normed[:1]
        ax.plot(angles, normed, 'o-', linewidth=2,
                color=GROUP_COLORS[gname], label=GROUP_SHORT[gname])
        ax.fill(angles, normed, alpha=0.12, color=GROUP_COLORS[gname])

        # 在各顶点标注原始均值
        for ang, nv, rv in zip(angles[:-1], normed[:-1], raw):
            ax.annotate(
                f'{rv:.1f}',
                xy=(ang, nv),
                xytext=(ang, nv + 0.08),
                fontsize=7,
                ha='center', va='center',
                color=GROUP_COLORS[gname],
            )

    ax.set_thetagrids(np.degrees(angles[:-1]), labels, fontsize=9)
    ax.set_ylim(0, 1.25)
    ax.set_yticks([0.25, 0.5, 0.75, 1.0])
    ax.set_yticklabels(['0.25', '0.50', '0.75', '1.00'], fontsize=7, color='grey')
    ax.legend(loc='upper right', bbox_to_anchor=(1.35, 1.15), fontsize=9)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(save_dir, 'radar_objective.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'  → {path}')


def plot_scatter_subj_obj(df: pd.DataFrame, save_dir: str):
    """主观 vs 客观综合得分散点图，按组着色（使用参与者内归一化分数）。"""
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.set_title('Subjective Score vs Objective Score\n'
                 '(per-participant normalized)', fontsize=11, fontweight='bold')

    for gname in GROUP_ORDER:
        sub = df[df['group'] == gname]
        ax.scatter(sub['objective_total_norm'], sub['subjective_score_norm'],
                   color=GROUP_COLORS[gname], label=GROUP_SHORT[gname],
                   s=70, alpha=0.85, edgecolors='white', linewidths=0.5)

    ax.set_xlabel('Objective Total Score (norm.)', fontsize=10)
    ax.set_ylabel('Subjective Score (norm.)', fontsize=10)
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)

    # 添加相关系数
    r, p = stats.pearsonr(df['objective_total_norm'], df['subjective_score_norm'])
    ax.text(0.05, 0.95, f'Pearson r = {r:.3f}  (p={p:.3f})',
            transform=ax.transAxes, fontsize=9,
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

    plt.tight_layout()
    path = os.path.join(save_dir, 'scatter_subj_vs_obj.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'  → {path}')


# ── 文字统计报告 ───────────────────────────────────────────────────────────────

def print_and_save_stats(df: pd.DataFrame, save_dir: str):
    lines = []
    sep  = '=' * 72

    def h1(s):  lines.append(f'\n{sep}\n  {s}\n{sep}')
    def h2(s):  lines.append(f'\n  ── {s} ──')
    def ln(s):  lines.append(s)

    # ── 原始记录预览 ──────────────────────────────────────────────────────────
    h1('0. RAW DATA PREVIEW (raw + per-participant normalized scores)')
    preview_cols = ['demo_id', 'participant', 'group',
                    'subjective_score', 'subjective_score_norm',
                    'objective_total',  'objective_total_norm',
                    'total_score',      'total_score_norm']
    ln(df[preview_cols].to_string(index=False, float_format=lambda x: f'{x:.2f}'))

    # ── 各组描述统计 ──────────────────────────────────────────────────────────
    h1('1. DESCRIPTIVE STATISTICS BY GROUP')
    all_cols = SUMMARY_COLS + SUBJECTIVE_ITEMS + OBJECTIVE_RAW + OBJECTIVE_SCORES
    for col in all_cols:
        h2(col)
        tbl = group_stats(df, col)
        ln(tbl.to_string(float_format=lambda x: f'{x:.4f}'))

    # ── 统计检验 ──────────────────────────────────────────────────────────────
    h1('2. STATISTICAL TESTS  (Kruskal-Wallis + pairwise Mann-Whitney U, Bonferroni)')
    key_cols = SUMMARY_COLS + SUBJECTIVE_ITEMS
    for col in key_cols:
        H, p_kw, pw = kruskal_pairwise(df, col)
        sig_flag = '***' if p_kw < 0.001 else ('**' if p_kw < 0.01 else ('*' if p_kw < 0.05 else 'ns'))
        h2(f'{col}   KW H={H:.3f}  p={p_kw:.4f} {sig_flag}')
        ln(pw.to_string(index=False))

    # ── 综合排名 ──────────────────────────────────────────────────────────────
    h1('3. GROUP RANKING (by mean normalized total_score, descending)')
    ranking = (df.groupby('group')[SUMMARY_COLS]
               .mean()
               .rename(columns={
                   'subjective_score_norm': 'subj_norm',
                   'objective_total_norm':  'obj_norm',
                   'total_score_norm':      'total_norm',
               })
               .sort_values('total_norm', ascending=False))
    ln(ranking.to_string(float_format=lambda x: f'{x:.4f}'))

    report = '\n'.join(lines)
    print(report)

    txt_path = os.path.join(save_dir, 'statistics_report.txt')
    with open(txt_path, 'w') as f:
        f.write(report)
    print(f'\n  → {txt_path}')

    # 同时保存 CSV
    csv_path = os.path.join(save_dir, 'data_with_groups.csv')
    df.to_csv(csv_path, index=False)
    print(f'  → {csv_path}')


# ── 主程序 ────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--json', default=None,
                        help='JSON 文件路径；默认使用脚本同目录下最新的 subjective_*.json')
    parser.add_argument('--out', default=None,
                        help='输出目录；默认与 JSON 文件同目录')
    args = parser.parse_args()

    script_dir = os.path.dirname(os.path.abspath(__file__))

    if args.json is None:
        candidates = sorted(
            [f for f in os.listdir(script_dir) if f.startswith('subjective_') and f.endswith('.json')]
        )
        if not candidates:
            raise FileNotFoundError('未在脚本目录找到 subjective_*.json，请用 --json 指定路径')
        json_path = os.path.join(script_dir, candidates[-1])
    else:
        json_path = args.json

    save_dir = args.out if args.out else os.path.dirname(os.path.abspath(json_path))
    os.makedirs(save_dir, exist_ok=True)

    print(f'[analyze] 读取: {json_path}')
    print(f'[analyze] 输出: {save_dir}\n')

    df = load_data(json_path)
    df = normalize_per_participant(df)

    participants = df['participant'].unique().tolist()
    print(f'[analyze] 参与者: {participants}')
    for p in participants:
        sub = df[df['participant'] == p]
        print(f'         {p}: {len(sub)} 条记录  '
              f'subj [{sub["subjective_score"].min():.1f}, {sub["subjective_score"].max():.1f}]  '
              f'obj [{sub["objective_total"].min():.1f}, {sub["objective_total"].max():.1f}]  '
              f'total [{sub["total_score"].min():.1f}, {sub["total_score"].max():.1f}]')

    _init_groups(df)
    print(f'[analyze] 发现分组（按首次出现顺序）: {GROUP_ORDER}\n')

    print('── 生成图表 ──────────────────────────────────────────────────────────')
    plot_summary(df, save_dir)
    plot_subjective_items(df, save_dir)
    plot_objective_raw(df, save_dir)
    plot_objective_scores(df, save_dir)
    plot_radar(df, save_dir)
    plot_radar_objective(df, save_dir)
    plot_scatter_subj_obj(df, save_dir)

    print('\n── 统计报告 ──────────────────────────────────────────────────────────')
    print_and_save_stats(df, save_dir)

    print('\n[analyze] 全部完成。')


if __name__ == '__main__':
    main()
