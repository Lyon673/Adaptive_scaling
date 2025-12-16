"""
轨迹相似度统计分析工具
比较两组演示轨迹在多种相似度度量下的统计学差异

相似度度量方法：
1. DTW距离 (Dynamic Time Warping)
2. Chamfer距离
3. Fréchet距离
4. 功率谱密度主频相似度
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.spatial.distance import directed_hausdorff
from scipy.fft import fft, fftfreq
from scipy.interpolate import interp1d
from fastdtw import fastdtw
from itertools import combinations
import os
from TbD import get_standard_trajectory
from load_data import load_demonstrations_state, load_demonstrations_label
from sklearn.preprocessing import StandardScaler

# 设置绘图样式
sns.set_style("whitegrid")
plt.rcParams['font.size'] = 10


class TrajectoryDistanceCalculator:
    """计算轨迹之间的各种距离度量"""
    
    @staticmethod
    def dtw_distance(traj1, traj2):
        """
        计算归一化的DTW距离（消除轨迹长度影响）
        Args:
            traj1, traj2: (T, D) 轨迹数据，T为时间步，D为特征维度
        Returns:
            归一化的DTW距离
        """
        distance, path = fastdtw(traj1, traj2, dist=lambda x, y: np.linalg.norm(x - y))
        
        # 方法1: 除以匹配路径长度（推荐）
        normalized_distance = distance / len(path)
        
        # 方法2（备选）: 除以两个轨迹长度的平均值
        # avg_length = (len(traj1) + len(traj2)) / 2
        # normalized_distance = distance / avg_length
        
        return normalized_distance
    
    @staticmethod
    def chamfer_distance(traj1, traj2):
        """
        计算Chamfer距离（对称）
        Args:
            traj1, traj2: (T, D) 轨迹数据
        Returns:
            Chamfer距离
        """
        # traj1中每个点到traj2的最小距离
        dist1 = np.mean(np.min(np.linalg.norm(
            traj1[:, np.newaxis, :] - traj2[np.newaxis, :, :], axis=2), axis=1))
        
        # traj2中每个点到traj1的最小距离
        dist2 = np.mean(np.min(np.linalg.norm(
            traj2[:, np.newaxis, :] - traj1[np.newaxis, :, :], axis=2), axis=1))
        
        return (dist1 + dist2) / 2
    
    @staticmethod
    def frechet_distance(traj1, traj2):
        """
        计算离散Fréchet距离（简化版本）
        Args:
            traj1, traj2: (T, D) 轨迹数据
        Returns:
            Fréchet距离的近似值
        """
        n1, n2 = len(traj1), len(traj2)
        ca = np.full((n1, n2), -1.0)
        
        def c(i, j):
            if ca[i, j] > -1:
                return ca[i, j]
            
            dist = np.linalg.norm(traj1[i] - traj2[j])
            
            if i == 0 and j == 0:
                ca[i, j] = dist
            elif i > 0 and j == 0:
                ca[i, j] = max(c(i-1, 0), dist)
            elif i == 0 and j > 0:
                ca[i, j] = max(c(0, j-1), dist)
            elif i > 0 and j > 0:
                ca[i, j] = max(min(c(i-1, j), c(i-1, j-1), c(i, j-1)), dist)
            else:
                ca[i, j] = float('inf')
            
            return ca[i, j]
        
        return c(n1-1, n2-1)
    
    @staticmethod
    def power_spectrum_similarity(traj1, traj2, top_k=5):
        """
        基于功率谱密度主频成分的相似度
        Args:
            traj1, traj2: (T, D) 轨迹数据
            top_k: 考虑前k个主频
        Returns:
            主频差异（越小越相似）
        """
        def get_dominant_frequencies(traj, k=top_k):
            """提取轨迹的主频成分"""
            # 对每个维度计算FFT
            n = len(traj)
            freqs = fftfreq(n)[:n//2]
            
            all_peaks = []
            for dim in range(traj.shape[1]):
                # 去均值（去除DC分量），这是频谱分析的标准做法
                signal = traj[:, dim] - np.mean(traj[:, dim])
                
                # FFT
                fft_vals = fft(signal)
                power = np.abs(fft_vals[:n//2]) ** 2
                
                # 排除频率0（DC分量），因为DC分量不包含频率信息
                # 从索引1开始，排除freqs[0]（即频率0）
                if len(power) > 1:
                    power_no_dc = power[1:]  # 排除DC分量
                    freqs_no_dc = freqs[1:]  # 对应的频率
                    
                    # 找到前k个最大功率对应的频率
                    top_indices = np.argsort(power_no_dc)[-k:][::-1]
                    dominant_freqs = freqs_no_dc[top_indices]
                    dominant_powers = power_no_dc[top_indices]
                    
                    all_peaks.extend(zip(dominant_freqs, dominant_powers))
                else:
                    # 如果数据太短，无法排除DC，则使用原始方法
                    top_indices = np.argsort(power)[-k:][::-1]
                    dominant_freqs = freqs[top_indices]
                    dominant_powers = power[top_indices]
                    all_peaks.extend(zip(dominant_freqs, dominant_powers))
            
            # 按功率排序，取前k个
            all_peaks.sort(key=lambda x: x[1], reverse=True)
            return np.array([f for f, p in all_peaks[:k]])
        
        freq1 = get_dominant_frequencies(traj1, top_k)
        freq2 = get_dominant_frequencies(traj2, top_k)
        
        # 计算频率差异（使用最优匹配）
        distances = []
        for f1 in freq1:
            min_dist = np.min(np.abs(freq2 - f1))
            distances.append(min_dist)
        
        return np.mean(distances)


class TrajectoryGroupComparison:
    """比较两组轨迹的相似度分布"""
    
    def __init__(self):
        self.calculator = TrajectoryDistanceCalculator()
        
        # 定义两组数据
        self.group1_indices = list(range(0, 30))  # 0~29
        self.group2_indices = list(range(28, 45)) + list(range(47, 51)) + list(range(55, 60))  # 28~49 + 47~51 + 55~59
        
        print(f"Group 1: demos {self.group1_indices[0]}-{self.group1_indices[-1]} ({len(self.group1_indices)} demos)")
        print(f"Group 2: demos 28-49 + 55-59 ({len(self.group2_indices)} demos)")
        
    def load_trajectories(self):
        """加载轨迹数据"""
        print("\nLoading trajectory data...")
        
        # 合并所有数据
        all_demos = load_demonstrations_state(shuffle=False)
        all_labels = load_demonstrations_label(shuffle=False)
        # 0-6+8-14 are positions and orientation, except 7 and 15
        all_demos = [demo[:, [0,1,2,3,4,5,6,8,9,10,11,12,13,14]] for demo in all_demos]
        # scaler = StandardScaler()

        # all_data_stacked = np.vstack(all_demos)
        # scaler.fit(all_data_stacked)
        # scaled_demos = [scaler.transform(arr) for arr in all_demos]
        
        print(f"Total demonstrations loaded: {len(all_demos)}")

        # 根据标签截取有效片段（从第一个类别0到最后一个类别5）
        # 保持索引对齐：对于不完整的demo，保留原始数据并给出警告
        self.trajectories = []
        incomplete_demos = []
        
        for idx, (demo, labels) in enumerate(zip(all_demos, all_labels)):
            labels_arr = np.array(labels)

            # 找到类别0的起始和类别5的结束
            start_candidates = np.where(labels_arr == 0)[0]
            end_candidates = np.where(labels_arr == 5)[0]

            # 只处理同时包含类别0和类别5的demo（保证完整性）
            if len(start_candidates) > 0 and len(end_candidates) > 0:
                start_idx = start_candidates[0]  # 第一个类别0
                end_idx = end_candidates[-1]      # 最后一个类别5
                
                # 保证索引合法且顺序正确
                if start_idx <= end_idx:
                    self.trajectories.append(demo[start_idx:end_idx + 1])
                    #print(f"start:{start_idx}, end:{end_idx}")
                else:
                    # 理论上不应该出现，但防御性处理
                    print(f"Warning: demo {idx} has invalid label order (class 0 after class 5), keeping full demo.")
                    self.trajectories.append(demo)
                    incomplete_demos.append(idx)
            else:
                # 缺少类别0或5，说明demo不完整，保留原始数据
                missing = []
                if len(start_candidates) == 0:
                    missing.append("class 0")
                if len(end_candidates) == 0:
                    missing.append("class 5")
                print(f"Warning: demo {idx} is missing {' and '.join(missing)}, keeping full demo.")
                self.trajectories.append(demo)
                incomplete_demos.append(idx)
        
        if incomplete_demos:
            print(f"\nTotal incomplete demos (kept full): {len(incomplete_demos)} (indices: {incomplete_demos})")
        
        print(f"Total valid demos (class 0-5): {len(all_demos) - len(incomplete_demos)}")
            
        print(f"max demonstration length: {max(len(traj) for traj in self.trajectories)}")
        
        return self.trajectories
    
    def compute_pairwise_distances(self, group_indices, distance_func, method_name):
        """
        计算组内所有轨迹对的距离
        Args:
            group_indices: 组内demo索引列表
            distance_func: 距离计算函数
            method_name: 方法名称（用于进度显示）
        Returns:
            距离列表
        """
        distances = []
        pairs = list(combinations(group_indices, 2))
        
        print(f"  Computing {len(pairs)} pairwise distances...")
        
        for i, (idx1, idx2) in enumerate(pairs):
            if i % 50 == 0:
                print(f"    Progress: {i}/{len(pairs)}")
            
            traj1 = self.trajectories[idx1]
            traj2 = self.trajectories[idx2]
            
            dist = distance_func(traj1, traj2)
            distances.append(dist)
        
        return np.array(distances)

    def compute_standard_distance(self, group_indices, distance_func, standard_traj, use_position_only=False):
        distances = []
        
        print(f"  Computing {len(group_indices)} distances...")
        
        for i in range(len(group_indices)):
            if i % 10 == 0:
                print(f"    Progress: {i}/{len(group_indices)}")
            
            traj = self.trajectories[i][:, [0,1,2,7,8,9]] if use_position_only else self.trajectories[i]
            
            dist = distance_func(traj, standard_traj)
            distances.append(dist)
        
        return np.array(distances)

    
    def analyze_all_methods(self):
        """对所有相似度方法进行分析"""
        self.load_trajectories()
        
        methods = {
            'DTW': self.calculator.dtw_distance,
            'Chamfer': self.calculator.chamfer_distance,
            'Frechet': self.calculator.frechet_distance,
            'Power Spectrum': self.calculator.power_spectrum_similarity
        }
        
        results = {}

        standard_traj1 = get_standard_trajectory('group1', use_position_only=True)
        standard_traj2 = get_standard_trajectory('group2', use_position_only=True)
        for method_name, distance_func in methods.items():
            print(f"\n{'='*60}")
            print(f"Analyzing method: {method_name}")
            print(f"{'='*60}")
            
            # 计算组1的距离
            print(f"Group 1 (demos 0-29):")
            # group1_distances = self.compute_pairwise_distances(
            #     self.group1_indices, distance_func, method_name)
            group1_distances = self.compute_standard_distance(
                self.group1_indices, distance_func, standard_traj1, use_position_only=True)
            
            # 计算组2的距离
            print(f"Group 2 (demos 28-49 + 55-59):")
            # group2_distances = self.compute_pairwise_distances(
            #     self.group2_indices, distance_func, method_name)
            group2_distances = self.compute_standard_distance(
                self.group2_indices, distance_func, standard_traj2, use_position_only=True)
            
            # 统计检验
            stat_results = self.statistical_tests(group1_distances, group2_distances)
            
            results[method_name] = {
                'group1': group1_distances,
                'group2': group2_distances,
                'stats': stat_results
            }
            
            # 打印统计结果
            self.print_statistics(method_name, group1_distances, group2_distances, stat_results)
        
        return results
    
    def statistical_tests(self, group1, group2):
        """进行统计检验"""
        # 描述性统计
        stats_dict = {
            'group1_mean': np.mean(group1),
            'group1_std': np.std(group1),
            'group1_median': np.median(group1),
            'group2_mean': np.mean(group2),
            'group2_std': np.std(group2),
            'group2_median': np.median(group2),
        }
        
        # Independent t-test
        t_stat, t_pvalue = stats.ttest_ind(group1, group2)
        stats_dict['t_statistic'] = t_stat
        stats_dict['t_pvalue'] = t_pvalue
        
        # Mann-Whitney U test (non-parametric)
        u_stat, u_pvalue = stats.mannwhitneyu(group1, group2, alternative='two-sided')
        stats_dict['mannwhitney_u'] = u_stat
        stats_dict['mannwhitney_pvalue'] = u_pvalue
        
        # Kolmogorov-Smirnov test (distribution equality)
        ks_stat, ks_pvalue = stats.ks_2samp(group1, group2)
        stats_dict['ks_statistic'] = ks_stat
        stats_dict['ks_pvalue'] = ks_pvalue
        
        # Levene's test for equality of variances
        levene_stat, levene_pvalue = stats.levene(group1, group2)
        stats_dict['levene_statistic'] = levene_stat
        stats_dict['levene_pvalue'] = levene_pvalue
        
        # Effect size (Cohen's d) with zero-variance guard to avoid divide-by-zero/NaN
        pooled_var = (np.var(group1) + np.var(group2)) / 2
        if pooled_var == 0 or np.isnan(pooled_var):
            cohens_d = np.nan
        else:
            cohens_d = (np.mean(group1) - np.mean(group2)) / np.sqrt(pooled_var)
        stats_dict['cohens_d'] = cohens_d
        
        return stats_dict
    
    def print_statistics(self, method_name, group1, group2, stats_dict):
        """打印统计结果"""
        print(f"\n{'-'*60}")
        print(f"Statistical Results for {method_name}")
        print(f"{'-'*60}")
        
        print(f"\nDescriptive Statistics:")
        print(f"  Group 1: Mean={stats_dict['group1_mean']:.4f}, "
              f"Std={stats_dict['group1_std']:.4f}, "
              f"Median={stats_dict['group1_median']:.4f}")
        print(f"  Group 2: Mean={stats_dict['group2_mean']:.4f}, "
              f"Std={stats_dict['group2_std']:.4f}, "
              f"Median={stats_dict['group2_median']:.4f}")
        
        print(f"\nHypothesis Tests:")
        print(f"  Independent t-test:")
        print(f"    t-statistic = {stats_dict['t_statistic']:.4f}")
        print(f"    p-value = {stats_dict['t_pvalue']:.4e}")
        print(f"    Significant: {'YES' if stats_dict['t_pvalue'] < 0.05 else 'NO'} (α=0.05)")
        
        print(f"\n  Mann-Whitney U test (non-parametric):")
        print(f"    U-statistic = {stats_dict['mannwhitney_u']:.4f}")
        print(f"    p-value = {stats_dict['mannwhitney_pvalue']:.4e}")
        print(f"    Significant: {'YES' if stats_dict['mannwhitney_pvalue'] < 0.05 else 'NO'} (α=0.05)")
        
        print(f"\n  Kolmogorov-Smirnov test (distribution equality):")
        print(f"    KS-statistic = {stats_dict['ks_statistic']:.4f}")
        print(f"    p-value = {stats_dict['ks_pvalue']:.4e}")
        print(f"    Significant: {'YES' if stats_dict['ks_pvalue'] < 0.05 else 'NO'} (α=0.05)")
        
        print(f"\n  Levene's test (variance equality):")
        print(f"    statistic = {stats_dict['levene_statistic']:.4f}")
        print(f"    p-value = {stats_dict['levene_pvalue']:.4e}")
        
        print(f"\n  Effect Size (Cohen's d): {stats_dict['cohens_d']:.4f}")
        effect_interpretation = "small" if abs(stats_dict['cohens_d']) < 0.5 else \
                               "medium" if abs(stats_dict['cohens_d']) < 0.8 else "large"
        print(f"    Interpretation: {effect_interpretation}")
    
    def visualize_results(self, results, save_dir=os.path.join(os.path.dirname(__file__), 'LSTM_visualization_results')):
        """可视化所有结果"""
        os.makedirs(save_dir, exist_ok=True)
        
        n_methods = len(results)
        fig, axes = plt.subplots(2, n_methods, figsize=(5*n_methods, 10))
        
        if n_methods == 1:
            axes = axes.reshape(-1, 1)
        
        for idx, (method_name, data) in enumerate(results.items()):
            group1 = data['group1']
            group2 = data['group2']
            stats_dict = data['stats']
            
            # 子图1: 分布直方图
            ax1 = axes[0, idx]
            ax1.hist(group1, bins=30, alpha=0.6, label='Group 1 (demos 0-29)', 
                    color='blue', density=True)
            ax1.hist(group2, bins=30, alpha=0.6, label='Group 2 (demos 28-49,55-59)', 
                    color='red', density=True)
            ax1.axvline(np.mean(group1), color='blue', linestyle='--', linewidth=2, 
                       label=f'Group 1 mean={np.mean(group1):.2f}')
            ax1.axvline(np.mean(group2), color='red', linestyle='--', linewidth=2,
                       label=f'Group 2 mean={np.mean(group2):.2f}')
            ax1.set_xlabel(f'{method_name} Distance')
            ax1.set_ylabel('Density')
            ax1.set_title(f'{method_name} Distance Distribution')
            ax1.legend(fontsize=8)
            ax1.grid(True, alpha=0.3)
            
            # 子图2: 箱线图
            ax2 = axes[1, idx]
            bp = ax2.boxplot([group1, group2], labels=['Group 1', 'Group 2'],
                             patch_artist=True, widths=0.6)
            
            # 设置颜色
            bp['boxes'][0].set_facecolor('lightblue')
            bp['boxes'][1].set_facecolor('lightcoral')
            
            ax2.set_ylabel(f'{method_name} Distance')
            ax2.set_title(f'{method_name} Comparison')
            ax2.grid(True, alpha=0.3, axis='y')
            
            # # 添加显著性标记
            # if stats_dict['t_pvalue'] < 0.001:
            #     sig_text = '***'
            # elif stats_dict['t_pvalue'] < 0.01:
            #     sig_text = '**'
            # elif stats_dict['t_pvalue'] < 0.05:
            #     sig_text = '*'
            # else:
            #     sig_text = 'ns'
            
            y_max = max(np.max(group1), np.max(group2))
            y_min = min(np.min(group1), np.min(group2))
            y_range = y_max - y_min
            
            # # 在两组之间上方添加显著性标记
            # ax2.text(1.5, y_max + y_range*0.05, 
            #         ha='center', va='bottom', fontsize=16, fontweight='bold')
            
            # 在图的右上角添加统计信息文本框
            stats_text = (f't-test p: {stats_dict["t_pvalue"]:.3f}\n'
                         f'K-S test statistic: {stats_dict["ks_statistic"]:.3f}\n'
                         f"Cohen's d: {stats_dict['cohens_d']:.3f}")
            
            # 创建带背景的文本框
            ax2.text(0.98, 0.98, stats_text,
                    transform=ax2.transAxes,
                    verticalalignment='top',
                    horizontalalignment='right',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8, edgecolor='black', linewidth=1.5),
                    fontsize=9,
                    fontfamily='monospace')
        
        plt.tight_layout()
        save_path = os.path.join(save_dir, 'trajectory_similarity_comparison.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"\nVisualization saved to: {save_path}")
        plt.show()
        
        # 创建汇总表格图
        self.create_summary_table(results, save_dir)
    
    def create_summary_table(self, results, save_dir):
        """创建统计检验结果汇总表"""
        fig, ax = plt.subplots(figsize=(14, 6))
        ax.axis('tight')
        ax.axis('off')
        
        # 准备表格数据
        headers = ['Method', 'Group1\nMean±Std', 'Group2\nMean±Std', 
                  't-test\np-value', 'Mann-Whitney\np-value', 'K-S\nstatistic', "Cohen's d", 'Significant?']
 
        
        table_data = []
        for method_name, data in results.items():
            s = data['stats']
            row = [
                method_name,
                f"{s['group1_mean']:.3f}±{s['group1_std']:.3f}",
                f"{s['group2_mean']:.3f}±{s['group2_std']:.3f}",
                f"{s['t_pvalue']:.2e}",
                f"{s['mannwhitney_pvalue']:.2e}",
                f"{s['ks_statistic']:.3f}",
                f"{s['cohens_d']:.3f}",
                'YES' if s['t_pvalue'] < 0.001 else 'NO'
            ]
            table_data.append(row)
        
        table = ax.table(cellText=table_data, colLabels=headers,
                        cellLoc='center', loc='center',
                        colWidths=[0.10, 0.14, 0.14, 0.11, 0.11, 0.11, 0.10, 0.10])
        
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 2)
        
        # 设置表头样式
        for i in range(len(headers)):
            table[(0, i)].set_facecolor('#4472C4')
            table[(0, i)].set_text_props(weight='bold', color='white')
        
        # 设置行颜色
        for i in range(1, len(table_data) + 1):
            for j in range(len(headers)):
                if i % 2 == 0:
                    table[(i, j)].set_facecolor('#E7E6E6')
                
                # 突出显示显著性结果
                if j == len(headers) - 1:  # Significant列
                    if table_data[i-1][j] == 'YES':
                        table[(i, j)].set_facecolor('#90EE90')
                        table[(i, j)].set_text_props(weight='bold')
                    else:
                        table[(i, j)].set_facecolor('#FFB6C6')
        
        plt.title('Statistical Comparison Summary\nGroup 1 (the same formula) vs Group 2 (different formulas)\n '+r"$\alpha$ = 0.001",
                 fontsize=14, fontweight='bold', pad=20)
        
        save_path = os.path.join(save_dir, 'statistical_summary_table.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Summary table saved to: {save_path}")
        plt.show()


def main():
    """主函数"""
    print("="*60)
    print("Trajectory Similarity Statistical Analysis")
    print("="*60)
    
    # 创建分析对象
    comparator = TrajectoryGroupComparison()
    
    # 执行分析
    results = comparator.analyze_all_methods()
    
    # 可视化结果
    comparator.visualize_results(results)
    
    print("\n" + "="*60)
    print("Analysis Complete!")
    print("="*60)


if __name__ == "__main__":
    main()

