"""
Statistical Analysis for Different Speed Groups
Analyzes: total_time, total_distance, clutch_times, gracefulness, smoothness
Groups: Slow (45-54), Medium (0-29), Fast (55-64)
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import f_oneway, kruskal, mannwhitneyu, shapiro, levene
import pandas as pd
import os
import sys

# Add parent directory to path
current_file_path = os.path.abspath(__file__)
current_dir = os.path.dirname(current_file_path)
project_dir = os.path.dirname(current_dir)
sys.path.insert(0, project_dir)

from gracefulness import cal_GS, calculate_distribution
from ipa import ipa_cal, Pupil


class MetricsStatistics:
    def __init__(self, data_base_dir):
        """
        Initialize the statistics analyzer
        
        Args:
            data_base_dir: Base directory containing data folders
        """
        self.data_base_dir = data_base_dir
        
        # Define groups
        self.groups = {
            'Slow': list(range(45, 55)),      # 45-54
            'Medium': list(range(20, 30)),      # 0-29
            'Fast': list(range(55, 65))        # 55-64
        }
        
        # Metrics to analyze
        self.metrics = ['total_time', 'total_distance', 'clutch_times', 
                       'ipa', 'gracefulness', 'smoothness']
        
        self.data = {group: {metric: [] for metric in self.metrics} 
                    for group in self.groups.keys()}
    
    def calculate_ipa_from_pupil(self, data_dir):
        """
        Calculate IPA from pupil data using left and right pupil diameters
        
        Args:
            data_dir: Directory containing pupil data files
            
        Returns:
            Average IPA value from both eyes
        """
        try:
            # Load pupil data
            pupilL_path = os.path.join(data_dir, 'pupilL_data.npy')
            pupilR_path = os.path.join(data_dir, 'pupilR_data.npy')
            
            if not (os.path.exists(pupilL_path) and os.path.exists(pupilR_path)):
                print(f"  Warning: Pupil data not found in {data_dir}")
                return None
            
            pupilL_data = np.load(pupilL_path, allow_pickle=True)
            pupilR_data = np.load(pupilR_path, allow_pickle=True)
            
            # pupil_data format: [diameter, timestamp]
            # Create Pupil objects for ipa_cal function
            pupilL_objects = [Pupil(pupilL_data[i, 0], pupilL_data[i, 1]) 
                             for i in range(len(pupilL_data))]
            pupilR_objects = [Pupil(pupilR_data[i, 0], pupilR_data[i, 1]) 
                             for i in range(len(pupilR_data))]
            
            # Calculate IPA for each eye
            ipaL = ipa_cal(pupilL_objects)
            ipaR = ipa_cal(pupilR_objects)
            
            # Handle None returns
            if ipaL is None or ipaR is None:
                if ipaL is not None:
                    return ipaL
                elif ipaR is not None:
                    return ipaR
                else:
                    return None
            
            # Return average IPA
            avg_ipa = (ipaL + ipaR) / 2.0
            return avg_ipa
            
        except Exception as e:
            print(f"  Error calculating IPA: {e}")
            return None
        
    def load_data(self):
        """Load data for all groups"""
        print("Loading data...")
        
        for group_name, data_ids in self.groups.items():
            print(f"\nProcessing {group_name} group (data {min(data_ids)}-{max(data_ids)})...")
            
            for data_id in data_ids:
                data_dir = os.path.join(self.data_base_dir, f'{data_id}_data_12-01')
                
                if not os.path.exists(data_dir):
                    print(f"  Warning: {data_dir} not found, skipping...")
                    continue
                
                try:
                    # Load gracefulness and smoothness
                    # cal_GS returns tuple (G_left, S_left) when use_left=True, use_right=False
                    gracefulness, smoothness = cal_GS(f'{data_id}_data_12-01')
                    
                    # Calculate IPA from pupil data
                    ipa = self.calculate_ipa_from_pupil(data_dir)
                    
                    # Load other metrics
                    clutch_times_path = os.path.join(data_dir, 'clutch_times.npy')
                    total_distance_path = os.path.join(data_dir, 'total_distance.npy')
                    total_time_path = os.path.join(data_dir, 'total_time.npy')
                    
                    if all(os.path.exists(p) for p in [clutch_times_path, total_distance_path, total_time_path]) and ipa is not None:
                        clutch_times = np.load(clutch_times_path, allow_pickle=True)
                        total_distance = np.load(total_distance_path, allow_pickle=True)
                        total_time = np.load(total_time_path, allow_pickle=True)[0]
                        
                        # Store data
                        self.data[group_name]['gracefulness'].append(gracefulness)
                        self.data[group_name]['smoothness'].append(smoothness)
                        self.data[group_name]['clutch_times'].append(clutch_times[0]+clutch_times[1])
                        self.data[group_name]['ipa'].append(ipa)
                        self.data[group_name]['total_distance'].append(total_distance[0])
                        self.data[group_name]['total_time'].append(total_time)
                        
                        print(f"  ✓ Data {data_id} loaded successfully (IPA={ipa:.4f})")
                    else:
                        if ipa is None:
                            print(f"  Warning: Could not calculate IPA for {data_dir}")
                        else:
                            print(f"  Warning: Missing files in {data_dir}")
                        
                except Exception as e:
                    print(f"  Error loading data {data_id}: {e}")
        
        # Print summary
        print("\n" + "="*60)
        print("Data Loading Summary:")
        print("="*60)
        for group_name in self.groups.keys():
            n = len(self.data[group_name]['total_time'])
            print(f"{group_name:10s}: {n:3d} samples")
        print("="*60)
    
    def check_normality(self):
        """Check normality assumption using Shapiro-Wilk test"""
        print("\n" + "="*60)
        print("Normality Test (Shapiro-Wilk)")
        print("="*60)
        
        results = {}
        for metric in self.metrics:
            print(f"\n{metric}:")
            results[metric] = {}
            
            for group_name in self.groups.keys():
                data = self.data[group_name][metric]
                if len(data) >= 3:  # Need at least 3 samples
                    stat, p_value = shapiro(data)
                    results[metric][group_name] = p_value
                    normality = "Normal" if p_value > 0.05 else "Not Normal"
                    print(f"  {group_name:10s}: p={p_value:.4f} ({normality})")
                else:
                    print(f"  {group_name:10s}: Insufficient data")
        
        return results
    
    def check_homogeneity(self):
        """Check homogeneity of variance using Levene's test"""
        print("\n" + "="*60)
        print("Homogeneity of Variance Test (Levene)")
        print("="*60)
        
        results = {}
        for metric in self.metrics:
            groups_data = [self.data[group][metric] for group in self.groups.keys()]
            
            # Filter out empty groups
            groups_data = [g for g in groups_data if len(g) >= 2]
            
            if len(groups_data) >= 2:
                stat, p_value = levene(*groups_data)
                results[metric] = p_value
                homogeneity = "Equal variance" if p_value > 0.05 else "Unequal variance"
                print(f"{metric:20s}: p={p_value:.4f} ({homogeneity})")
            else:
                print(f"{metric:20s}: Insufficient data")
        
        return results
    
    def perform_anova(self):
        """Perform one-way ANOVA or Kruskal-Wallis test"""
        print("\n" + "="*60)
        print("Statistical Tests (ANOVA / Kruskal-Wallis)")
        print("="*60)
        
        results = {}
        for metric in self.metrics:
            groups_data = [self.data[group][metric] for group in self.groups.keys()]
            
            # Filter out empty groups
            groups_data = [g for g in groups_data if len(g) >= 2]
            
            if len(groups_data) >= 2:
                # Try ANOVA
                f_stat, anova_p = f_oneway(*groups_data)
                
                # Try Kruskal-Wallis (non-parametric alternative)
                h_stat, kw_p = kruskal(*groups_data)
                
                results[metric] = {
                    'anova': {'statistic': f_stat, 'p_value': anova_p},
                    'kruskal_wallis': {'statistic': h_stat, 'p_value': kw_p}
                }
                
                print(f"\n{metric}:")
                print(f"  ANOVA:          F={f_stat:.4f}, p={anova_p:.4f} {'***' if anova_p < 0.001 else '**' if anova_p < 0.01 else '*' if anova_p < 0.05 else 'ns'}")
                print(f"  Kruskal-Wallis: H={h_stat:.4f}, p={kw_p:.4f} {'***' if kw_p < 0.001 else '**' if kw_p < 0.01 else '*' if kw_p < 0.05 else 'ns'}")
            else:
                print(f"\n{metric}: Insufficient data for testing")
        
        return results
    
    def perform_posthoc(self):
        """Perform pairwise Mann-Whitney U tests (post-hoc)"""
        print("\n" + "="*60)
        print("Post-hoc Tests (Mann-Whitney U, Bonferroni corrected)")
        print("="*60)
        
        results = {}
        group_names = list(self.groups.keys())
        n_comparisons = len(group_names) * (len(group_names) - 1) // 2
        bonferroni_alpha = 0.05 / n_comparisons
        
        print(f"Bonferroni corrected alpha: {bonferroni_alpha:.4f}")
        
        for metric in self.metrics:
            print(f"\n{metric}:")
            results[metric] = {}
            
            for i in range(len(group_names)):
                for j in range(i + 1, len(group_names)):
                    group1 = group_names[i]
                    group2 = group_names[j]
                    
                    data1 = self.data[group1][metric]
                    data2 = self.data[group2][metric]
                    
                    if len(data1) >= 2 and len(data2) >= 2:
                        stat, p_value = mannwhitneyu(data1, data2, alternative='two-sided')
                        
                        # Calculate effect size (rank-biserial correlation)
                        n1, n2 = len(data1), len(data2)
                        r = 1 - (2*stat) / (n1 * n2)  # rank-biserial correlation
                        
                        results[metric][f"{group1}_vs_{group2}"] = {
                            'statistic': stat,
                            'p_value': p_value,
                            'effect_size': r
                        }
                        
                        significance = '***' if p_value < bonferroni_alpha else 'ns'
                        print(f"  {group1:10s} vs {group2:10s}: U={stat:.2f}, p={p_value:.4f}, r={r:.3f} {significance}")
        
        return results
    
    def calculate_descriptive_stats(self):
        """Calculate descriptive statistics"""
        print("\n" + "="*60)
        print("Descriptive Statistics")
        print("="*60)
        
        stats_dict = {}
        
        for metric in self.metrics:
            print(f"\n{metric}:")
            print(f"{'Group':10s} {'N':>5s} {'Mean':>10s} {'SD':>10s} {'Median':>10s} {'IQR':>10s}")
            print("-" * 60)
            
            stats_dict[metric] = {}
            
            for group_name in self.groups.keys():
                data = np.array(self.data[group_name][metric])
                
                if len(data) > 0:
                    n = len(data)
                    mean = np.mean(data)
                    sd = np.std(data, ddof=1)
                    median = np.median(data)
                    q1, q3 = np.percentile(data, [25, 75])
                    iqr = q3 - q1
                    
                    stats_dict[metric][group_name] = {
                        'n': n,
                        'mean': mean,
                        'sd': sd,
                        'median': median,
                        'iqr': iqr
                    }
                    
                    print(f"{group_name:10s} {n:5d} {mean:10.4f} {sd:10.4f} {median:10.4f} {iqr:10.4f}")
        
        return stats_dict
    
    def visualize_results(self, save_dir=None):
        """Create visualization of results"""
        print("\nGenerating visualizations...")
        
        # Create figure with subplots
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        axes = axes.flatten()
        
        colors = {'Slow': '#FF6B6B', 'Medium': '#4ECDC4', 'Fast': '#45B7D1'}
        
        for idx, metric in enumerate(self.metrics):
            ax = axes[idx]
            
            # Prepare data for plotting
            plot_data = []
            plot_labels = []
            
            for group_name in self.groups.keys():
                data = self.data[group_name][metric]
                if len(data) > 0:
                    plot_data.extend(data)
                    plot_labels.extend([group_name] * len(data))
            
            if plot_data:
                # Create DataFrame
                df = pd.DataFrame({'Group': plot_labels, 'Value': plot_data})
                
                # Box plot with individual points
                sns.boxplot(data=df, x='Group', y='Value', ax=ax, 
                           order=['Slow', 'Medium', 'Fast'],
                           palette=colors, width=0.5)
                sns.stripplot(data=df, x='Group', y='Value', ax=ax,
                            order=['Slow', 'Medium', 'Fast'],
                            color='black', alpha=0.5, size=4)
                
                # Format
                ax.set_title(metric.replace('_', ' ').title(), fontsize=14, fontweight='bold')
                ax.set_xlabel('Group', fontsize=12)
                ax.set_ylabel('Value', fontsize=12)
                ax.grid(axis='y', alpha=0.3)
                
                # # Add sample size
                # for i, group in enumerate(['Slow', 'Medium', 'Fast']):
                #     n = len(self.data[group][metric])
                #     ax.text(i, ax.get_ylim()[0], f'n={n}', 
                #            ha='center', va='top', fontsize=10)
        
        plt.tight_layout()
        
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
            save_path = os.path.join(save_dir, 'metrics_comparison.png')
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"  ✓ Figure saved to {save_path}")
        
        plt.show()
    
    def generate_report(self, save_dir=None):
        """Generate comprehensive statistical report"""
        print("\n" + "="*80)
        print(" " * 20 + "COMPREHENSIVE STATISTICAL REPORT")
        print("="*80)
        
        # Load data
        self.load_data()
        
        # Descriptive statistics
        desc_stats = self.calculate_descriptive_stats()
        
        # Check assumptions
        normality_results = self.check_normality()
        homogeneity_results = self.check_homogeneity()
        
        # Perform tests
        anova_results = self.perform_anova()
        posthoc_results = self.perform_posthoc()
        
        # Visualize
        if save_dir is None:
            save_dir = os.path.join(self.data_base_dir, '..', 'statistical_results')
        
        self.visualize_results(save_dir=save_dir)
        
        # Save results to file
        self._save_results_to_file(desc_stats, anova_results, posthoc_results, save_dir)
        
        print("\n" + "="*80)
        print("Report generation complete!")
        print("="*80)
    
    def _save_results_to_file(self, desc_stats, anova_results, posthoc_results, save_dir):
        """Save statistical results to text file"""
        os.makedirs(save_dir, exist_ok=True)
        report_path = os.path.join(save_dir, 'statistical_report.txt')
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("="*80 + "\n")
            f.write(" " * 20 + "STATISTICAL ANALYSIS REPORT\n")
            f.write("="*80 + "\n\n")
            
            # Groups info
            f.write("Groups:\n")
            f.write("  Slow group:   data 45-54\n")
            f.write("  Medium group: data 0-29\n")
            f.write("  Fast group:   data 55-64\n\n")
            
            # Descriptive statistics
            f.write("="*80 + "\n")
            f.write("DESCRIPTIVE STATISTICS\n")
            f.write("="*80 + "\n\n")
            
            for metric in self.metrics:
                f.write(f"{metric}:\n")
                f.write(f"{'Group':10s} {'N':>5s} {'Mean':>10s} {'SD':>10s} {'Median':>10s} {'IQR':>10s}\n")
                f.write("-" * 60 + "\n")
                
                for group in ['Slow', 'Medium', 'Fast']:
                    if group in desc_stats[metric]:
                        stats = desc_stats[metric][group]
                        f.write(f"{group:10s} {stats['n']:5d} {stats['mean']:10.4f} "
                               f"{stats['sd']:10.4f} {stats['median']:10.4f} {stats['iqr']:10.4f}\n")
                f.write("\n")
            
            # ANOVA results
            f.write("="*80 + "\n")
            f.write("STATISTICAL TESTS\n")
            f.write("="*80 + "\n\n")
            
            for metric in self.metrics:
                if metric in anova_results:
                    f.write(f"{metric}:\n")
                    anova = anova_results[metric]['anova']
                    kw = anova_results[metric]['kruskal_wallis']
                    
                    f.write(f"  ANOVA:          F={anova['statistic']:.4f}, p={anova['p_value']:.4f}\n")
                    f.write(f"  Kruskal-Wallis: H={kw['statistic']:.4f}, p={kw['p_value']:.4f}\n")
                    
                    # Significance stars
                    p = anova['p_value']
                    if p < 0.001:
                        f.write("  Significance: *** (p < 0.001)\n")
                    elif p < 0.01:
                        f.write("  Significance: ** (p < 0.01)\n")
                    elif p < 0.05:
                        f.write("  Significance: * (p < 0.05)\n")
                    else:
                        f.write("  Significance: ns (not significant)\n")
                    f.write("\n")
            
            # Post-hoc results
            f.write("="*80 + "\n")
            f.write("POST-HOC TESTS (Mann-Whitney U, Bonferroni corrected)\n")
            f.write("="*80 + "\n\n")
            
            for metric in self.metrics:
                if metric in posthoc_results:
                    f.write(f"{metric}:\n")
                    
                    for comparison, result in posthoc_results[metric].items():
                        groups = comparison.replace('_vs_', ' vs ')
                        f.write(f"  {groups:25s}: U={result['statistic']:.2f}, "
                               f"p={result['p_value']:.4f}, r={result['effect_size']:.3f}\n")
                    f.write("\n")
            
            f.write("="*80 + "\n")
            f.write("Legend:\n")
            f.write("  *** : p < 0.001 (highly significant)\n")
            f.write("  **  : p < 0.01  (very significant)\n")
            f.write("  *   : p < 0.05  (significant)\n")
            f.write("  ns  : not significant\n")
            f.write("  r   : effect size (rank-biserial correlation)\n")
            f.write("="*80 + "\n")
        
        print(f"  ✓ Report saved to {report_path}")


def compare_curvature_phi_distribution(data_base_dir=None, save_fig=True):
    """
    Compare curvature and phi distributions between Slow and Fast groups.
    
    Parameters:
    -----------
    data_base_dir : str, optional
        Base directory containing data folders. If None, auto-detect.
    save_fig : bool, default True
        Whether to save the figure.
        
    Returns:
    --------
    dict : Statistical test results
    """
    from scipy.stats import ks_2samp, mannwhitneyu
    
    # Get data directory
    if data_base_dir is None:
        current_file_path = os.path.abspath(__file__)
        current_dir = os.path.dirname(current_file_path)
        project_dir = os.path.dirname(current_dir)
        data_base_dir = os.path.join(project_dir, 'data')
    
    # Define groups
    slow_ids = list(range(45, 55))   # 45-54
    fast_ids = list(range(55, 65))   # 55-64
    
    def load_group_distributions(data_ids, group_name):
        """Load curvature and phi distributions for a group"""
        all_curvature = []
        all_log_curvature = []
        all_phi = []
        all_log_phi = []
        
        print(f"\nLoading {group_name} group data...")
        
        for data_id in data_ids:
            data_dir = os.path.join(data_base_dir, f'{data_id}_data_12-01')
            position_path = os.path.join(data_dir, 'Lpsm_position.npy')
            
            if not os.path.exists(position_path):
                print(f"  Warning: {position_path} not found, skipping...")
                continue
            
            try:
                positions = np.load(position_path, allow_pickle=True)
                
                if positions.shape[1] >= 4:
                    pos = positions[:, :3]
                    timestamps = positions[:, 3]
                    
                    # Convert timestamps
                    if timestamps[0] > 1e9:
                        timestamps = timestamps - timestamps[0]
                        if timestamps[-1] > 1e6:
                            timestamps = timestamps / 1e6
                    else:
                        timestamps = timestamps - timestamps[0]
                    
                    data = np.column_stack([pos, timestamps])
                    result = calculate_distribution(data, return_stats=False)
                    
                    # Filter valid values
                    valid_curv = result['curvature'][np.isfinite(result['curvature'])]
                    valid_log_curv = result['log_curvature'][np.isfinite(result['log_curvature'])]
                    valid_phi = result['phi'][np.isfinite(result['phi'])]
                    valid_log_phi = result['log_phi'][np.isfinite(result['log_phi'])]
                    
                    all_curvature.extend(valid_curv)
                    all_log_curvature.extend(valid_log_curv)
                    all_phi.extend(valid_phi)
                    all_log_phi.extend(valid_log_phi)
                    
                    print(f"  ✓ Data {data_id}: {len(valid_curv)} points")
                    
            except Exception as e:
                print(f"  Error loading data {data_id}: {e}")
        
        return {
            'curvature': np.array(all_curvature),
            'log_curvature': np.array(all_log_curvature),
            'phi': np.array(all_phi),
            'log_phi': np.array(all_log_phi)
        }
    
    # Load data for both groups
    slow_data = load_group_distributions(slow_ids, "Slow")
    fast_data = load_group_distributions(fast_ids, "Fast")
    
    # Statistical tests
    print("\n" + "="*80)
    print("Statistical Tests: Slow vs Fast Group Distribution Comparison")
    print("="*80)
    
    test_results = {}
    metrics = ['curvature', 'log_curvature', 'phi', 'log_phi']
    metric_names = ['Curvature (κ)', 'Log Curvature (G)', 'Phi (φ)', 'Log Phi (S)']
    
    for metric, name in zip(metrics, metric_names):
        slow_arr = slow_data[metric]
        fast_arr = fast_data[metric]
        
        # K-S test
        ks_stat, ks_p = ks_2samp(slow_arr, fast_arr)
        
        # Mann-Whitney U test
        u_stat, mw_p = mannwhitneyu(slow_arr, fast_arr, alternative='two-sided')
        
        # Effect size (rank-biserial correlation)
        n1, n2 = len(slow_arr), len(fast_arr)
        r = 1 - (2 * u_stat) / (n1 * n2)
        
        test_results[metric] = {
            'ks_stat': ks_stat,
            'ks_p': ks_p,
            'mw_u': u_stat,
            'mw_p': mw_p,
            'effect_size': r,
            'slow_median': np.median(slow_arr),
            'fast_median': np.median(fast_arr),
            'slow_mean': np.mean(slow_arr),
            'fast_mean': np.mean(fast_arr),
            'slow_n': n1,
            'fast_n': n2
        }
        
        print(f"\n{name}:")
        print(f"  Slow:  n={n1:5d}, Mean={np.mean(slow_arr):12.4f}, Median={np.median(slow_arr):12.4f}")
        print(f"  Fast:  n={n2:5d}, Mean={np.mean(fast_arr):12.4f}, Median={np.median(fast_arr):12.4f}")
        print(f"  K-S test:        D={ks_stat:.4f}, p={ks_p:.4e} {'***' if ks_p < 0.001 else '**' if ks_p < 0.01 else '*' if ks_p < 0.05 else 'ns'}")
        print(f"  Mann-Whitney U:  U={u_stat:.2f}, p={mw_p:.4e}, r={r:.4f} {'***' if mw_p < 0.001 else '**' if mw_p < 0.01 else '*' if mw_p < 0.05 else 'ns'}")
    
    # Visualization
    print("\nGenerating visualization...")
    
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    
    colors = {'Slow': '#FF6B6B', 'Fast': '#45B7D1'}
    
    for idx, (metric, name) in enumerate(zip(metrics, metric_names)):
        slow_arr = slow_data[metric]
        fast_arr = fast_data[metric]
        
        # Row 1: Overlaid histograms with KDE
        ax = axes[0, idx]
        
        # Determine common bins
        all_data = np.concatenate([slow_arr, fast_arr])
        bins = np.histogram_bin_edges(all_data, bins=50)
        
        ax.hist(slow_arr, bins=bins, alpha=0.5, color=colors['Slow'], 
                label=f'Slow (n={len(slow_arr)})', density=True, edgecolor='white')
        ax.hist(fast_arr, bins=bins, alpha=0.5, color=colors['Fast'], 
                label=f'Fast (n={len(fast_arr)})', density=True, edgecolor='white')
        
        # Add KDE
        from scipy.stats import gaussian_kde
        if len(slow_arr) > 10 and len(fast_arr) > 10:
            x_range = np.linspace(np.min(all_data), np.max(all_data), 200)
            try:
                kde_slow = gaussian_kde(slow_arr)
                kde_fast = gaussian_kde(fast_arr)
                ax.plot(x_range, kde_slow(x_range), color=colors['Slow'], linewidth=2, linestyle='-')
                ax.plot(x_range, kde_fast(x_range), color=colors['Fast'], linewidth=2, linestyle='-')
            except:
                pass
        
        # Add median lines
        ax.axvline(np.median(slow_arr), color=colors['Slow'], linestyle='--', linewidth=2, alpha=0.8)
        ax.axvline(np.median(fast_arr), color=colors['Fast'], linestyle='--', linewidth=2, alpha=0.8)
        
        ax.set_xlabel(name, fontsize=11)
        ax.set_ylabel('Density', fontsize=11)
        ax.set_title(f'{name} Distribution', fontsize=12, fontweight='bold')
        ax.legend(loc='upper right', fontsize=9)
        ax.grid(alpha=0.3)
        
        # Add statistical annotation
        res = test_results[metric]
        sig = '***' if res['ks_p'] < 0.001 else '**' if res['ks_p'] < 0.01 else '*' if res['ks_p'] < 0.05 else 'ns'
        ax.text(0.02, 0.98, f"K-S: D={res['ks_stat']:.3f}, p={res['ks_p']:.2e} {sig}",
                transform=ax.transAxes, fontsize=9, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        # Row 2: Box plots
        ax = axes[1, idx]
        
        # Prepare data for boxplot
        df = pd.DataFrame({
            'Group': ['Slow'] * len(slow_arr) + ['Fast'] * len(fast_arr),
            'Value': np.concatenate([slow_arr, fast_arr])
        })
        
        # Sample for visualization if too many points
        if len(df) > 5000:
            df_sample = df.sample(n=5000, random_state=42)
        else:
            df_sample = df
        
        sns.boxplot(data=df_sample, x='Group', y='Value', ax=ax, 
                   order=['Slow', 'Fast'], palette=colors, width=0.5)
        sns.stripplot(data=df_sample.sample(n=min(500, len(df_sample)), random_state=42), 
                     x='Group', y='Value', ax=ax,
                     order=['Slow', 'Fast'], color='black', alpha=0.3, size=2)
        
        ax.set_xlabel('Group', fontsize=11)
        ax.set_ylabel(name, fontsize=11)
        ax.set_title(f'{name} Box Plot', fontsize=12, fontweight='bold')
        ax.grid(axis='y', alpha=0.3)
        
        # Add Mann-Whitney annotation
        sig = '***' if res['mw_p'] < 0.001 else '**' if res['mw_p'] < 0.01 else '*' if res['mw_p'] < 0.05 else 'ns'
        ax.text(0.5, 0.98, f"M-W: p={res['mw_p']:.2e}, r={res['effect_size']:.3f} {sig}",
                transform=ax.transAxes, fontsize=9, verticalalignment='top', ha='center',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    plt.suptitle('Curvature and Phi Distribution Comparison: Slow vs Fast Groups', 
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    if save_fig:
        save_dir = os.path.join(data_base_dir, '..', 'statistical_results')
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, 'curvature_phi_comparison.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"  ✓ Figure saved to {save_path}")
        
        # Save statistics to file
        report_path = os.path.join(save_dir, 'curvature_phi_statistics.txt')
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("="*80 + "\n")
            f.write("CURVATURE AND PHI DISTRIBUTION COMPARISON: SLOW VS FAST GROUPS\n")
            f.write("="*80 + "\n\n")
            
            f.write("Groups:\n")
            f.write("  Slow group: data 45-54\n")
            f.write("  Fast group: data 55-64\n\n")
            
            for metric, name in zip(metrics, metric_names):
                res = test_results[metric]
                f.write(f"\n{name}:\n")
                f.write("-"*60 + "\n")
                f.write(f"  Slow:  n={res['slow_n']:5d}, Mean={res['slow_mean']:12.4f}, Median={res['slow_median']:12.4f}\n")
                f.write(f"  Fast:  n={res['fast_n']:5d}, Mean={res['fast_mean']:12.4f}, Median={res['fast_median']:12.4f}\n")
                f.write(f"  K-S test:        D={res['ks_stat']:.4f}, p={res['ks_p']:.4e}\n")
                f.write(f"  Mann-Whitney U:  U={res['mw_u']:.2f}, p={res['mw_p']:.4e}, r={res['effect_size']:.4f}\n")
                
                sig = '***' if res['ks_p'] < 0.001 else '**' if res['ks_p'] < 0.01 else '*' if res['ks_p'] < 0.05 else 'ns'
                f.write(f"  Significance: {sig}\n")
            
            f.write("\n" + "="*80 + "\n")
            f.write("Legend:\n")
            f.write("  *** : p < 0.001 (highly significant)\n")
            f.write("  **  : p < 0.01  (very significant)\n")
            f.write("  *   : p < 0.05  (significant)\n")
            f.write("  ns  : not significant\n")
            f.write("  r   : effect size (rank-biserial correlation)\n")
            f.write("="*80 + "\n")
        
        print(f"  ✓ Report saved to {report_path}")
    
    plt.show()
    
    return test_results


def visualize_curvature_phi_over_time(data_base_dir=None, save_fig=True):
    """
    Visualize curvature and phi changes over time steps for Slow and Fast groups.
    
    Parameters:
    -----------
    data_base_dir : str, optional
        Base directory containing data folders. If None, auto-detect.
    save_fig : bool, default True
        Whether to save the figure.
    """
    from scipy.ndimage import uniform_filter1d
    
    # Get data directory
    if data_base_dir is None:
        current_file_path = os.path.abspath(__file__)
        current_dir = os.path.dirname(current_file_path)
        project_dir = os.path.dirname(current_dir)
        data_base_dir = os.path.join(project_dir, 'data')
    
    # Define groups
    slow_ids = list(range(45, 55))   # 45-54
    fast_ids = list(range(55, 65))   # 55-64
    
    def load_single_trajectory(data_id):
        """Load curvature and phi time series for a single trajectory"""
        data_dir = os.path.join(data_base_dir, f'{data_id}_data_12-01')
        position_path = os.path.join(data_dir, 'Rpsm_position.npy')
        
        if not os.path.exists(position_path):
            return None
        
        try:
            positions = np.load(position_path, allow_pickle=True)
            
            if positions.shape[1] >= 4:
                pos = positions[:, :3]
                timestamps = positions[:, 3]
                
                # Convert timestamps
                if timestamps[0] > 1e9:
                    timestamps = timestamps - timestamps[0]
                    if timestamps[-1] > 1e6:
                        timestamps = timestamps / 1e6
                else:
                    timestamps = timestamps - timestamps[0]
                
                data = np.column_stack([pos, timestamps])
                result = calculate_distribution(data, return_stats=False)
                
                return {
                    'curvature': result['curvature'],
                    'log_curvature': result['log_curvature'],
                    'phi': result['phi'],
                    'log_phi': result['log_phi'],
                    'time': result['time'],
                    'n_points': len(result['curvature'])
                }
        except Exception as e:
            print(f"  Error loading data {data_id}: {e}")
        
        return None
    
    # Load all trajectories
    print("Loading trajectories...")
    slow_trajs = []
    fast_trajs = []
    
    for data_id in slow_ids:
        traj = load_single_trajectory(data_id)
        if traj is not None:
            slow_trajs.append(traj)
            print(f"  Slow {data_id}: {traj['n_points']} points")
    
    for data_id in fast_ids:
        traj = load_single_trajectory(data_id)
        if traj is not None:
            fast_trajs.append(traj)
            print(f"  Fast {data_id}: {traj['n_points']} points")
    
    # Normalize time to percentage (0-100%)
    def normalize_to_percentage(trajs):
        """Normalize each trajectory to percentage of completion"""
        normalized = []
        for traj in trajs:
            n = traj['n_points']
            pct = np.linspace(0, 100, n)
            normalized.append({
                'pct': pct,
                'curvature': traj['curvature'],
                'log_curvature': traj['log_curvature'],
                'phi': traj['phi'],
                'log_phi': traj['log_phi']
            })
        return normalized
    
    slow_norm = normalize_to_percentage(slow_trajs)
    fast_norm = normalize_to_percentage(fast_trajs)
    
    # Interpolate to common time grid for averaging
    common_pct = np.linspace(0, 100, 200)
    
    def interpolate_to_common(trajs_norm, metric):
        """Interpolate all trajectories to common percentage grid"""
        interpolated = []
        for traj in trajs_norm:
            # Filter out non-finite values
            valid_mask = np.isfinite(traj[metric])
            if np.sum(valid_mask) > 10:
                pct_valid = traj['pct'][valid_mask]
                metric_valid = traj[metric][valid_mask]
                interp_values = np.interp(common_pct, pct_valid, metric_valid)
                interpolated.append(interp_values)
        return np.array(interpolated)
    
    # Create visualization
    print("\nGenerating time-series visualization...")
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    colors = {'Slow': '#FF6B6B', 'Fast': '#45B7D1'}
    metrics = ['log_curvature', 'log_phi']
    metric_names = ['Log Curvature (G)', 'Log Phi (S)']
    
    for col, (metric, name) in enumerate(zip(metrics, metric_names)):
        # Get interpolated data
        slow_interp = interpolate_to_common(slow_norm, metric)
        fast_interp = interpolate_to_common(fast_norm, metric)
        
        # Row 1: Individual trajectories
        ax = axes[0, col]
        
        # Plot individual trajectories with transparency
        for i, traj in enumerate(slow_norm):
            valid_mask = np.isfinite(traj[metric])
            ax.plot(traj['pct'][valid_mask], traj[metric][valid_mask], 
                   color=colors['Slow'], alpha=0.3, linewidth=0.8,
                   label='Slow' if i == 0 else '')
        
        for i, traj in enumerate(fast_norm):
            valid_mask = np.isfinite(traj[metric])
            ax.plot(traj['pct'][valid_mask], traj[metric][valid_mask], 
                   color=colors['Fast'], alpha=0.3, linewidth=0.8,
                   label='Fast' if i == 0 else '')
        
        ax.set_xlabel('Task Completion (%)', fontsize=11)
        ax.set_ylabel(name, fontsize=11)
        ax.set_title(f'{name} - Individual Trajectories', fontsize=12, fontweight='bold')
        ax.legend(loc='upper right')
        ax.grid(alpha=0.3)
        ax.set_xlim(0, 100)
        
        # Row 2: Mean with confidence interval
        ax = axes[1, col]
        
        if len(slow_interp) > 0:
            slow_mean = np.mean(slow_interp, axis=0)
            slow_std = np.std(slow_interp, axis=0)
            
            # Smooth the curves
            smooth_window = 10
            slow_mean_smooth = uniform_filter1d(slow_mean, smooth_window)
            slow_std_smooth = uniform_filter1d(slow_std, smooth_window)
            
            ax.plot(common_pct, slow_mean_smooth, color=colors['Slow'], 
                   linewidth=2.5, label=f'Slow (n={len(slow_interp)})')
            ax.fill_between(common_pct, 
                           slow_mean_smooth - slow_std_smooth,
                           slow_mean_smooth + slow_std_smooth,
                           color=colors['Slow'], alpha=0.2)
        
        if len(fast_interp) > 0:
            fast_mean = np.mean(fast_interp, axis=0)
            fast_std = np.std(fast_interp, axis=0)
            
            # Smooth the curves
            fast_mean_smooth = uniform_filter1d(fast_mean, smooth_window)
            fast_std_smooth = uniform_filter1d(fast_std, smooth_window)
            
            ax.plot(common_pct, fast_mean_smooth, color=colors['Fast'], 
                   linewidth=2.5, label=f'Fast (n={len(fast_interp)})')
            ax.fill_between(common_pct, 
                           fast_mean_smooth - fast_std_smooth,
                           fast_mean_smooth + fast_std_smooth,
                           color=colors['Fast'], alpha=0.2)
        
        ax.set_xlabel('Task Completion (%)', fontsize=11)
        ax.set_ylabel(name, fontsize=11)
        ax.set_title(f'{name} - Mean ± Std', fontsize=12, fontweight='bold')
        ax.legend(loc='upper right')
        ax.grid(alpha=0.3)
        ax.set_xlim(0, 100)
    
    plt.suptitle('Curvature and Phi Over Time: Slow vs Fast Groups', 
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    if save_fig:
        save_dir = os.path.join(data_base_dir, '..', 'statistical_results')
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, 'curvature_phi_over_time.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"  ✓ Figure saved to {save_path}")
    
    plt.show()
    
    # Create additional figure: Raw curvature and phi (not log)
    fig2, axes2 = plt.subplots(2, 2, figsize=(16, 12))
    
    raw_metrics = ['curvature', 'phi']
    raw_names = ['Curvature (κ)', 'Phi (φ)']
    
    for col, (metric, name) in enumerate(zip(raw_metrics, raw_names)):
        # Get interpolated data
        slow_interp = interpolate_to_common(slow_norm, metric)
        fast_interp = interpolate_to_common(fast_norm, metric)
        
        # Apply log transform for better visualization (raw values vary too much)
        slow_interp_log = np.log10(slow_interp + 1e-10)
        fast_interp_log = np.log10(fast_interp + 1e-10)
        
        # Row 1: Individual trajectories (log scale)
        ax = axes2[0, col]
        
        for i, traj in enumerate(slow_norm):
            valid_mask = np.isfinite(traj[metric]) & (traj[metric] > 0)
            if np.sum(valid_mask) > 10:
                ax.plot(traj['pct'][valid_mask], np.log10(traj[metric][valid_mask] + 1e-10), 
                       color=colors['Slow'], alpha=0.3, linewidth=0.8,
                       label='Slow' if i == 0 else '')
        
        for i, traj in enumerate(fast_norm):
            valid_mask = np.isfinite(traj[metric]) & (traj[metric] > 0)
            if np.sum(valid_mask) > 10:
                ax.plot(traj['pct'][valid_mask], np.log10(traj[metric][valid_mask] + 1e-10), 
                       color=colors['Fast'], alpha=0.3, linewidth=0.8,
                       label='Fast' if i == 0 else '')
        
        ax.set_xlabel('Task Completion (%)', fontsize=11)
        ax.set_ylabel(f'log₁₀({name})', fontsize=11)
        ax.set_title(f'{name} - Individual Trajectories (Log Scale)', fontsize=12, fontweight='bold')
        ax.legend(loc='upper right')
        ax.grid(alpha=0.3)
        ax.set_xlim(0, 100)
        
        # Row 2: Mean with confidence interval
        ax = axes2[1, col]
        
        if len(slow_interp_log) > 0:
            slow_mean = np.nanmean(slow_interp_log, axis=0)
            slow_std = np.nanstd(slow_interp_log, axis=0)
            
            # Smooth
            slow_mean_smooth = uniform_filter1d(slow_mean, smooth_window)
            slow_std_smooth = uniform_filter1d(slow_std, smooth_window)
            
            ax.plot(common_pct, slow_mean_smooth, color=colors['Slow'], 
                   linewidth=2.5, label=f'Slow (n={len(slow_interp)})')
            ax.fill_between(common_pct, 
                           slow_mean_smooth - slow_std_smooth,
                           slow_mean_smooth + slow_std_smooth,
                           color=colors['Slow'], alpha=0.2)
        
        if len(fast_interp_log) > 0:
            fast_mean = np.nanmean(fast_interp_log, axis=0)
            fast_std = np.nanstd(fast_interp_log, axis=0)
            
            # Smooth
            fast_mean_smooth = uniform_filter1d(fast_mean, smooth_window)
            fast_std_smooth = uniform_filter1d(fast_std, smooth_window)
            
            ax.plot(common_pct, fast_mean_smooth, color=colors['Fast'], 
                   linewidth=2.5, label=f'Fast (n={len(fast_interp)})')
            ax.fill_between(common_pct, 
                           fast_mean_smooth - fast_std_smooth,
                           fast_mean_smooth + fast_std_smooth,
                           color=colors['Fast'], alpha=0.2)
        
        ax.set_xlabel('Task Completion (%)', fontsize=11)
        ax.set_ylabel(f'log₁₀({name})', fontsize=11)
        ax.set_title(f'{name} - Mean ± Std (Log Scale)', fontsize=12, fontweight='bold')
        ax.legend(loc='upper right')
        ax.grid(alpha=0.3)
        ax.set_xlim(0, 100)
    
    plt.suptitle('Raw Curvature and Phi Over Time (Log Scale): Slow vs Fast Groups', 
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    if save_fig:
        save_path2 = os.path.join(save_dir, 'curvature_phi_over_time_raw.png')
        plt.savefig(save_path2, dpi=300, bbox_inches='tight')
        print(f"  ✓ Figure saved to {save_path2}")
    
    plt.show()
    
    print("\nVisualization complete!")


def main():
    """Main function"""
    # Get data directory
    current_file_path = os.path.abspath(__file__)
    current_dir = os.path.dirname(current_file_path)
    project_dir = os.path.dirname(current_dir)
    data_base_dir = os.path.join(project_dir, 'data')
    
    print("Statistical Analysis for Speed Groups")
    print("="*80)
    print(f"Data directory: {data_base_dir}")
    print("="*80)
    
    # Create analyzer
    analyzer = MetricsStatistics(data_base_dir)
    
    # Generate report
    analyzer.generate_report()


if __name__ == "__main__":
    main()
    
    # Also run curvature/phi comparison
    print("\n\n")
    compare_curvature_phi_distribution()
    
    # Visualize over time
    print("\n\n")
    visualize_curvature_phi_over_time()

