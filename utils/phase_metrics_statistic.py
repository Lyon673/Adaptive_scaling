"""
Phase-wise Gracefulness and Smoothness Analysis
Compare Slow vs Fast groups across different task phases
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import mannwhitneyu, kruskal
import pandas as pd
import os
import sys

# Add parent directory to path
current_file_path = os.path.abspath(__file__)
current_dir = os.path.dirname(current_file_path)
project_dir = os.path.dirname(current_dir)
sys.path.insert(0, project_dir)
sys.path.insert(0, os.path.join(project_dir, 'Trajectory'))

from gracefulness import calculate_distribution, calculate_gracefulness, calculate_smoothness
from Trajectory.load_data import generate_frame_label_map, needle_dir_path
from smoothness import sparc, log_dimensionless_jerk


class PhaseMetricsAnalyzer:
    """Analyze gracefulness and smoothness across different task phases"""
    
    def __init__(self, data_base_dir=None):
        if data_base_dir is None:
            self.data_base_dir = os.path.join(project_dir, 'data')
        else:
            self.data_base_dir = data_base_dir
        
        # Define groups - use demo IDs directly (not data folder IDs)
        # Slow group: data 45-54, Fast group: data 55-64
        self.groups = {
            'Slow': list(range(46, 55)),
            'Fast': list(range(56, 65))
        }
        
        # Phase names (based on typical surgical task phases)
        self.phase_names = {
            -1: 'Unlabeled',
            0: 'Phase 0',
            1: 'Phase 1', 
            2: 'Phase 2',
            3: 'Phase 3',
            4: 'Phase 4',
            5: 'Phase 5'
        }
        
        # Data storage
        self.phase_data = {}
        
    def load_trajectory_with_labels(self, data_id, psm_side='L'):
        """
        Load trajectory data and corresponding phase labels for a demo
        
        Args:
            data_id: Demo ID (e.g., 45, 46, ...)
            psm_side: 'L' for left PSM, 'R' for right PSM
            
        Returns:
            dict with trajectory data and labels, or None if failed
        """
        # Load position data from data folder
        data_dir = os.path.join(self.data_base_dir, f'{data_id}_data_12-01')
        position_path = os.path.join(data_dir, f'{psm_side}psm_position.npy')
        
        if not os.path.exists(position_path):
            return None
        
        try:
            positions = np.load(position_path, allow_pickle=True)
            
            if positions.shape[1] < 4:
                return None
            
            pos = positions[:, :3]
            timestamps = positions[:, 3]
            
            # Convert timestamps
            if timestamps[0] > 1e9:
                timestamps = timestamps - timestamps[0]
                if timestamps[-1] > 1e6:
                    timestamps = timestamps / 1e6
            else:
                timestamps = timestamps - timestamps[0]
            
            # Store raw position and time data for phase-wise G/S calculation
            data = np.column_stack([pos, timestamps])
            n_frames = len(data)
            
            # Calculate curvature and phi distribution (for visualization)
            dist_result = calculate_distribution(data, return_stats=False)
            
            # Load phase labels
            try:
                labels = generate_frame_label_map(needle_dir_path, data_id)
                
                # Ensure labels match trajectory length
                if len(labels) != n_frames:
                    # Resample labels to match trajectory length
                    label_indices = np.linspace(0, len(labels)-1, n_frames).astype(int)
                    labels = [labels[i] for i in label_indices]
                
            except Exception as e:
                labels = [-1] * n_frames
            
            return {
                'data': data,  # Raw [x, y, z, time] data for G/S calculation
                'curvature': dist_result['curvature'],
                'log_curvature': dist_result['log_curvature'],
                'phi': dist_result['phi'],
                'log_phi': dist_result['log_phi'],
                'labels': np.array(labels),
                'n_frames': n_frames
            }
            
        except Exception as e:
            return None
    
    def load_all_data(self):
        """Load data for all groups and organize by phase, for both L and R PSM"""
        print("Loading data for phase analysis (Left and Right PSM)...")
        print("Note: G and S are calculated using methods from gracefulness.py")
        print("  G = median(log10(curvature)), where curvature = ||v × a|| / ||v||³")
        print("  S = log10(phi), where phi = (duration⁵/peak_velocity²) × ∫jerk² dt")
        
        # Initialize storage for both L and R
        self.psm_sides = ['L', 'R']
        for psm_side in self.psm_sides:
            self.phase_data[psm_side] = {}
            for group_name in self.groups.keys():
                self.phase_data[psm_side][group_name] = {}
                for phase in range(-1, 6):
                    self.phase_data[psm_side][group_name][phase] = {
                        'curvature': [],
                        'log_curvature': [],
                        'phi': [],
                        'log_phi': [],
                        'G': [],  # Gracefulness: median(log10(curvature))
                        'S': []   # Smoothness: log10(normalized_jerk_integral)
                    }
        
        for psm_side in self.psm_sides:
            psm_name = 'Left' if psm_side == 'L' else 'Right'
            print(f"\n{'='*60}")
            print(f"Processing {psm_name} PSM ({psm_side}psm)")
            print(f"{'='*60}")
            
            for group_name, data_ids in self.groups.items():
                print(f"\n  {group_name} group...")
                
                for data_id in data_ids:
                    result = self.load_trajectory_with_labels(data_id, psm_side=psm_side)
                    
                    if result is None:
                        continue
                    
                    # Organize data by phase
                    unique_phases = np.unique(result['labels'])
                    
                    for phase in unique_phases:
                        phase_mask = result['labels'] == phase
                        
                        if np.sum(phase_mask) < 5:  # Skip phases with too few points
                            continue
                        
                        # Get phase data for distribution visualization
                        phase_curv = result['curvature'][phase_mask]
                        phase_log_curv = result['log_curvature'][phase_mask]
                        phase_phi = result['phi'][phase_mask]
                        phase_log_phi = result['log_phi'][phase_mask]
                        
                        # Filter valid values
                        valid_curv = phase_curv[np.isfinite(phase_curv)]
                        valid_log_curv = phase_log_curv[np.isfinite(phase_log_curv)]
                        valid_phi = phase_phi[np.isfinite(phase_phi)]
                        valid_log_phi = phase_log_phi[np.isfinite(phase_log_phi)]
                        
                        if len(valid_log_curv) > 0 and len(valid_log_phi) > 0:
                            # Store all points for distribution visualization
                            self.phase_data[psm_side][group_name][phase]['curvature'].extend(valid_curv)
                            self.phase_data[psm_side][group_name][phase]['log_curvature'].extend(valid_log_curv)
                            self.phase_data[psm_side][group_name][phase]['phi'].extend(valid_phi)
                            self.phase_data[psm_side][group_name][phase]['log_phi'].extend(valid_log_phi)
                            
                            # Calculate G and S using the exact methods from gracefulness.py
                            # Extract phase-specific raw data [x, y, z, time]
                            phase_data = result['data'][phase_mask]
                            
                            # G: Gracefulness = median(log10(curvature))
                            try:
                                G = calculate_gracefulness(phase_data)
                            except:
                                G = np.median(valid_log_curv)  # fallback
                            
                            # S: Smoothness = log10(phi), where phi is normalized jerk integral
                            try:
                                S = calculate_smoothness(phase_data)
                            except:
                                S = np.median(valid_log_phi)  # fallback
                            
                            self.phase_data[psm_side][group_name][phase]['G'].append(G)
                            self.phase_data[psm_side][group_name][phase]['S'].append(S)
                    
                    print(f"    ✓ Demo {data_id}")
        
        # Print summary
        print("\n" + "="*60)
        print("Data Loading Summary:")
        print("="*60)
        for psm_side in self.psm_sides:
            psm_name = 'Left PSM' if psm_side == 'L' else 'Right PSM'
            print(f"\n{psm_name}:")
            for group_name in self.groups.keys():
                print(f"  {group_name} group:")
                for phase in range(-1, 6):
                    n_demos = len(self.phase_data[psm_side][group_name][phase]['G'])
                    n_points = len(self.phase_data[psm_side][group_name][phase]['log_curvature'])
                    if n_demos > 0:
                        print(f"    Phase {phase}: {n_demos} demos, {n_points} points")
    
    def perform_statistical_tests(self):
        """Perform statistical tests comparing Slow vs Fast for each phase, for both PSMs"""
        print("\n" + "="*80)
        print("Statistical Tests: Slow vs Fast Group by Phase (Left & Right PSM)")
        print("="*80)
        
        results = {}
        
        for psm_side in self.psm_sides:
            psm_name = 'Left PSM' if psm_side == 'L' else 'Right PSM'
            results[psm_side] = {}
            
            print(f"\n{'='*60}")
            print(f"{psm_name}")
            print(f"{'='*60}")
            
            for phase in range(0, 6):  # Only test labeled phases
                phase_name = self.phase_names.get(phase, f'Phase {phase}')
                
                slow_G = np.array(self.phase_data[psm_side]['Slow'][phase]['G'])
                fast_G = np.array(self.phase_data[psm_side]['Fast'][phase]['G'])
                slow_S = np.array(self.phase_data[psm_side]['Slow'][phase]['S'])
                fast_S = np.array(self.phase_data[psm_side]['Fast'][phase]['S'])
                
                if len(slow_G) < 2 or len(fast_G) < 2:
                    print(f"\n{phase_name}: Insufficient data")
                    continue
                
                results[psm_side][phase] = {}
                
                print(f"\n{phase_name}:")
                print("-" * 60)
                
                # Gracefulness (G)
                u_stat_G, p_G = mannwhitneyu(slow_G, fast_G, alternative='two-sided')
                n1, n2 = len(slow_G), len(fast_G)
                r_G = 1 - (2 * u_stat_G) / (n1 * n2)
                
                results[psm_side][phase]['G'] = {
                    'slow_mean': np.mean(slow_G),
                    'slow_std': np.std(slow_G),
                    'fast_mean': np.mean(fast_G),
                    'fast_std': np.std(fast_G),
                    'u_stat': u_stat_G,
                    'p_value': p_G,
                    'effect_size': r_G,
                    'slow_n': n1,
                    'fast_n': n2
                }
                
                sig_G = '***' if p_G < 0.001 else '**' if p_G < 0.01 else '*' if p_G < 0.05 else 'ns'
                print(f"  Gracefulness (G):")
                print(f"    Slow:  n={n1}, Mean={np.mean(slow_G):.4f} ± {np.std(slow_G):.4f}")
                print(f"    Fast:  n={n2}, Mean={np.mean(fast_G):.4f} ± {np.std(fast_G):.4f}")
                print(f"    Mann-Whitney U: U={u_stat_G:.2f}, p={p_G:.4e}, r={r_G:.4f} {sig_G}")
                
                # Smoothness (S)
                u_stat_S, p_S = mannwhitneyu(slow_S, fast_S, alternative='two-sided')
                r_S = 1 - (2 * u_stat_S) / (n1 * n2)
                
                results[psm_side][phase]['S'] = {
                    'slow_mean': np.mean(slow_S),
                    'slow_std': np.std(slow_S),
                    'fast_mean': np.mean(fast_S),
                    'fast_std': np.std(fast_S),
                    'u_stat': u_stat_S,
                    'p_value': p_S,
                    'effect_size': r_S,
                    'slow_n': n1,
                    'fast_n': n2
                }
                
                sig_S = '***' if p_S < 0.001 else '**' if p_S < 0.01 else '*' if p_S < 0.05 else 'ns'
                print(f"  Smoothness (S):")
                print(f"    Slow:  n={n1}, Mean={np.mean(slow_S):.4f} ± {np.std(slow_S):.4f}")
                print(f"    Fast:  n={n2}, Mean={np.mean(fast_S):.4f} ± {np.std(fast_S):.4f}")
                print(f"    Mann-Whitney U: U={u_stat_S:.2f}, p={p_S:.4e}, r={r_S:.4f} {sig_S}")
        
        return results
    
    def visualize_phase_comparison(self, save_fig=True):
        """Create visualization comparing Slow vs Fast across phases for both PSMs"""
        print("\nGenerating phase comparison visualization (Left & Right PSM)...")
        
        colors = {'Slow': '#FF6B6B', 'Fast': '#45B7D1'}
        
        # Create figure with 2 rows (L, R) x 4 columns (G box, S box, G bar, S bar)
        fig, axes = plt.subplots(2, 4, figsize=(24, 12))
        
        for row, psm_side in enumerate(self.psm_sides):
            psm_name = 'Left PSM' if psm_side == 'L' else 'Right PSM'
            
            # Prepare data for plotting
            phases_to_plot = [p for p in range(0, 6) 
                            if len(self.phase_data[psm_side]['Slow'][p]['G']) > 0 
                            and len(self.phase_data[psm_side]['Fast'][p]['G']) > 0]
            
            if not phases_to_plot:
                continue
            
            # Plot 1: Gracefulness (G) by phase - Box plot
            ax = axes[row, 0]
            plot_data = []
            for phase in phases_to_plot:
                for group in ['Slow', 'Fast']:
                    for val in self.phase_data[psm_side][group][phase]['G']:
                        plot_data.append({
                            'Phase': f'P{phase}',
                            'Group': group,
                            'Value': val
                        })
            
            df_G = pd.DataFrame(plot_data)
            sns.boxplot(data=df_G, x='Phase', y='Value', hue='Group', ax=ax, palette=colors)
            ax.set_title(f'{psm_name} - Gracefulness (G)', fontsize=12, fontweight='bold')
            ax.set_ylabel('Log Curvature (G)', fontsize=10)
            ax.set_xlabel('Phase', fontsize=10)
            ax.legend(title='Group', fontsize=8)
            ax.grid(axis='y', alpha=0.3)
            
            # Plot 2: Smoothness (S) by phase - Box plot
            ax = axes[row, 1]
            plot_data = []
            for phase in phases_to_plot:
                for group in ['Slow', 'Fast']:
                    for val in self.phase_data[psm_side][group][phase]['S']:
                        plot_data.append({
                            'Phase': f'P{phase}',
                            'Group': group,
                            'Value': val
                        })
            
            df_S = pd.DataFrame(plot_data)
            sns.boxplot(data=df_S, x='Phase', y='Value', hue='Group', ax=ax, palette=colors)
            ax.set_title(f'{psm_name} - Smoothness (S)', fontsize=12, fontweight='bold')
            ax.set_ylabel('Log Phi (S)', fontsize=10)
            ax.set_xlabel('Phase', fontsize=10)
            ax.legend(title='Group', fontsize=8)
            ax.grid(axis='y', alpha=0.3)
            
            # Plot 3: Mean G with error bars
            ax = axes[row, 2]
            x_positions = np.arange(len(phases_to_plot))
            width = 0.35
            
            slow_means_G = [np.mean(self.phase_data[psm_side]['Slow'][p]['G']) for p in phases_to_plot]
            slow_stds_G = [np.std(self.phase_data[psm_side]['Slow'][p]['G']) for p in phases_to_plot]
            fast_means_G = [np.mean(self.phase_data[psm_side]['Fast'][p]['G']) for p in phases_to_plot]
            fast_stds_G = [np.std(self.phase_data[psm_side]['Fast'][p]['G']) for p in phases_to_plot]
            
            ax.bar(x_positions - width/2, slow_means_G, width, yerr=slow_stds_G,
                          label='Slow', color=colors['Slow'], alpha=0.8, capsize=3)
            ax.bar(x_positions + width/2, fast_means_G, width, yerr=fast_stds_G,
                          label='Fast', color=colors['Fast'], alpha=0.8, capsize=3)
            
            ax.set_xlabel('Phase', fontsize=10)
            ax.set_ylabel('Mean G ± Std', fontsize=10)
            ax.set_title(f'{psm_name} - G Comparison', fontsize=12, fontweight='bold')
            ax.set_xticks(x_positions)
            ax.set_xticklabels([f'P{p}' for p in phases_to_plot])
            ax.legend(fontsize=8)
            ax.grid(axis='y', alpha=0.3)
            
            # Plot 4: Mean S with error bars
            ax = axes[row, 3]
            
            slow_means_S = [np.mean(self.phase_data[psm_side]['Slow'][p]['S']) for p in phases_to_plot]
            slow_stds_S = [np.std(self.phase_data[psm_side]['Slow'][p]['S']) for p in phases_to_plot]
            fast_means_S = [np.mean(self.phase_data[psm_side]['Fast'][p]['S']) for p in phases_to_plot]
            fast_stds_S = [np.std(self.phase_data[psm_side]['Fast'][p]['S']) for p in phases_to_plot]
            
            ax.bar(x_positions - width/2, slow_means_S, width, yerr=slow_stds_S,
                          label='Slow', color=colors['Slow'], alpha=0.8, capsize=3)
            ax.bar(x_positions + width/2, fast_means_S, width, yerr=fast_stds_S,
                          label='Fast', color=colors['Fast'], alpha=0.8, capsize=3)
            
            ax.set_xlabel('Phase', fontsize=10)
            ax.set_ylabel('Mean S ± Std', fontsize=10)
            ax.set_title(f'{psm_name} - S Comparison', fontsize=12, fontweight='bold')
            ax.set_xticks(x_positions)
            ax.set_xticklabels([f'P{p}' for p in phases_to_plot])
            ax.legend(fontsize=8)
            ax.grid(axis='y', alpha=0.3)
        
        plt.suptitle('Phase-wise Gracefulness and Smoothness: Left & Right PSM',
                    fontsize=16, fontweight='bold', y=1.02)
        plt.tight_layout()
        
        if save_fig:
            save_dir = os.path.join(self.data_base_dir, '..', 'statistical_results')
            os.makedirs(save_dir, exist_ok=True)
            save_path = os.path.join(save_dir, 'phase_metrics_comparison_LR.png')
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"  ✓ Figure saved to {save_path}")
        
        plt.show()
        
        return fig
    
    def visualize_distribution_by_phase(self, save_fig=True):
        """Visualize distributions of G and S for each phase for both PSMs"""
        print("\nGenerating phase distribution visualization (Left & Right PSM)...")
        
        colors = {'Slow': '#FF6B6B', 'Fast': '#45B7D1'}
        
        for psm_side in self.psm_sides:
            psm_name = 'Left PSM' if psm_side == 'L' else 'Right PSM'
            
            phases_to_plot = [p for p in range(0, 6) 
                            if len(self.phase_data[psm_side]['Slow'][p]['G']) > 0 
                            and len(self.phase_data[psm_side]['Fast'][p]['G']) > 0]
            
            if not phases_to_plot:
                print(f"No valid phases to plot for {psm_name}")
                continue
            
            n_phases = len(phases_to_plot)
            fig, axes = plt.subplots(n_phases, 2, figsize=(14, 3.5 * n_phases))
            
            if n_phases == 1:
                axes = axes.reshape(1, -1)
            
            for row, phase in enumerate(phases_to_plot):
                # Gracefulness distribution
                ax = axes[row, 0]
                slow_G = self.phase_data[psm_side]['Slow'][phase]['G']
                fast_G = self.phase_data[psm_side]['Fast'][phase]['G']
                
                bins = np.histogram_bin_edges(slow_G + fast_G, bins=15)
                ax.hist(slow_G, bins=bins, alpha=0.6, color=colors['Slow'], 
                       label=f'Slow (n={len(slow_G)})', density=True, edgecolor='white')
                ax.hist(fast_G, bins=bins, alpha=0.6, color=colors['Fast'], 
                       label=f'Fast (n={len(fast_G)})', density=True, edgecolor='white')
                
                ax.axvline(np.median(slow_G), color=colors['Slow'], linestyle='--', linewidth=2)
                ax.axvline(np.median(fast_G), color=colors['Fast'], linestyle='--', linewidth=2)
                
                ax.set_xlabel('Gracefulness (G)', fontsize=10)
                ax.set_ylabel('Density', fontsize=10)
                ax.set_title(f'Phase {phase} - Gracefulness', fontsize=11, fontweight='bold')
                ax.legend(fontsize=8)
                ax.grid(alpha=0.3)
                
                # Smoothness distribution
                ax = axes[row, 1]
                slow_S = self.phase_data[psm_side]['Slow'][phase]['S']
                fast_S = self.phase_data[psm_side]['Fast'][phase]['S']
                
                bins = np.histogram_bin_edges(slow_S + fast_S, bins=15)
                ax.hist(slow_S, bins=bins, alpha=0.6, color=colors['Slow'], 
                       label=f'Slow (n={len(slow_S)})', density=True, edgecolor='white')
                ax.hist(fast_S, bins=bins, alpha=0.6, color=colors['Fast'], 
                       label=f'Fast (n={len(fast_S)})', density=True, edgecolor='white')
                
                ax.axvline(np.median(slow_S), color=colors['Slow'], linestyle='--', linewidth=2)
                ax.axvline(np.median(fast_S), color=colors['Fast'], linestyle='--', linewidth=2)
                
                ax.set_xlabel('Smoothness (S)', fontsize=10)
                ax.set_ylabel('Density', fontsize=10)
                ax.set_title(f'Phase {phase} - Smoothness', fontsize=11, fontweight='bold')
                ax.legend(fontsize=8)
                ax.grid(alpha=0.3)
            
            plt.suptitle(f'{psm_name} - G and S Distributions by Phase',
                        fontsize=14, fontweight='bold', y=1.01)
            plt.tight_layout()
            
            if save_fig:
                save_dir = os.path.join(self.data_base_dir, '..', 'statistical_results')
                os.makedirs(save_dir, exist_ok=True)
                save_path = os.path.join(save_dir, f'phase_distribution_{psm_side}psm.png')
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                print(f"  ✓ Figure saved to {save_path}")
            
            plt.show()
    
    def generate_summary_table(self, results):
        """Generate summary table of results for both PSMs"""
        print("\n" + "="*110)
        print("SUMMARY TABLE: Phase-wise Gracefulness and Smoothness Comparison (Left & Right PSM)")
        print("="*110)
        
        for psm_side in self.psm_sides:
            psm_name = 'Left PSM' if psm_side == 'L' else 'Right PSM'
            
            if psm_side not in results:
                continue
                
            print(f"\n{'-'*50}")
            print(f"{psm_name}")
            print(f"{'-'*50}")
            
            print(f"\n{'Phase':<10} | {'Metric':<15} | {'Slow Mean±Std':<18} | {'Fast Mean±Std':<18} | {'p-value':<12} | {'Effect (r)':<12} | {'Sig':<5}")
            print("-" * 100)
            
            for phase in sorted(results[psm_side].keys()):
                for metric in ['G', 'S']:
                    if metric not in results[psm_side][phase]:
                        continue
                    r = results[psm_side][phase][metric]
                    metric_name = 'Gracefulness' if metric == 'G' else 'Smoothness'
                    sig = '***' if r['p_value'] < 0.001 else '**' if r['p_value'] < 0.01 else '*' if r['p_value'] < 0.05 else 'ns'
                    
                    slow_str = f"{r['slow_mean']:.3f}±{r['slow_std']:.3f}"
                    fast_str = f"{r['fast_mean']:.3f}±{r['fast_std']:.3f}"
                    
                    print(f"Phase {phase:<4} | {metric_name:<15} | {slow_str:<18} | {fast_str:<18} | {r['p_value']:<12.4e} | {r['effect_size']:<12.4f} | {sig:<5}")
        
        print("="*110)
        print("Legend: *** p<0.001, ** p<0.01, * p<0.05, ns: not significant")
        print("Effect size (r): rank-biserial correlation")
    
    def save_report(self, results, save_dir=None):
        """Save detailed report to file for both PSMs"""
        if save_dir is None:
            save_dir = os.path.join(self.data_base_dir, '..', 'statistical_results')
        os.makedirs(save_dir, exist_ok=True)
        
        report_path = os.path.join(save_dir, 'phase_metrics_report.txt')
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("="*80 + "\n")
            f.write("PHASE-WISE GRACEFULNESS AND SMOOTHNESS ANALYSIS\n")
            f.write("Left and Right PSM\n")
            f.write("Slow Group (data 45-54) vs Fast Group (data 55-64)\n")
            f.write("="*80 + "\n\n")
            
            for psm_side in self.psm_sides:
                psm_name = 'Left PSM' if psm_side == 'L' else 'Right PSM'
                
                if psm_side not in results:
                    continue
                
                f.write(f"\n{'='*60}\n")
                f.write(f"{psm_name}\n")
                f.write(f"{'='*60}\n")
                
                for phase in sorted(results[psm_side].keys()):
                    f.write(f"\nPhase {phase}:\n")
                    f.write("-"*50 + "\n")
                    
                    for metric in ['G', 'S']:
                        if metric not in results[psm_side][phase]:
                            continue
                        r = results[psm_side][phase][metric]
                        metric_name = 'Gracefulness (G)' if metric == 'G' else 'Smoothness (S)'
                        sig = '***' if r['p_value'] < 0.001 else '**' if r['p_value'] < 0.01 else '*' if r['p_value'] < 0.05 else 'ns'
                        
                        f.write(f"\n  {metric_name}:\n")
                        f.write(f"    Slow:  n={r['slow_n']}, Mean={r['slow_mean']:.4f} ± {r['slow_std']:.4f}\n")
                        f.write(f"    Fast:  n={r['fast_n']}, Mean={r['fast_mean']:.4f} ± {r['fast_std']:.4f}\n")
                        f.write(f"    Mann-Whitney U: U={r['u_stat']:.2f}, p={r['p_value']:.4e}\n")
                        f.write(f"    Effect size (r): {r['effect_size']:.4f}\n")
                        f.write(f"    Significance: {sig}\n")
            
            f.write("\n" + "="*80 + "\n")
            f.write("Legend:\n")
            f.write("  *** : p < 0.001 (highly significant)\n")
            f.write("  **  : p < 0.01  (very significant)\n")
            f.write("  *   : p < 0.05  (significant)\n")
            f.write("  ns  : not significant\n")
            f.write("  r   : effect size (rank-biserial correlation)\n")
            f.write("="*80 + "\n")
        
        print(f"  ✓ Report saved to {report_path}")
    
    def run_full_analysis(self, save_fig=True):
        """Run complete phase analysis"""
        print("\n" + "="*80)
        print("PHASE-WISE GRACEFULNESS AND SMOOTHNESS ANALYSIS")
        print("="*80)
        
        # Load data
        self.load_all_data()
        
        # Statistical tests
        results = self.perform_statistical_tests()
        
        # Generate summary table
        self.generate_summary_table(results)
        
        # Visualizations
        self.visualize_phase_comparison(save_fig=save_fig)
        #self.visualize_distribution_by_phase(save_fig=save_fig)
        
        # Save report
        self.save_report(results)
        
        print("\n" + "="*80)
        print("Analysis complete!")
        print("="*80)
        
        return results


class SPARCPhaseAnalyzer:
    """
    Analyze phase-wise metrics using SPARC (Spectral Arc Length) smoothness.
    SPARC is a robust smoothness metric based on frequency spectrum analysis.
    """
    
    def __init__(self, data_base_dir=None):
        if data_base_dir is None:
            self.data_base_dir = os.path.join(project_dir, 'data')
        else:
            self.data_base_dir = data_base_dir
        
        # Define groups
        self.groups = {
            'Slow': list(range(46, 55)),
            'Fast': list(range(56, 65))
        }
        
        self.phase_names = {
            -1: 'Unlabeled',
            0: 'Phase 0', 1: 'Phase 1', 2: 'Phase 2',
            3: 'Phase 3', 4: 'Phase 4', 5: 'Phase 5'
        }
        
        self.phase_data = {}
        self.psm_sides = ['L', 'R']
    
    def calculate_sparc_metrics(self, data):
        """
        Calculate SPARC smoothness and Gracefulness from trajectory data.
        
        Args:
            data: ndarray of shape (n, 4) with [x, y, z, time]
            
        Returns:
            dict with 'G' (gracefulness), 'SPARC' (spectral arc length smoothness)
        """
        positions = data[:, :3]
        timestamps = data[:, 3]
        
        # Calculate velocity (speed profile)
        velocities = np.gradient(positions, axis=0)
        speed = np.linalg.norm(velocities, axis=1)
        
        # Estimate sampling frequency from timestamps
        dt = np.median(np.diff(timestamps))
        if dt <= 0:
            dt = 0.01  # default to 100Hz if invalid
        fs = 1.0 / dt
        print(f"Sampling frequency: {fs}")
        
        # Calculate SPARC smoothness
        try:
            sparc_val, _, _ = sparc(speed, fs, padlevel=4, fc=10.0, amp_th=0.05)
        except Exception as e:
            sparc_val = np.nan
        
        # Calculate Gracefulness (for comparison)
        try:
            G = calculate_gracefulness(data)
        except:
            G = np.nan
        
        # Calculate Log Dimensionless Jerk (LDLJ) for additional comparison
        try:
            ldlj = log_dimensionless_jerk(speed, fs)
        except:
            ldlj = np.nan
        
        return {'G': G, 'SPARC': sparc_val, 'LDLJ': ldlj}
    
    def load_trajectory_with_labels(self, data_id, psm_side='L'):
        """Load trajectory data and phase labels"""
        data_dir = os.path.join(self.data_base_dir, f'{data_id}_data_12-01')
        position_path = os.path.join(data_dir, f'{psm_side}psm_position.npy')
        
        if not os.path.exists(position_path):
            return None
        
        try:
            positions = np.load(position_path, allow_pickle=True)
            
            if positions.shape[1] < 4:
                return None
            
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
            n_frames = len(data)
            
            # Load phase labels
            try:
                labels = generate_frame_label_map(needle_dir_path, data_id)
                if len(labels) != n_frames:
                    label_indices = np.linspace(0, len(labels)-1, n_frames).astype(int)
                    labels = [labels[i] for i in label_indices]
            except:
                labels = [-1] * n_frames
            
            return {
                'data': data,
                'labels': np.array(labels),
                'n_frames': n_frames
            }
            
        except Exception as e:
            return None
    
    def load_all_data(self):
        """Load and process all data using SPARC smoothness"""
        print("Loading data for SPARC-based phase analysis...")
        print("Metrics:")
        print("  G = median(log10(curvature)) - Gracefulness")
        print("  SPARC = Spectral Arc Length - Smoothness (more negative = smoother)")
        print("  LDLJ = Log Dimensionless Jerk - Smoothness (more negative = smoother)")
        
        # Initialize storage
        for psm_side in self.psm_sides:
            self.phase_data[psm_side] = {}
            for group_name in self.groups.keys():
                self.phase_data[psm_side][group_name] = {}
                for phase in range(-1, 6):
                    self.phase_data[psm_side][group_name][phase] = {
                        'G': [], 'SPARC': [], 'LDLJ': []
                    }
        
        for psm_side in self.psm_sides:
            psm_name = 'Left' if psm_side == 'L' else 'Right'
            print(f"\n{'='*50}")
            print(f"Processing {psm_name} PSM")
            print(f"{'='*50}")
            
            for group_name, data_ids in self.groups.items():
                print(f"\n  {group_name} group...")
                
                for data_id in data_ids:
                    result = self.load_trajectory_with_labels(data_id, psm_side)
                    
                    if result is None:
                        continue
                    
                    unique_phases = np.unique(result['labels'])
                    
                    for phase in unique_phases:
                        phase_mask = result['labels'] == phase
                        
                        if np.sum(phase_mask) < 10:  # Need enough points for SPARC
                            continue
                        
                        phase_data = result['data'][phase_mask]
                        metrics = self.calculate_sparc_metrics(phase_data)
                        
                        if not np.isnan(metrics['SPARC']):
                            self.phase_data[psm_side][group_name][phase]['G'].append(metrics['G'])
                            self.phase_data[psm_side][group_name][phase]['SPARC'].append(metrics['SPARC'])
                            self.phase_data[psm_side][group_name][phase]['LDLJ'].append(metrics['LDLJ'])
                    
                    print(f"    ✓ Demo {data_id}")
        
        # Print summary
        print("\n" + "="*60)
        print("Data Loading Summary:")
        print("="*60)
        for psm_side in self.psm_sides:
            psm_name = 'Left PSM' if psm_side == 'L' else 'Right PSM'
            print(f"\n{psm_name}:")
            for group_name in self.groups.keys():
                print(f"  {group_name} group:")
                for phase in range(0, 6):
                    n_demos = len(self.phase_data[psm_side][group_name][phase]['SPARC'])
                    if n_demos > 0:
                        print(f"    Phase {phase}: {n_demos} demos")
    
    def perform_statistical_tests(self):
        """Perform Mann-Whitney U tests for SPARC metrics"""
        print("\n" + "="*80)
        print("Statistical Tests: SPARC Smoothness - Slow vs Fast")
        print("="*80)
        
        results = {}
        
        for psm_side in self.psm_sides:
            psm_name = 'Left PSM' if psm_side == 'L' else 'Right PSM'
            results[psm_side] = {}
            
            print(f"\n{'='*60}")
            print(f"{psm_name}")
            print(f"{'='*60}")
            
            for phase in range(0, 6):
                slow_sparc = np.array(self.phase_data[psm_side]['Slow'][phase]['SPARC'])
                fast_sparc = np.array(self.phase_data[psm_side]['Fast'][phase]['SPARC'])
                slow_G = np.array(self.phase_data[psm_side]['Slow'][phase]['G'])
                fast_G = np.array(self.phase_data[psm_side]['Fast'][phase]['G'])
                slow_ldlj = np.array(self.phase_data[psm_side]['Slow'][phase]['LDLJ'])
                fast_ldlj = np.array(self.phase_data[psm_side]['Fast'][phase]['LDLJ'])
                
                if len(slow_sparc) < 2 or len(fast_sparc) < 2:
                    continue
                
                results[psm_side][phase] = {}
                
                print(f"\nPhase {phase}:")
                print("-" * 50)
                
                # SPARC test
                u_sparc, p_sparc = mannwhitneyu(slow_sparc, fast_sparc, alternative='two-sided')
                n1, n2 = len(slow_sparc), len(fast_sparc)
                r_sparc = 1 - (2 * u_sparc) / (n1 * n2)
                
                results[psm_side][phase]['SPARC'] = {
                    'slow_mean': np.mean(slow_sparc), 'slow_std': np.std(slow_sparc),
                    'fast_mean': np.mean(fast_sparc), 'fast_std': np.std(fast_sparc),
                    'u_stat': u_sparc, 'p_value': p_sparc, 'effect_size': r_sparc,
                    'slow_n': n1, 'fast_n': n2
                }
                
                sig = '***' if p_sparc < 0.001 else '**' if p_sparc < 0.01 else '*' if p_sparc < 0.05 else 'ns'
                print(f"  SPARC Smoothness:")
                print(f"    Slow:  n={n1}, Mean={np.mean(slow_sparc):.4f} ± {np.std(slow_sparc):.4f}")
                print(f"    Fast:  n={n2}, Mean={np.mean(fast_sparc):.4f} ± {np.std(fast_sparc):.4f}")
                print(f"    Mann-Whitney U: p={p_sparc:.4e}, r={r_sparc:.4f} {sig}")
                
                # Gracefulness test
                u_G, p_G = mannwhitneyu(slow_G, fast_G, alternative='two-sided')
                r_G = 1 - (2 * u_G) / (n1 * n2)
                
                results[psm_side][phase]['G'] = {
                    'slow_mean': np.mean(slow_G), 'slow_std': np.std(slow_G),
                    'fast_mean': np.mean(fast_G), 'fast_std': np.std(fast_G),
                    'u_stat': u_G, 'p_value': p_G, 'effect_size': r_G,
                    'slow_n': n1, 'fast_n': n2
                }
                
                sig_G = '***' if p_G < 0.001 else '**' if p_G < 0.01 else '*' if p_G < 0.05 else 'ns'
                print(f"  Gracefulness (G):")
                print(f"    Slow:  Mean={np.mean(slow_G):.4f} ± {np.std(slow_G):.4f}")
                print(f"    Fast:  Mean={np.mean(fast_G):.4f} ± {np.std(fast_G):.4f}")
                print(f"    Mann-Whitney U: p={p_G:.4e}, r={r_G:.4f} {sig_G}")
                
                # LDLJ test
                valid_slow_ldlj = slow_ldlj[np.isfinite(slow_ldlj)]
                valid_fast_ldlj = fast_ldlj[np.isfinite(fast_ldlj)]
                if len(valid_slow_ldlj) >= 2 and len(valid_fast_ldlj) >= 2:
                    u_ldlj, p_ldlj = mannwhitneyu(valid_slow_ldlj, valid_fast_ldlj, alternative='two-sided')
                    r_ldlj = 1 - (2 * u_ldlj) / (len(valid_slow_ldlj) * len(valid_fast_ldlj))
                    
                    results[psm_side][phase]['LDLJ'] = {
                        'slow_mean': np.mean(valid_slow_ldlj), 'slow_std': np.std(valid_slow_ldlj),
                        'fast_mean': np.mean(valid_fast_ldlj), 'fast_std': np.std(valid_fast_ldlj),
                        'u_stat': u_ldlj, 'p_value': p_ldlj, 'effect_size': r_ldlj
                    }
                    
                    sig_ldlj = '***' if p_ldlj < 0.001 else '**' if p_ldlj < 0.01 else '*' if p_ldlj < 0.05 else 'ns'
                    print(f"  Log Dimensionless Jerk (LDLJ):")
                    print(f"    Slow:  Mean={np.mean(valid_slow_ldlj):.4f} ± {np.std(valid_slow_ldlj):.4f}")
                    print(f"    Fast:  Mean={np.mean(valid_fast_ldlj):.4f} ± {np.std(valid_fast_ldlj):.4f}")
                    print(f"    Mann-Whitney U: p={p_ldlj:.4e}, r={r_ldlj:.4f} {sig_ldlj}")
        
        return results
    
    def visualize_sparc_comparison(self, save_fig=True):
        """Visualize SPARC smoothness comparison"""
        print("\nGenerating SPARC comparison visualization...")
        
        colors = {'Slow': '#FF6B6B', 'Fast': '#45B7D1'}
        
        # Create figure: 2 rows (L/R PSM) x 3 columns (SPARC, G, LDLJ)
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        
        for row, psm_side in enumerate(self.psm_sides):
            psm_name = 'Left PSM' if psm_side == 'L' else 'Right PSM'
            
            phases_to_plot = [p for p in range(0, 6) 
                            if len(self.phase_data[psm_side]['Slow'][p]['SPARC']) > 0 
                            and len(self.phase_data[psm_side]['Fast'][p]['SPARC']) > 0]
            
            if not phases_to_plot:
                continue
            
            # Plot SPARC
            ax = axes[row, 2]
            plot_data = []
            for phase in phases_to_plot:
                for group in ['Slow', 'Fast']:
                    for val in self.phase_data[psm_side][group][phase]['SPARC']:
                        plot_data.append({'Phase': f'P{phase}', 'Group': group, 'Value': val})
            
            df = pd.DataFrame(plot_data)
            sns.boxplot(data=df, x='Phase', y='Value', hue='Group', ax=ax, palette=colors)
            ax.set_title(f'{psm_name} - SPARC Smoothness', fontsize=12, fontweight='bold')
            ax.set_ylabel('SPARC (more negative = smoother)', fontsize=10)
            ax.set_xlabel('Phase', fontsize=10)
            ax.legend(title='Group', fontsize=8)
            ax.grid(axis='y', alpha=0.3)
            
            # Plot Gracefulness
            ax = axes[row, 0]
            plot_data = []
            for phase in phases_to_plot:
                for group in ['Slow', 'Fast']:
                    for val in self.phase_data[psm_side][group][phase]['G']:
                        plot_data.append({'Phase': f'P{phase}', 'Group': group, 'Value': val})
            
            df = pd.DataFrame(plot_data)
            sns.boxplot(data=df, x='Phase', y='Value', hue='Group', ax=ax, palette=colors)
            ax.set_title(f'{psm_name} - Log Dimensionless Jerk(position)', fontsize=12, fontweight='bold')
            ax.set_ylabel('Log Curvature (G)', fontsize=10)
            ax.set_xlabel('Phase', fontsize=10)
            ax.legend(title='Group', fontsize=8)
            ax.grid(axis='y', alpha=0.3)
            
            # Plot LDLJ
            ax = axes[row, 1]
            plot_data = []
            for phase in phases_to_plot:
                for group in ['Slow', 'Fast']:
                    for val in self.phase_data[psm_side][group][phase]['LDLJ']:
                        if np.isfinite(val):
                            plot_data.append({'Phase': f'P{phase}', 'Group': group, 'Value': val})
            
            df = pd.DataFrame(plot_data)
            if len(df) > 0:
                sns.boxplot(data=df, x='Phase', y='Value', hue='Group', ax=ax, palette=colors)
            ax.set_title(f'{psm_name} - Log Dimensionless Jerk(velocity)', fontsize=12, fontweight='bold')
            ax.set_ylabel('LDLJ (more negative = smoother)', fontsize=10)
            ax.set_xlabel('Phase', fontsize=10)
            ax.legend(title='Group', fontsize=8)
            ax.grid(axis='y', alpha=0.3)
        
        plt.suptitle('Phase-wise Smoothness Metrics Comparison (SPARC, G, LDLJ)',
                    fontsize=14, fontweight='bold', y=1.02)
        plt.tight_layout()
        
        if save_fig:
            save_dir = os.path.join(self.data_base_dir, '..', 'statistical_results')
            os.makedirs(save_dir, exist_ok=True)
            save_path = os.path.join(save_dir, 'phase_sparc_comparison.png')
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"  ✓ Figure saved to {save_path}")
        
        plt.show()
    
    def generate_summary_table(self, results):
        """Generate summary table"""
        print("\n" + "="*120)
        print("SUMMARY TABLE: SPARC-based Phase Analysis")
        print("="*120)
        
        for psm_side in self.psm_sides:
            psm_name = 'Left PSM' if psm_side == 'L' else 'Right PSM'
            
            if psm_side not in results:
                continue
            
            print(f"\n{'-'*50}")
            print(f"{psm_name}")
            print(f"{'-'*50}")
            
            print(f"\n{'Phase':<8} | {'Metric':<8} | {'Slow Mean±Std':<20} | {'Fast Mean±Std':<20} | {'p-value':<12} | {'r':<8} | {'Sig':<5}")
            print("-" * 100)
            
            for phase in sorted(results[psm_side].keys()):
                for metric in ['SPARC', 'G', 'LDLJ']:
                    if metric not in results[psm_side][phase]:
                        continue
                    r = results[psm_side][phase][metric]
                    sig = '***' if r['p_value'] < 0.001 else '**' if r['p_value'] < 0.01 else '*' if r['p_value'] < 0.05 else 'ns'
                    
                    slow_str = f"{r['slow_mean']:.3f}±{r['slow_std']:.3f}"
                    fast_str = f"{r['fast_mean']:.3f}±{r['fast_std']:.3f}"
                    
                    print(f"Phase {phase:<2} | {metric:<8} | {slow_str:<20} | {fast_str:<20} | {r['p_value']:<12.4e} | {r['effect_size']:<8.4f} | {sig:<5}")
        
        print("="*120)
        print("Note: For SPARC and LDLJ, more negative values indicate smoother movement")
    
    def save_report(self, results, save_dir=None):
        """Save detailed report"""
        if save_dir is None:
            save_dir = os.path.join(self.data_base_dir, '..', 'statistical_results')
        os.makedirs(save_dir, exist_ok=True)
        
        report_path = os.path.join(save_dir, 'phase_sparc_report.txt')
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("="*80 + "\n")
            f.write("SPARC-BASED PHASE ANALYSIS\n")
            f.write("Slow Group vs Fast Group\n")
            f.write("="*80 + "\n\n")
            f.write("Metrics:\n")
            f.write("  SPARC: Spectral Arc Length (more negative = smoother)\n")
            f.write("  G: Gracefulness = median(log10(curvature))\n")
            f.write("  LDLJ: Log Dimensionless Jerk (more negative = smoother)\n\n")
            
            for psm_side in self.psm_sides:
                psm_name = 'Left PSM' if psm_side == 'L' else 'Right PSM'
                
                if psm_side not in results:
                    continue
                
                f.write(f"\n{'='*60}\n")
                f.write(f"{psm_name}\n")
                f.write(f"{'='*60}\n")
                
                for phase in sorted(results[psm_side].keys()):
                    f.write(f"\nPhase {phase}:\n")
                    f.write("-"*50 + "\n")
                    
                    for metric in ['SPARC', 'G', 'LDLJ']:
                        if metric not in results[psm_side][phase]:
                            continue
                        r = results[psm_side][phase][metric]
                        sig = '***' if r['p_value'] < 0.001 else '**' if r['p_value'] < 0.01 else '*' if r['p_value'] < 0.05 else 'ns'
                        
                        f.write(f"\n  {metric}:\n")
                        f.write(f"    Slow:  Mean={r['slow_mean']:.4f} ± {r['slow_std']:.4f}\n")
                        f.write(f"    Fast:  Mean={r['fast_mean']:.4f} ± {r['fast_std']:.4f}\n")
                        f.write(f"    p-value: {r['p_value']:.4e}, effect size: {r['effect_size']:.4f}\n")
                        f.write(f"    Significance: {sig}\n")
            
            f.write("\n" + "="*80 + "\n")
            f.write("Legend: *** p<0.001, ** p<0.01, * p<0.05, ns: not significant\n")
            f.write("="*80 + "\n")
        
        print(f"  ✓ Report saved to {report_path}")
    
    def run_full_analysis(self, save_fig=True):
        """Run complete SPARC-based analysis"""
        print("\n" + "="*80)
        print("SPARC-BASED PHASE ANALYSIS")
        print("="*80)
        
        self.load_all_data()
        results = self.perform_statistical_tests()
        self.generate_summary_table(results)
        self.visualize_sparc_comparison(save_fig=save_fig)
        self.save_report(results)
        
        print("\n" + "="*80)
        print("SPARC Analysis complete!")
        print("="*80)
        
        return results


def main():
    """Main function"""
    analyzer = PhaseMetricsAnalyzer()
    results = analyzer.run_full_analysis()
    return results


def main_sparc():
    """Main function for SPARC-based analysis"""
    analyzer = SPARCPhaseAnalyzer()
    results = analyzer.run_full_analysis()
    return results


if __name__ == "__main__":
    # Run SPARC-based analysis
    main_sparc()

