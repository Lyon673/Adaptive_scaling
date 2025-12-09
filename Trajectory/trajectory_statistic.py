import numpy as np
import matplotlib.pyplot as plt
from load_data import (load_demonstrations_state, load_demonstrations_label, 
                       demo_id_list, needle_dir_path, read_demo_kinematics_state)
import os


def filter_valid_frames(trajectory, labels):
    """
    过滤掉label为-1的帧，只保留有效数据
    
    参数:
        trajectory: numpy array, shape (N, 16)
        labels: list of int, 长度为N，每帧的label
    
    返回:
        filtered_trajectory: 只包含有效label的轨迹数据
        valid_indices: 有效帧的索引
    """
    labels = np.array(labels)
    valid_mask = labels != -1
    valid_indices = np.where(valid_mask)[0]
    filtered_trajectory = trajectory[valid_mask]
    
    return filtered_trajectory, valid_indices


def compute_step_distances(trajectory, labels=None):
    """
    计算轨迹中每一步的移动距离
    
    参数:
        trajectory: numpy array, shape (N, 16)
                   左手: [pos(3), quat(4), grip(1)]
                   右手: [pos(3), quat(4), grip(1)]
        labels: list of int, 可选。如果提供，只计算label不为-1的帧之间的距离
    
    返回:
        left_dists: 左手每步的移动距离
        right_dists: 右手每步的移动距离  
        combined_dists: 双臂6D联合空间的移动距离
    """
    # 如果提供了labels，先过滤
    if labels is not None:
        trajectory, valid_indices = filter_valid_frames(trajectory, labels)
        if len(trajectory) < 2:
            # 有效帧太少，返回空数组
            return np.array([]), np.array([]), np.array([])
    
    # 提取位置信息
    l_pos = trajectory[:, :3]      # 左手位置
    r_pos = trajectory[:, 8:11]    # 右手位置
    
    # 计算左手每步距离
    l_deltas = np.diff(l_pos, axis=0)
    left_dists = np.linalg.norm(l_deltas, axis=1)
    
    # 计算右手每步距离
    r_deltas = np.diff(r_pos, axis=0)
    right_dists = np.linalg.norm(r_deltas, axis=1)
    
    # 计算双臂联合6D空间距离
    combined_traj = np.hstack((l_pos, r_pos))
    combined_deltas = np.diff(combined_traj, axis=0)
    combined_dists = np.linalg.norm(combined_deltas, axis=1)
    
    return left_dists, right_dists, combined_dists


def analyze_all_demos_distances(use_labels=True, filter_zero_motion=True, zero_threshold=1e-9):
    """
    分析所有demo的差分距离统计信息
    
    参数:
        use_labels: bool, 是否使用label过滤数据（只统计label不为-1的帧）
        filter_zero_motion: bool, 是否过滤掉位移为0的数据
        zero_threshold: float, 判断位移为0的阈值（小于此值视为0）
    
    返回:
        stats: 包含统计信息的字典
    """
    demos = load_demonstrations_state()
    labels_list = load_demonstrations_label() if use_labels else None
    
    all_left_dists = []
    all_right_dists = []
    all_combined_dists = []
    
    demo_stats = []  # 每个demo的统计信息
    
    total_zero_steps = 0  # 统计被过滤掉的0位移步数
    
    for demo_idx, demo in enumerate(demos):
        # 获取当前demo的label（如果使用）
        demo_labels = labels_list[demo_idx] if use_labels else None
        
        l_dists, r_dists, c_dists = compute_step_distances(demo, demo_labels)
        
        # 跳过空结果
        if len(c_dists) == 0:
            print(f"Warning: Demo {demo_idx} has no valid steps after filtering")
            continue
        
        # 记录过滤前的步数
        num_steps_before = len(c_dists)
        
        # 过滤掉位移为0的数据（如果启用）
        if filter_zero_motion:
            # 找出非零位移的索引
            non_zero_mask = c_dists > zero_threshold
            l_dists = l_dists[non_zero_mask]
            r_dists = r_dists[non_zero_mask]
            c_dists = c_dists[non_zero_mask]
            
            zero_steps = num_steps_before - len(c_dists)
            total_zero_steps += zero_steps
            
            if len(c_dists) == 0:
                print(f"Warning: Demo {demo_idx} has no non-zero motion steps")
                continue
        
        all_left_dists.extend(l_dists)
        all_right_dists.extend(r_dists)
        all_combined_dists.extend(c_dists)
        
        # 记录每个demo的统计
        valid_frame_count = np.sum(np.array(demo_labels) != -1) if use_labels else len(demo)
        
        demo_stats.append({
            'demo_id': demo_idx,
            'left_mean': np.mean(l_dists),
            'right_mean': np.mean(r_dists),
            'combined_mean': np.mean(c_dists),
            'left_max': np.max(l_dists),
            'right_max': np.max(r_dists),
            'combined_max': np.max(c_dists),
            'num_steps': len(c_dists),
            'total_frames': len(demo),
            'valid_frames': valid_frame_count
        })
    
    # 转换为numpy数组
    all_left_dists = np.array(all_left_dists)
    all_right_dists = np.array(all_right_dists)
    all_combined_dists = np.array(all_combined_dists)
    
    stats = {
        'left': {
            'all_dists': all_left_dists,
            'mean': np.mean(all_left_dists),
            'median': np.median(all_left_dists),
            'std': np.std(all_left_dists),
            'min': np.min(all_left_dists),
            'max': np.max(all_left_dists),
            'percentile_25': np.percentile(all_left_dists, 25),
            'percentile_75': np.percentile(all_left_dists, 75),
            'percentile_95': np.percentile(all_left_dists, 95),
        },
        'right': {
            'all_dists': all_right_dists,
            'mean': np.mean(all_right_dists),
            'median': np.median(all_right_dists),
            'std': np.std(all_right_dists),
            'min': np.min(all_right_dists),
            'max': np.max(all_right_dists),
            'percentile_25': np.percentile(all_right_dists, 25),
            'percentile_75': np.percentile(all_right_dists, 75),
            'percentile_95': np.percentile(all_right_dists, 95),
        },
        'combined': {
            'all_dists': all_combined_dists,
            'mean': np.mean(all_combined_dists),
            'median': np.median(all_combined_dists),
            'std': np.std(all_combined_dists),
            'min': np.min(all_combined_dists),
            'max': np.max(all_combined_dists),
            'percentile_25': np.percentile(all_combined_dists, 25),
            'percentile_75': np.percentile(all_combined_dists, 75),
            'percentile_95': np.percentile(all_combined_dists, 95),
        },
        'demo_stats': demo_stats,
        'filter_info': {
            'use_labels': use_labels,
            'filter_zero_motion': filter_zero_motion,
            'zero_threshold': zero_threshold,
            'total_zero_steps_filtered': total_zero_steps
        }
    }
    
    return stats


def visualize_distance_distribution(stats, save_path=None):
    """
    可视化差分距离分布
    
    参数:
        stats: analyze_all_demos_distances() 返回的统计信息
        save_path: 保存图片的路径，如果为None则显示
    """
    fig = plt.figure(figsize=(16, 10))
    
    # 子图1: 左手距离直方图
    plt.subplot(3, 3, 1)
    left_dists = stats['left']['all_dists']
    plt.hist(left_dists, bins=100, alpha=0.7, color='blue', edgecolor='black')
    plt.axvline(stats['left']['mean'], color='red', linestyle='--', linewidth=2, label=f"Mean: {stats['left']['mean']:.4f}")
    plt.axvline(stats['left']['median'], color='green', linestyle='--', linewidth=2, label=f"Median: {stats['left']['median']:.4f}")
    plt.xlabel('Distance')
    plt.ylabel('Frequency')
    plt.title('Left Hand Step Distance Distribution')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 子图2: 右手距离直方图
    plt.subplot(3, 3, 2)
    right_dists = stats['right']['all_dists']
    plt.hist(right_dists, bins=100, alpha=0.7, color='orange', edgecolor='black')
    plt.axvline(stats['right']['mean'], color='red', linestyle='--', linewidth=2, label=f"Mean: {stats['right']['mean']:.4f}")
    plt.axvline(stats['right']['median'], color='green', linestyle='--', linewidth=2, label=f"Median: {stats['right']['median']:.4f}")
    plt.xlabel('Distance')
    plt.ylabel('Frequency')
    plt.title('Right Hand Step Distance Distribution')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 子图3: 双臂联合距离直方图
    plt.subplot(3, 3, 3)
    combined_dists = stats['combined']['all_dists']
    plt.hist(combined_dists, bins=100, alpha=0.7, color='purple', edgecolor='black')
    plt.axvline(stats['combined']['mean'], color='red', linestyle='--', linewidth=2, label=f"Mean: {stats['combined']['mean']:.4f}")
    plt.axvline(stats['combined']['median'], color='green', linestyle='--', linewidth=2, label=f"Median: {stats['combined']['median']:.4f}")
    plt.xlabel('Distance')
    plt.ylabel('Frequency')
    plt.title('Combined (6D) Step Distance Distribution')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 子图4: 累积分布函数 (CDF)
    plt.subplot(3, 3, 4)
    sorted_left = np.sort(left_dists)
    cdf_left = np.arange(1, len(sorted_left) + 1) / len(sorted_left)
    plt.plot(sorted_left, cdf_left, label='Left', alpha=0.7, linewidth=2)
    
    sorted_right = np.sort(right_dists)
    cdf_right = np.arange(1, len(sorted_right) + 1) / len(sorted_right)
    plt.plot(sorted_right, cdf_right, label='Right', alpha=0.7, linewidth=2)
    
    sorted_combined = np.sort(combined_dists)
    cdf_combined = np.arange(1, len(sorted_combined) + 1) / len(sorted_combined)
    plt.plot(sorted_combined, cdf_combined, label='Combined', alpha=0.7, linewidth=2)
    
    plt.xlabel('Distance')
    plt.ylabel('Cumulative Probability')
    plt.title('Cumulative Distribution Function (CDF)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 子图5: 箱线图对比
    plt.subplot(3, 3, 5)
    data_to_plot = [left_dists, right_dists, combined_dists]
    plt.boxplot(data_to_plot, labels=['Left', 'Right', 'Combined'])
    plt.ylabel('Distance')
    plt.title('Step Distance Box Plot Comparison')
    plt.grid(True, alpha=0.3)
    
    # 子图6: 对数尺度直方图（用于查看小值分布）
    plt.subplot(3, 3, 6)
    plt.hist(left_dists[left_dists > 0], bins=100, alpha=0.5, color='blue', label='Left')
    plt.hist(right_dists[right_dists > 0], bins=100, alpha=0.5, color='orange', label='Right')
    plt.hist(combined_dists[combined_dists > 0], bins=100, alpha=0.5, color='purple', label='Combined')
    plt.xlabel('Distance')
    plt.ylabel('Frequency')
    plt.yscale('log')
    plt.title('Distance Distribution (Log Scale)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 子图7: 每个demo的平均距离对比
    plt.subplot(3, 3, 7)
    demo_stats = stats['demo_stats']
    demo_indices = [s['demo_id'] for s in demo_stats]
    left_means = [s['left_mean'] for s in demo_stats]
    right_means = [s['right_mean'] for s in demo_stats]
    combined_means = [s['combined_mean'] for s in demo_stats]
    
    x = np.arange(len(demo_indices))
    width = 0.25
    plt.bar(x - width, left_means, width, label='Left', alpha=0.8)
    plt.bar(x, right_means, width, label='Right', alpha=0.8)
    plt.bar(x + width, combined_means, width, label='Combined', alpha=0.8)
    
    plt.xlabel('Demo Index')
    plt.ylabel('Mean Step Distance')
    plt.title('Average Step Distance per Demo')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 子图8: 统计信息文本
    plt.subplot(3, 3, 8)
    plt.axis('off')
    
    # 计算有效帧统计
    total_valid_frames = sum([s['valid_frames'] for s in stats['demo_stats']])
    total_all_frames = sum([s['total_frames'] for s in stats['demo_stats']])
    valid_ratio = total_valid_frames / total_all_frames if total_all_frames > 0 else 0
    
    # 过滤信息
    filter_info = stats.get('filter_info', {})
    filter_text = ""
    if filter_info.get('use_labels', False):
        filter_text += "    Filters: label≠-1"
    if filter_info.get('filter_zero_motion', False):
        if filter_text:
            filter_text += ", motion>0"
        else:
            filter_text = "    Filters: motion>0"
        zero_steps = filter_info.get('total_zero_steps_filtered', 0)
        filter_text += f"\n    Zero steps filtered: {zero_steps}"
    
    stats_text = f"""
    === Distance Statistics Summary ===
{filter_text}
    
    LEFT HAND:
      Mean:    {stats['left']['mean']:.6f}
      Median:  {stats['left']['median']:.6f}
      Std Dev: {stats['left']['std']:.6f}
      Min:     {stats['left']['min']:.6f}
      Max:     {stats['left']['max']:.6f}
      Q1 (25%): {stats['left']['percentile_25']:.6f}
      Q3 (75%): {stats['left']['percentile_75']:.6f}
      95%:     {stats['left']['percentile_95']:.6f}
    
    RIGHT HAND:
      Mean:    {stats['right']['mean']:.6f}
      Median:  {stats['right']['median']:.6f}
      Std Dev: {stats['right']['std']:.6f}
      Min:     {stats['right']['min']:.6f}
      Max:     {stats['right']['max']:.6f}
      Q1 (25%): {stats['right']['percentile_25']:.6f}
      Q3 (75%): {stats['right']['percentile_75']:.6f}
      95%:     {stats['right']['percentile_95']:.6f}
    
    COMBINED (6D):
      Mean:    {stats['combined']['mean']:.6f}
      Median:  {stats['combined']['median']:.6f}
      Std Dev: {stats['combined']['std']:.6f}
      Min:     {stats['combined']['min']:.6f}
      Max:     {stats['combined']['max']:.6f}
      Q1 (25%): {stats['combined']['percentile_25']:.6f}
      Q3 (75%): {stats['combined']['percentile_75']:.6f}
      95%:     {stats['combined']['percentile_95']:.6f}
    
    Total Steps: {len(stats['combined']['all_dists'])}
    Total Demos: {len(stats['demo_stats'])}
    Valid Frames: {total_valid_frames} / {total_all_frames}
    Valid Ratio: {valid_ratio*100:.1f}%
    """
    
    plt.text(0.05, 0.95, stats_text, transform=plt.gca().transAxes, 
             fontsize=8, verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
    
    # 子图9: 散点图 - 左右手距离相关性
    plt.subplot(3, 3, 9)
    # 使用已经计算好的数据
    left_steps = stats['left']['all_dists']
    right_steps = stats['right']['all_dists']
    
    # 随机采样以避免点太多
    sample_size = min(10000, len(left_steps))
    indices = np.random.choice(len(left_steps), sample_size, replace=False)
    plt.scatter(np.array(left_steps)[indices], np.array(right_steps)[indices], 
                alpha=0.3, s=1)
    plt.xlabel('Left Hand Distance')
    plt.ylabel('Right Hand Distance')
    plt.title('Left vs Right Hand Distance Correlation')
    plt.grid(True, alpha=0.3)
    
    # 添加对角线
    max_val = max(plt.xlim()[1], plt.ylim()[1])
    plt.plot([0, max_val], [0, max_val], 'r--', alpha=0.5, label='y=x')
    plt.legend()
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to {save_path}")
    else:
        plt.show()


def suggest_resampling_step_size(stats, percentile=50):
    """
    根据距离分布建议重采样步长
    
    参数:
        stats: 统计信息
        percentile: 使用第几百分位数作为步长参考
    
    返回:
        建议的步长
    """
    combined_dists = stats['combined']['all_dists']
    suggested_step = np.percentile(combined_dists, percentile)
    
    print(f"\n=== Resampling Step Size Suggestions ===")
    print(f"Based on {percentile}th percentile: {suggested_step:.6f}")
    print(f"Mean distance: {stats['combined']['mean']:.6f}")
    print(f"Median distance: {stats['combined']['median']:.6f}")
    print(f"\nSuggested step sizes:")
    print(f"  Conservative (fine): {suggested_step * 0.5:.6f}")
    print(f"  Moderate: {suggested_step:.6f}")
    print(f"  Aggressive (coarse): {suggested_step * 2:.6f}")
    
    return suggested_step


def print_detailed_stats(stats):
    """打印详细的统计信息"""
    print("\n" + "="*60)
    print("DETAILED STEP DISTANCE STATISTICS")
    print("="*60)
    
    for hand in ['left', 'right', 'combined']:
        hand_name = hand.upper() + (" (6D)" if hand == 'combined' else "")
        print(f"\n{hand_name}:")
        print(f"  Mean:          {stats[hand]['mean']:.8f}")
        print(f"  Median:        {stats[hand]['median']:.8f}")
        print(f"  Std Dev:       {stats[hand]['std']:.8f}")
        print(f"  Min:           {stats[hand]['min']:.8f}")
        print(f"  Max:           {stats[hand]['max']:.8f}")
        print(f"  25th percentile: {stats[hand]['percentile_25']:.8f}")
        print(f"  75th percentile: {stats[hand]['percentile_75']:.8f}")
        print(f"  95th percentile: {stats[hand]['percentile_95']:.8f}")
        print(f"  Total steps:   {len(stats[hand]['all_dists'])}")
    
    # 计算帧统计信息
    total_valid_frames = sum([s['valid_frames'] for s in stats['demo_stats']])
    total_all_frames = sum([s['total_frames'] for s in stats['demo_stats']])
    valid_ratio = total_valid_frames / total_all_frames * 100 if total_all_frames > 0 else 0
    
    print("\n" + "="*60)
    print(f"Total Demonstrations: {len(stats['demo_stats'])}")
    print(f"Total Frames (all):   {total_all_frames}")
    print(f"Valid Frames (used):  {total_valid_frames}")
    print(f"Valid Frame Ratio:    {valid_ratio:.2f}%")
    
    # 显示过滤信息
    filter_info = stats.get('filter_info', {})
    if filter_info.get('filter_zero_motion', False):
        zero_steps = filter_info.get('total_zero_steps_filtered', 0)
        print(f"\nZero Motion Filtering:")
        print(f"  Threshold:        {filter_info.get('zero_threshold', 0):.2e}")
        print(f"  Steps filtered:   {zero_steps}")
        print(f"  Steps retained:   {len(stats['combined']['all_dists'])}")
    
    print("="*60)


if __name__ == '__main__':
    # ==================== 配置参数 ====================
    USE_LABELS = True           # 是否使用label过滤（只统计label不为-1的数据）
    FILTER_ZERO_MOTION = True   # 是否过滤掉位移为0的数据
    ZERO_THRESHOLD = 1e-3       # 判断位移为0的阈值
    # =================================================
    
    print("="*60)
    print("Analyzing step distance distribution")
    if USE_LABELS:
        print("  - Filtering: label != -1")
    if FILTER_ZERO_MOTION:
        print(f"  - Filtering: motion > {ZERO_THRESHOLD}")
    print("="*60)
    
    # 分析所有demo的距离统计
    stats = analyze_all_demos_distances(
        use_labels=USE_LABELS,
        filter_zero_motion=FILTER_ZERO_MOTION,
        zero_threshold=ZERO_THRESHOLD
    )
    
    # 打印详细统计信息
    print_detailed_stats(stats)
    
    # 建议重采样步长
    suggest_resampling_step_size(stats, percentile=50)
    
    # 可视化
    suffix_parts = []
    if USE_LABELS:
        suffix_parts.append('valid_labels')
    if FILTER_ZERO_MOTION:
        suffix_parts.append('nonzero')
    suffix = '_' + '_'.join(suffix_parts) if suffix_parts else ''
    
    save_path = os.path.join(os.path.dirname(__file__), f'distance_distribution_analysis{suffix}.png')
    visualize_distance_distribution(stats, save_path=save_path)
    
    print("\nAnalysis complete!")

