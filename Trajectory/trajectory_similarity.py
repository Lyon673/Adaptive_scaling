"""
DTW距离计算模块 - 轨迹相似度分析

主要功能:
1. 计算demo之间的DTW距离（支持归一化）
2. 过滤特定demo（按ID、长度等）
3. 可视化距离矩阵
4. 查找最近邻demo

DTW距离说明:
- 原始DTW (normalize=False): 累计帧间距离，受轨迹长度影响
- 归一化DTW (normalize=True): 平均每步距离，消除长度影响 ⭐推荐

快速使用:
    # 计算归一化DTW距离（推荐）
    dtw_matrix, demo_ids = compute_dtw_matrix(max_demo_id=65, normalize=True)
    
    # 比较两种距离的差异
    compare_normalized_vs_raw(max_demo_id=10)
    
    # 过滤后计算
    dtw_matrix, demo_ids = compute_dtw_matrix(
        max_demo_id=65, 
        min_length=150, 
        max_length=300,
        exclude_ids=[46],
        normalize=True
    )

详细说明请参考: README_DTW.md
"""

import numpy as np
import os
from load_data import read_demo_kinematics_state, needle_dir_path, resample_bimanual_trajectory
from scipy.spatial.distance import euclidean
from fastdtw import fastdtw
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import pickle
from scipy import stats
from sklearn.metrics import confusion_matrix


def compute_dtw_distance(traj1, traj2, metric='euclidean', normalize=False):
    """
    计算两条轨迹之间的DTW距离
    
    Args:
        traj1: 第一条轨迹 (n_frames1, n_features)
        traj2: 第二条轨迹 (n_frames2, n_features)
        metric: 距离度量方式，默认使用欧氏距离
        normalize: 是否归一化（按对齐路径长度归一化）
    
    Returns:
        distance: DTW距离（归一化或原始）
        path: 对齐路径
    
    说明:
        - 原始DTW距离：累计的帧间距离总和，受轨迹长度影响
        - 归一化DTW距离：原始距离除以对齐路径长度，更适合比较不同长度的轨迹
    """
    distance, path = fastdtw(traj1, traj2, dist=euclidean)
    
    if normalize:
        # 按对齐路径长度归一化
        path_length = len(path)
        distance = distance / path_length
    
    return distance, path


def filter_demos(max_demo_id=65, max_length=350, min_length=0, 
                 include_ids=None, exclude_ids=None):
    """
    过滤并加载demo数据
    
    Args:
        max_demo_id: 最大demo编号
        max_length: 过滤掉长度超过此值的demo
        min_length: 过滤掉长度小于此值的demo
        include_ids: 只包含这些ID的demo列表（优先级最高）
        exclude_ids: 排除这些ID的demo列表
    
    Returns:
        valid_demos: 有效的demo数据列表
        valid_demo_ids: 有效的demo ID列表
    """
    valid_demos = []
    valid_demo_ids = []
    
    print("Loading and filtering demonstrations...")
    
    # 如果指定了include_ids，只加载这些demo
    if include_ids is not None:
        demo_range = include_ids
    else:
        demo_range = range(max_demo_id)
    
    for demo_id in tqdm(demo_range):
        # 检查是否在排除列表中
        if exclude_ids is not None and demo_id in exclude_ids:
            continue
            
        try:
            state = read_demo_kinematics_state(needle_dir_path, demo_id)
            demo_length = state.shape[0]
            
            # 检查长度是否满足条件
            if min_length <= demo_length <= max_length:
                valid_demos.append(state)
                valid_demo_ids.append(demo_id)
        except Exception as e:
            print(f"Warning: Failed to load demo {demo_id}: {e}")
            continue
    
    print(f"Loaded {len(valid_demos)} valid demonstrations")
    return valid_demos, valid_demo_ids


def compute_dtw_matrix(max_demo_id=65, max_length=350, min_length=0,
                      include_ids=None, exclude_ids=None, normalize=False, 
                      save_path=None):
    """
    计算所有demo之间的DTW距离矩阵
    
    Args:
        max_demo_id: 最大demo编号
        max_length: 过滤掉长度超过此值的demo
        min_length: 过滤掉长度小于此值的demo
        include_ids: 只包含这些ID的demo列表（优先级最高）
        exclude_ids: 排除这些ID的demo列表
        normalize: 是否归一化DTW距离（推荐：True用于比较不同长度轨迹）
        save_path: 保存路径，如果为None则不保存
    
    Returns:
        dtw_matrix: DTW距离矩阵
        valid_demo_ids: 有效的demo ID列表
        
    说明:
        normalize=False: 原始DTW距离，适合长度相似的轨迹
        normalize=True: 归一化DTW距离（按路径长度），适合不同长度的轨迹比较
    """
    # 使用filter_demos函数过滤demo
    valid_demos, valid_demo_ids = filter_demos(
        max_demo_id=max_demo_id,
        max_length=max_length,
        min_length=min_length,
        include_ids=include_ids,
        exclude_ids=exclude_ids
    )
    
    n_demos = len(valid_demos)
    
    # 初始化DTW距离矩阵
    dtw_matrix = np.zeros((n_demos, n_demos))
    
    # 计算所有demo对之间的DTW距离
    normalize_str = "normalized " if normalize else ""
    print(f"Computing {normalize_str}DTW distances...")
    for i in tqdm(range(n_demos)):
        for j in range(i, n_demos):
            if i == j:
                dtw_matrix[i, j] = 0.0
            else:
                distance, _ = compute_dtw_distance(valid_demos[i], valid_demos[j], 
                                                   normalize=normalize)
                dtw_matrix[i, j] = distance
                dtw_matrix[j, i] = distance  # 对称矩阵
    
    # 保存结果
    if save_path is not None:
        result = {
            'dtw_matrix': dtw_matrix,
            'valid_demo_ids': valid_demo_ids,
            'n_demos': n_demos,
            'normalized': normalize
        }
        with open(save_path, 'wb') as f:
            pickle.dump(result, f)
        print(f"Results saved to {save_path}")
    
    return dtw_matrix, valid_demo_ids


def visualize_dtw_matrix(dtw_matrix, valid_demo_ids, save_fig_path=None):
    """
    可视化DTW距离矩阵
    
    Args:
        dtw_matrix: DTW距离矩阵
        valid_demo_ids: 有效的demo ID列表
        save_fig_path: 图片保存路径
    """
    plt.figure(figsize=(12, 10))
    
    # 使用热力图可视化
    sns.heatmap(dtw_matrix, 
                xticklabels=valid_demo_ids, 
                yticklabels=valid_demo_ids,
                cmap='viridis',
                cbar_kws={'label': 'DTW Distance'})
    
    plt.title('DTW Distance Matrix between Demonstrations', fontsize=14, pad=20)
    plt.xlabel('Demo ID', fontsize=12)
    plt.ylabel('Demo ID', fontsize=12)
    plt.tight_layout()
    
    if save_fig_path is not None:
        plt.savefig(save_fig_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to {save_fig_path}")
    
    plt.show()


def analyze_dtw_statistics(dtw_matrix, valid_demo_ids):
    """
    分析DTW距离的统计信息
    
    Args:
        dtw_matrix: DTW距离矩阵
        valid_demo_ids: 有效的demo ID列表
    """
    # 提取上三角矩阵（不包括对角线）
    upper_triangle = dtw_matrix[np.triu_indices_from(dtw_matrix, k=1)]
    
    print("\n=== DTW Distance Statistics ===")
    print(f"Total number of valid demos: {len(valid_demo_ids)}")
    print(f"Total number of pairwise distances: {len(upper_triangle)}")
    print(f"Mean DTW distance: {np.mean(upper_triangle):.2f}")
    print(f"Median DTW distance: {np.median(upper_triangle):.2f}")
    print(f"Std DTW distance: {np.std(upper_triangle):.2f}")
    print(f"Min DTW distance: {np.min(upper_triangle):.2f}")
    print(f"Max DTW distance: {np.max(upper_triangle):.2f}")
    
    # 找出最相似和最不相似的demo对
    min_idx = np.unravel_index(np.argmin(dtw_matrix + np.eye(len(dtw_matrix)) * 1e10), 
                                dtw_matrix.shape)
    max_idx = np.unravel_index(np.argmax(dtw_matrix), dtw_matrix.shape)
    
    print(f"\nMost similar demos: {valid_demo_ids[min_idx[0]]} and {valid_demo_ids[min_idx[1]]} "
          f"(distance: {dtw_matrix[min_idx]:.2f})")
    print(f"Most dissimilar demos: {valid_demo_ids[max_idx[0]]} and {valid_demo_ids[max_idx[1]]} "
          f"(distance: {dtw_matrix[max_idx]:.2f})")
    
    # 可视化距离分布
    plt.figure(figsize=(10, 6))
    plt.hist(upper_triangle, bins=50, alpha=0.7, color='skyblue', edgecolor='navy')
    plt.axvline(np.mean(upper_triangle), color='red', linestyle='--', 
                linewidth=2, label=f'Mean: {np.mean(upper_triangle):.2f}')
    plt.axvline(np.median(upper_triangle), color='green', linestyle='--', 
                linewidth=2, label=f'Median: {np.median(upper_triangle):.2f}')
    plt.xlabel('DTW Distance', fontsize=12)
    plt.ylabel('Frequency', fontsize=12)
    plt.title('Distribution of Pairwise DTW Distances', fontsize=14)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
    
    return upper_triangle


def find_k_nearest_neighbors(dtw_matrix, valid_demo_ids, demo_id, k=5):
    """
    找到指定demo的k个最近邻
    
    Args:
        dtw_matrix: DTW距离矩阵
        valid_demo_ids: 有效的demo ID列表
        demo_id: 查询的demo ID
        k: 最近邻的数量
    
    Returns:
        neighbors: k个最近邻的demo ID及其距离
    """
    if demo_id not in valid_demo_ids:
        print(f"Error: Demo {demo_id} not in valid demo list")
        return None
    
    idx = valid_demo_ids.index(demo_id)
    distances = dtw_matrix[idx, :]
    
    # 排除自己，找到k个最小距离
    sorted_indices = np.argsort(distances)[1:k+1]
    
    neighbors = [(valid_demo_ids[i], distances[i]) for i in sorted_indices]
    
    print(f"\nTop {k} nearest neighbors for demo {demo_id}:")
    for i, (neighbor_id, dist) in enumerate(neighbors, 1):
        print(f"  {i}. Demo {neighbor_id}: distance = {dist:.2f}")
    
    return neighbors


def compute_dtw_distance_for_subset(demo_ids, normalize=False, save_path=None):
    """
    仅计算指定demo子集之间的DTW距离
    
    Args:
        demo_ids: 要计算的demo ID列表
        normalize: 是否归一化DTW距离
        save_path: 保存路径，如果为None则不保存
    
    Returns:
        dtw_matrix: DTW距离矩阵
        demo_ids: demo ID列表
    """
    return compute_dtw_matrix(
        include_ids=demo_ids,
        normalize=normalize,
        save_path=save_path
    )


def get_demo_info(max_demo_id=65):
    """
    获取所有demo的信息（ID和长度）
    
    Args:
        max_demo_id: 最大demo编号
    
    Returns:
        demo_info: 包含demo ID和长度的字典列表
    """
    demo_info = []
    
    print("Collecting demo information...")
    for demo_id in tqdm(range(max_demo_id)):
        try:
            state = read_demo_kinematics_state(needle_dir_path, demo_id)
            demo_info.append({
                'id': demo_id,
                'length': state.shape[0],
                'features': state.shape[1]
            })
        except Exception as e:
            print(f"Warning: Failed to load demo {demo_id}: {e}")
            continue
    
    # 打印统计信息
    lengths = [info['length'] for info in demo_info]
    print(f"\n=== Demo Information Summary ===")
    print(f"Total demos: {len(demo_info)}")
    print(f"Length range: {min(lengths)} - {max(lengths)} frames")
    print(f"Mean length: {np.mean(lengths):.1f} frames")
    print(f"Median length: {np.median(lengths):.1f} frames")
    
    return demo_info


def filter_demos_by_length_range(min_length, max_length):
    """
    根据长度范围过滤demo
    
    Args:
        min_length: 最小长度
        max_length: 最大长度
    
    Returns:
        filtered_ids: 符合条件的demo ID列表
    """
    demo_info = get_demo_info()
    filtered_ids = [info['id'] for info in demo_info 
                   if min_length <= info['length'] <= max_length]
    
    print(f"\nFiltered {len(filtered_ids)} demos with length between {min_length} and {max_length}")
    print(f"Demo IDs: {filtered_ids}")
    
    return filtered_ids


def comprehensive_resampling_analysis(max_demo_id=65, exclude_ids=None, 
                                     max_length=350, normalize=False,
                                     similarity_threshold=None):
    """
    Comprehensive comparison analysis before and after resampling
    
    Includes:
    1. DTW distance distribution comparison before/after resampling
    2. Distance matrix heatmaps before/after resampling
    3. Confusion matrix (based on similarity threshold)
    4. Statistical significance test results
    
    Args:
        max_demo_id: Maximum demo ID
        exclude_ids: List of demo IDs to exclude
        max_length: Maximum length filter
        normalize: Whether to use normalized DTW distance (default: False, uses raw fastdtw)
        similarity_threshold: Similarity threshold (for confusion matrix), use median if None
    """
    print("\n" + "="*70)
    print("DTW Distance Comparison Analysis: Before vs After Resampling")
    print("="*70)
    
    # ========== 1. Load raw data and compute DTW distances ==========
    print("\n[1/4] Computing DTW distances before resampling...")
    valid_demos_raw = []
    valid_demo_ids = []
    
    demo_range = range(max_demo_id)
    for demo_id in tqdm(demo_range, desc="Loading raw data"):
        if exclude_ids is not None and demo_id in exclude_ids:
            continue
        try:
            state = read_demo_kinematics_state(needle_dir_path, demo_id)
            if state.shape[0] <= max_length:
                valid_demos_raw.append(state)
                valid_demo_ids.append(demo_id)
        except:
            continue
    
    n_demos = len(valid_demos_raw)
    print(f"Number of valid demos: {n_demos}")
    
    # Compute DTW distance matrix for raw data
    dtw_matrix_raw = np.zeros((n_demos, n_demos))
    for i in tqdm(range(n_demos), desc="Computing DTW (raw)"):
        for j in range(i+1, n_demos):
            distance, _ = compute_dtw_distance(valid_demos_raw[i], valid_demos_raw[j], 
                                              normalize=normalize)
            dtw_matrix_raw[i, j] = distance
            dtw_matrix_raw[j, i] = distance
    
    # ========== 2. Resample data and compute DTW distances ==========
    print("\n[2/4] Computing DTW distances after resampling...")
    valid_demos_resampled = []
    for demo in tqdm(valid_demos_raw, desc="Resampling data"):
        resampled = resample_bimanual_trajectory(demo, step_size=0.0015)
        valid_demos_resampled.append(resampled)
    
    # Compute DTW distance matrix for resampled data
    dtw_matrix_resampled = np.zeros((n_demos, n_demos))
    for i in tqdm(range(n_demos), desc="Computing DTW (resampled)"):
        for j in range(i+1, n_demos):
            distance, _ = compute_dtw_distance(valid_demos_resampled[i], 
                                              valid_demos_resampled[j], 
                                              normalize=normalize)
            dtw_matrix_resampled[i, j] = distance
            dtw_matrix_resampled[j, i] = distance
    
    # ========== 3. Statistical significance tests ==========
    print("\n[3/4] Performing statistical significance tests...")
    
    # Extract upper triangle (exclude diagonal)
    triu_indices = np.triu_indices(n_demos, k=1)
    distances_raw = dtw_matrix_raw[triu_indices]
    distances_resampled = dtw_matrix_resampled[triu_indices]
    
    # Paired t-test
    t_stat, t_pvalue = stats.ttest_rel(distances_raw, distances_resampled)
    
    # Wilcoxon signed-rank test (non-parametric)
    wilcoxon_stat, wilcoxon_pvalue = stats.wilcoxon(distances_raw, distances_resampled)
    
    # Compute effect size (Cohen's d)
    mean_diff = np.mean(distances_raw - distances_resampled)
    std_diff = np.std(distances_raw - distances_resampled)
    cohens_d = mean_diff / std_diff if std_diff > 0 else 0
    
    # ========== 4. Create confusion matrix (similarity judgment) ==========
    print("\n[4/4] Generating confusion matrix...")
    
    # Use median as threshold if not specified
    if similarity_threshold is None:
        similarity_threshold = np.median(distances_raw)
    
    # Convert distance matrices to similarity labels (0=similar, 1=dissimilar)
    labels_raw = (dtw_matrix_raw > similarity_threshold).astype(int)
    labels_resampled = (dtw_matrix_resampled > similarity_threshold).astype(int)
    
    # Extract upper triangle only (avoid duplication)
    labels_raw_flat = labels_raw[triu_indices]
    labels_resampled_flat = labels_resampled[triu_indices]
    
    # Compute confusion matrix
    conf_matrix = confusion_matrix(labels_raw_flat, labels_resampled_flat)
    
    # ========== 5. Comprehensive visualization ==========
    print("\n[5/5] Generating visualizations...")
    
    # Calculate correlation coefficient (needed for both figures)
    correlation = np.corrcoef(distances_raw, distances_resampled)[0, 1]
    
    # ==================== Figure 1: Visual Comparisons ====================
    fig1 = plt.figure(figsize=(20, 10))
    gs1 = fig1.add_gridspec(2, 4, hspace=0.25, wspace=0.3)
    
    # Subplot 1: Distance Distribution Before Resampling
    ax1 = fig1.add_subplot(gs1[0, 0])
    ax1.hist(distances_raw, bins=50, alpha=0.7, color='blue', edgecolor='navy', label='Before')
    ax1.axvline(np.mean(distances_raw), color='red', linestyle='--', linewidth=2, 
                label=f'Mean: {np.mean(distances_raw):.4f}')
    ax1.axvline(np.median(distances_raw), color='green', linestyle='--', linewidth=2,
                label=f'Median: {np.median(distances_raw):.4f}')
    ax1.set_xlabel('DTW Distance', fontsize=11)
    ax1.set_ylabel('Frequency', fontsize=11)
    ax1.set_title('Distribution Before Resampling', fontsize=12, fontweight='bold')
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.3)
    
    # Subplot 2: Distance Distribution After Resampling
    ax2 = fig1.add_subplot(gs1[0, 1])
    ax2.hist(distances_resampled, bins=50, alpha=0.7, color='orange', edgecolor='darkred', 
             label='After')
    ax2.axvline(np.mean(distances_resampled), color='red', linestyle='--', linewidth=2,
                label=f'Mean: {np.mean(distances_resampled):.4f}')
    ax2.axvline(np.median(distances_resampled), color='green', linestyle='--', linewidth=2,
                label=f'Median: {np.median(distances_resampled):.4f}')
    ax2.set_xlabel('DTW Distance', fontsize=11)
    ax2.set_ylabel('Frequency', fontsize=11)
    ax2.set_title('Distribution After Resampling', fontsize=12, fontweight='bold')
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)
    
    # Subplot 3: Overlaid Distribution Comparison
    ax3 = fig1.add_subplot(gs1[0, 2])
    ax3.hist(distances_raw, bins=50, alpha=0.5, color='blue', label='Before', density=True)
    ax3.hist(distances_resampled, bins=50, alpha=0.5, color='orange', label='After', density=True)
    ax3.set_xlabel('DTW Distance', fontsize=11)
    ax3.set_ylabel('Density', fontsize=11)
    ax3.set_title('Overlaid Distribution Comparison', fontsize=12, fontweight='bold')
    ax3.legend(fontsize=10)
    ax3.grid(True, alpha=0.3)
    
    # Subplot 4: Scatter Plot - Distance Correlation
    ax4 = fig1.add_subplot(gs1[0, 3])
    ax4.scatter(distances_raw, distances_resampled, alpha=0.5, s=10)
    
    # Add diagonal line
    min_val = min(distances_raw.min(), distances_resampled.min())
    max_val = max(distances_raw.max(), distances_resampled.max())
    ax4.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='y=x')
    
    # Add regression line
    z = np.polyfit(distances_raw, distances_resampled, 1)
    p = np.poly1d(z)
    ax4.plot(distances_raw, p(distances_raw), 'g-', linewidth=2, alpha=0.7,
             label=f'Fit: y={z[0]:.2f}x+{z[1]:.4f}')
    
    ax4.set_xlabel('Distance Before', fontsize=11)
    ax4.set_ylabel('Distance After', fontsize=11)
    ax4.set_title('Distance Correlation', fontsize=12, fontweight='bold')
    ax4.legend(fontsize=9)
    ax4.grid(True, alpha=0.3)
    
    ax4.text(0.05, 0.95, f'r = {correlation:.4f}', 
             transform=ax4.transAxes, fontsize=10,
             bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7))
    
    # Subplot 5: DTW Matrix Before Resampling
    ax5 = fig1.add_subplot(gs1[1, 0])
    im1 = ax5.imshow(dtw_matrix_raw, cmap='viridis', aspect='auto')
    ax5.set_title('DTW Matrix Before', fontsize=12, fontweight='bold')
    ax5.set_xlabel('Demo ID', fontsize=10)
    ax5.set_ylabel('Demo ID', fontsize=10)
    plt.colorbar(im1, ax=ax5, label='DTW Distance')
    
    # Subplot 6: DTW Matrix After Resampling
    ax6 = fig1.add_subplot(gs1[1, 1])
    im2 = ax6.imshow(dtw_matrix_resampled, cmap='viridis', aspect='auto')
    ax6.set_title('DTW Matrix After', fontsize=12, fontweight='bold')
    ax6.set_xlabel('Demo ID', fontsize=10)
    ax6.set_ylabel('Demo ID', fontsize=10)
    plt.colorbar(im2, ax=ax6, label='DTW Distance')
    
    # Subplot 7: Distance Change Matrix
    ax7 = fig1.add_subplot(gs1[1, 2])
    diff_matrix = dtw_matrix_resampled - dtw_matrix_raw
    im3 = ax7.imshow(diff_matrix, cmap='RdBu_r', aspect='auto', 
                     vmin=-np.abs(diff_matrix).max(), vmax=np.abs(diff_matrix).max())
    ax7.set_title('Distance Change (After - Before)', fontsize=12, fontweight='bold')
    ax7.set_xlabel('Demo ID', fontsize=10)
    ax7.set_ylabel('Demo ID', fontsize=10)
    plt.colorbar(im3, ax=ax7, label='Distance Change')
    
    # Subplot 8: Confusion Matrix
    ax8 = fig1.add_subplot(gs1[1, 3])
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', ax=ax8,
                xticklabels=['Similar', 'Dissimilar'], yticklabels=['Similar', 'Dissimilar'],
                cbar_kws={'label': 'Count'})
    ax8.set_xlabel('After Resampling', fontsize=11)
    ax8.set_ylabel('Before Resampling', fontsize=11)
    ax8.set_title(f'Confusion Matrix\n(Threshold={similarity_threshold:.4f})', 
                  fontsize=12, fontweight='bold')
    
    # Calculate accuracy
    accuracy = (conf_matrix[0,0] + conf_matrix[1,1]) / conf_matrix.sum()
    ax8.text(1.0, -0.15, f'Accuracy: {accuracy*100:.1f}%', 
             transform=ax8.transAxes, fontsize=10, ha='center',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # Main title for Figure 1
    fig1.suptitle('DTW Distance Comparison: Before vs After Resampling', 
                  fontsize=16, fontweight='bold', y=0.98)
    
    # ==================== Figure 2: Statistical Tests ====================
    fig2 = plt.figure(figsize=(16, 10))
    gs2 = fig2.add_gridspec(2, 2, hspace=0.3, wspace=0.3)
    
    # Subplot 1: Descriptive Statistics (Table format)
    ax1_fig2 = fig2.add_subplot(gs2[0, 0])
    ax1_fig2.axis('off')
    
    # Create descriptive statistics table
    desc_stats_data = [
        ['Metric', 'Before', 'After', 'Change'],
        ['Mean', f'{np.mean(distances_raw):.6f}', f'{np.mean(distances_resampled):.6f}', 
         f'{np.mean(distances_resampled) - np.mean(distances_raw):+.6f}'],
        ['Median', f'{np.median(distances_raw):.6f}', f'{np.median(distances_resampled):.6f}',
         f'{np.median(distances_resampled) - np.median(distances_raw):+.6f}'],
        ['Std Dev', f'{np.std(distances_raw):.6f}', f'{np.std(distances_resampled):.6f}',
         f'{np.std(distances_resampled) - np.std(distances_raw):+.6f}'],
        ['Min', f'{np.min(distances_raw):.6f}', f'{np.min(distances_resampled):.6f}',
         f'{np.min(distances_resampled) - np.min(distances_raw):+.6f}'],
        ['Max', f'{np.max(distances_raw):.6f}', f'{np.max(distances_resampled):.6f}',
         f'{np.max(distances_resampled) - np.max(distances_raw):+.6f}'],
    ]
    
    table = ax1_fig2.table(cellText=desc_stats_data, cellLoc='center', loc='center',
                           colWidths=[0.2, 0.25, 0.25, 0.3])
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2.5)
    
    # Style header row
    for i in range(4):
        table[(0, i)].set_facecolor('#4472C4')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    ax1_fig2.set_title('Descriptive Statistics', fontsize=14, fontweight='bold', pad=20)
    
    # Subplot 2: Statistical Test Results
    ax2_fig2 = fig2.add_subplot(gs2[0, 1])
    ax2_fig2.axis('off')
    
    sig_t = '***' if t_pvalue < 0.001 else ('**' if t_pvalue < 0.01 else ('*' if t_pvalue < 0.05 else 'ns'))
    sig_w = '***' if wilcoxon_pvalue < 0.001 else ('**' if wilcoxon_pvalue < 0.01 else ('*' if wilcoxon_pvalue < 0.05 else 'ns'))
    
    stats_text = f"""
    === Statistical Significance Tests ===
    
    Sample Size: {len(distances_raw)} demo pairs
    
    [Paired t-test]
    t-statistic = {t_stat:.4f}
    p-value = {t_pvalue:.6f} {sig_t}
    Conclusion: {"Significant" if t_pvalue < 0.05 else "Not Significant"}
    
    [Wilcoxon Signed-Rank Test]
    Statistic = {wilcoxon_stat:.4f}
    p-value = {wilcoxon_pvalue:.6f} {sig_w}
    Conclusion: {"Significant" if wilcoxon_pvalue < 0.05 else "Not Significant"}
    
    [Effect Size]
    Cohen's d = {cohens_d:.4f}
    Magnitude: {get_effect_size_label(cohens_d)}
    
    [Correlation]
    Pearson's r = {correlation:.4f}
    r² = {correlation**2:.4f}
    
    [Significance Levels]
    * p < 0.05: Significant
    ** p < 0.01: Very Significant
    *** p < 0.001: Extremely Significant
    ns: Not Significant
    """
    
    ax2_fig2.text(0.05, 0.95, stats_text, transform=ax2_fig2.transAxes,
                  fontsize=10, verticalalignment='top', fontfamily='monospace',
                  bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
    ax2_fig2.set_title('Hypothesis Tests', fontsize=14, fontweight='bold', pad=20)
    
    # Subplot 3: Distribution of Differences
    ax3_fig2 = fig2.add_subplot(gs2[1, 0])
    differences = distances_resampled - distances_raw
    ax3_fig2.hist(differences, bins=50, alpha=0.7, color='purple', edgecolor='black')
    ax3_fig2.axvline(0, color='red', linestyle='--', linewidth=2, label='No Change')
    ax3_fig2.axvline(np.mean(differences), color='green', linestyle='--', linewidth=2,
                     label=f'Mean Diff: {np.mean(differences):.6f}')
    ax3_fig2.set_xlabel('Distance Change (After - Before)', fontsize=11)
    ax3_fig2.set_ylabel('Frequency', fontsize=11)
    ax3_fig2.set_title('Distribution of Distance Changes', fontsize=12, fontweight='bold')
    ax3_fig2.legend(fontsize=10)
    ax3_fig2.grid(True, alpha=0.3)
    
    # Subplot 4: Box Plot Comparison
    ax4_fig2 = fig2.add_subplot(gs2[1, 1])
    box_data = [distances_raw, distances_resampled]
    bp = ax4_fig2.boxplot(box_data, labels=['Before', 'After'], patch_artist=True,
                           showmeans=True, meanline=True)
    
    # Color the boxes
    colors = ['lightblue', 'lightcoral']
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
    
    ax4_fig2.set_ylabel('DTW Distance', fontsize=11)
    ax4_fig2.set_title('Box Plot Comparison', fontsize=12, fontweight='bold')
    ax4_fig2.grid(True, alpha=0.3, axis='y')
    
    # Add statistical annotation
    y_max = max(distances_raw.max(), distances_resampled.max())
    sig_text = sig_t if sig_t != 'ns' else 'ns'
    ax4_fig2.text(1.5, y_max * 1.05, sig_text, ha='center', fontsize=14, fontweight='bold',
                  bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7))
    
    # Main title for Figure 2
    fig2.suptitle('Statistical Analysis: Resampling Effect on DTW Distance', 
                  fontsize=16, fontweight='bold', y=0.98)
    
    plt.show()
    
    # Print summary
    print("\n" + "="*70)
    print("Analysis completed!")
    print(f"Statistical test: p = {t_pvalue:.6f} ({'Significant' if t_pvalue < 0.05 else 'Not Significant'})")
    print(f"Effect size: Cohen's d = {cohens_d:.4f} ({get_effect_size_label(cohens_d)})")
    print(f"Correlation: r = {correlation:.4f}")
    print(f"Similarity judgment consistency: {accuracy*100:.1f}%")
    print("="*70 + "\n")
    
    return {
        'dtw_raw': dtw_matrix_raw,
        'dtw_resampled': dtw_matrix_resampled,
        'demo_ids': valid_demo_ids,
        'statistics': {
            't_stat': t_stat,
            't_pvalue': t_pvalue,
            'wilcoxon_stat': wilcoxon_stat,
            'wilcoxon_pvalue': wilcoxon_pvalue,
            'cohens_d': cohens_d,
            'correlation': correlation
        },
        'confusion_matrix': conf_matrix,
        'accuracy': accuracy
    }


def get_effect_size_label(cohens_d):
    """Get Cohen's d effect size label"""
    abs_d = abs(cohens_d)
    if abs_d < 0.2:
        return "Negligible"
    elif abs_d < 0.5:
        return "Small"
    elif abs_d < 0.8:
        return "Medium"
    else:
        return "Large"


def compare_normalized_vs_raw(demo_ids=None, max_demo_id=10):
    """
    比较归一化和原始DTW距离的差异
    
    Args:
        demo_ids: 要比较的demo ID列表，如果为None则使用前max_demo_id个demo
        max_demo_id: 当demo_ids为None时，使用的最大demo编号
    
    Returns:
        comparison_results: 包含两种距离的比较结果
    """
    print("\n" + "="*60)
    print("比较归一化 vs 原始DTW距离")
    print("="*60)
    
    # 加载demo
    if demo_ids is None:
        valid_demos, valid_demo_ids = filter_demos(max_demo_id=max_demo_id, max_length=350)
    else:
        valid_demos, valid_demo_ids = filter_demos(include_ids=demo_ids, max_length=350)
    
    # 获取demo长度
    demo_lengths = [len(demo) for demo in valid_demos]
    
    print(f"\n加载了 {len(valid_demos)} 个demo")
    print(f"长度范围: {min(demo_lengths)} - {max(demo_lengths)} 帧")
    
    # 计算原始DTW距离
    print("\n计算原始DTW距离...")
    raw_distances = []
    normalized_distances = []
    
    # 选择几个代表性的demo对进行比较
    n_demos = len(valid_demos)
    comparisons = []
    
    for i in range(min(5, n_demos)):
        for j in range(i+1, min(5, n_demos)):
            raw_dist, path = compute_dtw_distance(valid_demos[i], valid_demos[j], 
                                                  normalize=False)
            norm_dist, _ = compute_dtw_distance(valid_demos[i], valid_demos[j], 
                                               normalize=True)
            
            comparisons.append({
                'demo_i': valid_demo_ids[i],
                'demo_j': valid_demo_ids[j],
                'length_i': len(valid_demos[i]),
                'length_j': len(valid_demos[j]),
                'raw_distance': raw_dist,
                'normalized_distance': norm_dist,
                'path_length': len(path)
            })
    
    # 打印比较结果
    print("\n" + "="*90)
    print(f"{'Demo对':<12} {'长度(i,j)':<15} {'原始距离':<15} {'归一化距离':<15} {'路径长度':<12}")
    print("="*90)
    
    for comp in comparisons:
        demo_pair = f"{comp['demo_i']}-{comp['demo_j']}"
        lengths = f"({comp['length_i']}, {comp['length_j']})"
        print(f"{demo_pair:<12} {lengths:<15} {comp['raw_distance']:<15.2f} "
              f"{comp['normalized_distance']:<15.4f} {comp['path_length']:<12}")
    
    print("="*90)
    
    # 分析说明
    print("\n【距离含义说明】")
    print("1. 原始DTW距离：")
    print("   - 表示对齐路径上所有帧间距离的累加和")
    print("   - 数值大小受轨迹长度影响，长轨迹通常有更大的距离")
    print("   - 适用于比较长度相近的轨迹")
    print("\n2. 归一化DTW距离：")
    print("   - 原始距离除以对齐路径长度")
    print("   - 表示平均每一步的距离，消除了长度影响")
    print("   - 适用于比较不同长度的轨迹，更具可比性")
    print("\n【建议】")
    print("- 如果demo长度差异较大 → 使用归一化距离 (normalize=True)")
    print("- 如果demo长度相近 → 两种都可以，原始距离更直观")
    
    return comparisons


if __name__ == '__main__':
    # ============== Main: Comprehensive Resampling Analysis ==============
    # Display all analysis results in a single figure (no saving)
    print("\n" + "="*70)
    print("Starting Comprehensive DTW Distance Analysis")
    print("="*70)
    
    results = comprehensive_resampling_analysis(
        max_demo_id=65,
        exclude_ids=[46],  # Exclude anomalous demos
        max_length=350,
        normalize=False,  # Use raw DTW distance (not normalized)
        similarity_threshold=None  # Use automatic threshold (median)
    )
    
    print("\nAnalysis results generated!")
    print(f"- DTW distance matrix shape: {results['dtw_raw'].shape}")
    print(f"- Number of valid demos: {len(results['demo_ids'])}")
    print(f"- Correlation coefficient: {results['statistics']['correlation']:.4f}")
    print(f"- Consistency: {results['accuracy']*100:.1f}%")
    
    # ============== Optional: Additional Analyses ==============
    # Uncomment to run other analyses
    
    # # Example 1: Compare normalized vs raw DTW distances
    # print("\n" + "="*60)
    # print("Example: Compare normalized vs raw DTW distances")
    # print("="*60)
    # compare_normalized_vs_raw(max_demo_id=10)
    
    # # Example 2: Find nearest neighbors
    # print("\n" + "="*60)
    # print("Example: Find nearest neighbor demos")
    # print("="*60)
    # if 0 in results['demo_ids']:
    #     find_k_nearest_neighbors(results['dtw_resampled'], 
    #                             results['demo_ids'], demo_id=0, k=5)

