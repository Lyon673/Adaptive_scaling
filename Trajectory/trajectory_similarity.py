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
from load_data import read_demo_kinematics_state, needle_dir_path
from scipy.spatial.distance import euclidean
from fastdtw import fastdtw
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import pickle


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
    # 设置保存路径
    current_dir = os.path.dirname(__file__)
    
    # ============== 示例1: 比较归一化与原始DTW距离 ==============
    print("\n" + "="*60)
    print("示例1: 比较归一化与原始DTW距离的差异")
    print("="*60)
    compare_normalized_vs_raw(max_demo_id=10)
    
    # ============== 示例2: 计算原始DTW距离矩阵 ==============
    print("\n" + "="*60)
    print("示例2: 计算原始DTW距离矩阵")
    print("="*60)
    save_path_raw = os.path.join(current_dir, 'dtw_results_raw.pkl')
    fig_path_raw = os.path.join(current_dir, 'dtw_distance_matrix_raw.png')
    
    dtw_matrix_raw, valid_demo_ids = compute_dtw_matrix(
        max_demo_id=65, 
        max_length=350, 
        normalize=False,
        exclude_ids=[46],
        save_path=save_path_raw
    )
    analyze_dtw_statistics(dtw_matrix_raw, valid_demo_ids)
    visualize_dtw_matrix(dtw_matrix_raw, valid_demo_ids, save_fig_path=fig_path_raw)
    
    # ============== 示例3: 计算归一化DTW距离矩阵 ==============
    print("\n" + "="*60)
    print("示例3: 计算归一化DTW距离矩阵（推荐）")
    print("="*60)
    save_path_norm = os.path.join(current_dir, 'dtw_results_normalized.pkl')
    fig_path_norm = os.path.join(current_dir, 'dtw_distance_matrix_normalized.png')
    
    dtw_matrix_norm, valid_demo_ids = compute_dtw_matrix(
        max_demo_id=65, 
        max_length=350, 
        normalize=True,  # 使用归一化
        exclude_ids=[46],
        save_path=save_path_norm
    )
    analyze_dtw_statistics(dtw_matrix_norm, valid_demo_ids)
    visualize_dtw_matrix(dtw_matrix_norm, valid_demo_ids, save_fig_path=fig_path_norm)
    
    # ============== 示例4: 找到最近邻 ==============
    print("\n" + "="*60)
    print("示例4: 查找最近邻demo（基于归一化距离）")
    print("="*60)
    if 0 in valid_demo_ids:
        find_k_nearest_neighbors(dtw_matrix_norm, valid_demo_ids, demo_id=0, k=5)
    
    print("\n" + "="*60)
    print("所有计算完成！")
    print("="*60)

