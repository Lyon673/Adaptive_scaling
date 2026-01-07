import numpy as np
import os

def calculate_gracefulness(data):
    # 计算λ(t)，即位置向量
    positions = data[:, :3]
    
    # 计算速度向量 (一阶导数)
    velocities = np.gradient(positions, axis=0)
    
    # 计算加速度向量 (二阶导数)
    accelerations = np.gradient(velocities, axis=0)
    
    # 计算曲率 κ
    # κ = ||λ'(t) × λ''(t)|| / ||λ'(t)||^3
    cross_products = np.cross(velocities, accelerations)
    numerator = np.linalg.norm(cross_products, axis=1)
    denominator = np.power(np.linalg.norm(velocities, axis=1), 3)
    
    # 避免除以0
    denominator = np.where(denominator == 0, np.inf, denominator)
    curvature = numerator / denominator
    
    # 计算G值 (使用log10的中位数)
    G = np.median(np.log10(curvature + 1e-10))  # 添加小量避免log(0)
    
    return G

def calculate_smoothness(data):
    # 计算位置向量的三阶导数
    positions = data[:, :3]
    first_deriv = np.gradient(positions, axis=0)
    second_deriv = np.gradient(first_deriv, axis=0)
    third_deriv = np.gradient(second_deriv, axis=0)
    
    # 计算三阶导数的平方范数
    jerk_squared = np.sum(np.square(third_deriv), axis=1)
    
    # 计算时间间隔
    time = data[:, 3]
    dt = time[1] - time[0]
    
    # 计算积分（使用simpson's rule积分法则）
    integral = np.trapz(jerk_squared, dx=dt)
    
    # 根据公式中的参数计算
    duration = time[-1] - time[0]
    peak_velocity = np.max(np.linalg.norm(first_deriv, axis=1))
    
    # 计算 φ
    phi = (np.power(duration, 5) / np.square(peak_velocity)) * integral
    
    # 计算S值
    S = np.median(np.log10(phi + 1e-10))  # 添加小量避免log(0)
    
    return S

def calculate_metrics(data):
    G = calculate_gracefulness(data)
    S = calculate_smoothness(data)
    return G, S


def calculate_distribution(data, return_stats=True):
    """
    计算 curvature 和 phi 的完整分布，而不是中值
    
    Parameters:
    -----------
    data : ndarray
        形状为 (n, 4) 的数组，包含 [x, y, z, time]
    return_stats : bool, default True
        是否返回统计量
        
    Returns:
    --------
    dict : 包含以下内容的字典:
        - 'curvature': curvature 数组
        - 'log_curvature': log10(curvature) 数组
        - 'phi': phi 数组 (每个时间点的 jerk 贡献)
        - 'log_phi': log10(phi) 数组
        如果 return_stats=True，还包含:
        - 'curvature_stats': curvature 的统计量
        - 'phi_stats': phi 的统计量
    """
    # ========== 计算 Curvature ==========
    positions = data[:, :3]
    
    # 计算速度向量 (一阶导数)
    velocities = np.gradient(positions, axis=0)
    
    # 计算加速度向量 (二阶导数)
    accelerations = np.gradient(velocities, axis=0)
    
    # 计算曲率 κ = ||λ'(t) × λ''(t)|| / ||λ'(t)||^3
    cross_products = np.cross(velocities, accelerations)
    numerator = np.linalg.norm(cross_products, axis=1)
    denominator = np.power(np.linalg.norm(velocities, axis=1), 3)
    
    # 避免除以0
    denominator = np.where(denominator == 0, np.inf, denominator)
    curvature = numerator / denominator
    
    # 计算 log10(curvature)
    log_curvature = np.log10(curvature + 1e-10)
    
    # ========== 计算 Phi (Jerk-based) ==========
    # 计算三阶导数 (jerk)
    third_deriv = np.gradient(accelerations, axis=0)
    
    # 计算每个时间点的 jerk 平方范数
    jerk_squared = np.sum(np.square(third_deriv), axis=1)
    
    # 计算时间参数
    time = data[:, 3]
    duration = time[-1] - time[0]
    peak_velocity = np.max(np.linalg.norm(velocities, axis=1))
    
    # 计算每个时间点的 phi 值 (归一化的 jerk)
    # phi(t) = (duration^5 / peak_velocity^2) * jerk^2(t)
    if peak_velocity > 1e-10:
        phi = (np.power(duration, 5) / np.square(peak_velocity)) * jerk_squared
    else:
        phi = jerk_squared
    
    # 计算 log10(phi)
    log_phi = np.log10(phi + 1e-10)
    
    # 构建结果字典
    result = {
        'curvature': curvature,
        'log_curvature': log_curvature,
        'phi': phi,
        'log_phi': log_phi,
        'time': time
    }
    
    if return_stats:
        # 计算统计量
        def compute_stats(arr, name):
            # 过滤无穷值和 NaN
            valid_arr = arr[np.isfinite(arr)]
            if len(valid_arr) == 0:
                return {f'{name}_mean': np.nan, f'{name}_std': np.nan, 
                        f'{name}_median': np.nan, f'{name}_min': np.nan, 
                        f'{name}_max': np.nan, f'{name}_p25': np.nan, 
                        f'{name}_p75': np.nan, f'{name}_p5': np.nan, 
                        f'{name}_p95': np.nan, f'{name}_n_valid': 0}
            
            return {
                f'{name}_mean': np.mean(valid_arr),
                f'{name}_std': np.std(valid_arr),
                f'{name}_median': np.median(valid_arr),
                f'{name}_min': np.min(valid_arr),
                f'{name}_max': np.max(valid_arr),
                f'{name}_p25': np.percentile(valid_arr, 25),
                f'{name}_p75': np.percentile(valid_arr, 75),
                f'{name}_p5': np.percentile(valid_arr, 5),
                f'{name}_p95': np.percentile(valid_arr, 95),
                f'{name}_n_valid': len(valid_arr)
            }
        
        result['curvature_stats'] = compute_stats(curvature, 'curvature')
        result['log_curvature_stats'] = compute_stats(log_curvature, 'log_curvature')
        result['phi_stats'] = compute_stats(phi, 'phi')
        result['log_phi_stats'] = compute_stats(log_phi, 'log_phi')
    
    return result


def visualize_distribution(data_dir=None, use_left=True, use_right=True, save_fig=False):
    """
    可视化 curvature 和 phi 的分布
    
    Parameters:
    -----------
    data_dir : str, optional
        指定数据目录路径，如果为None则自动使用最新的数据目录
    use_left : bool, default True
        使用左PSM数据
    use_right : bool, default False
        使用右PSM数据
    save_fig : bool, default False
        是否保存图片
        
    Returns:
    --------
    dict : 包含分布数据和统计量的字典
    """
    import matplotlib.pyplot as plt
    
    current_file_path = os.path.abspath(__file__)
    current_dir = os.path.dirname(current_file_path)
    
    # 确定数据目录
    if data_dir is None:
        data_base_dir = os.path.join(current_dir, 'data')
        latest_dir = get_latest_data_dir(data_base_dir)
        print(f"Using latest data directory: {os.path.basename(latest_dir)}")
    else:
        latest_dir = os.path.join(current_dir, 'data', data_dir)
    
    results = {}
    
    def load_and_process(position_path, name):
        """加载并处理数据"""
        if not os.path.exists(position_path):
            print(f"文件不存在: {position_path}")
            return None
        
        positions = np.load(position_path, allow_pickle=True)
        
        if positions.shape[1] >= 4:
            pos = positions[:, :3]
            timestamps = positions[:, 3]
            
            # 转换时间戳
            if timestamps[0] > 1e9:
                timestamps = timestamps - timestamps[0]
                if timestamps[-1] > 1e6:
                    timestamps = timestamps / 1e6
            else:
                timestamps = timestamps - timestamps[0]
            
            data = np.column_stack([pos, timestamps])
            return calculate_distribution(data, return_stats=True)
        return None
    
    # 加载数据
    if use_left:
        left_path = os.path.join(latest_dir, 'Lpsm_position.npy')
        results['left'] = load_and_process(left_path, 'Left PSM')
    
    if use_right:
        right_path = os.path.join(latest_dir, 'Rpsm_position.npy')
        results['right'] = load_and_process(right_path, 'Right PSM')
    
    # 创建可视化
    n_plots = sum([use_left, use_right])
    if n_plots == 0:
        print("请至少选择一个 PSM 数据")
        return results
    
    fig, axes = plt.subplots(n_plots, 4, figsize=(20, 5 * n_plots))
    if n_plots == 1:
        axes = axes.reshape(1, -1)
    
    plot_idx = 0
    for side, data in results.items():
        if data is None:
            continue
        
        # 1. Curvature 分布 (直方图)
        ax = axes[plot_idx, 0]
        valid_curv = data['curvature'][np.isfinite(data['curvature'])]
        ax.hist(valid_curv, bins=50, color='#FF6B6B', alpha=0.7, edgecolor='black')
        ax.axvline(np.median(valid_curv), color='red', linestyle='--', linewidth=2, label=f'Median: {np.median(valid_curv):.4f}')
        ax.set_xlabel('Curvature (κ)')
        ax.set_ylabel('Frequency')
        ax.set_title(f'{side.upper()} PSM - Curvature Distribution')
        ax.legend()
        ax.grid(alpha=0.3)
        
        # 2. Log Curvature 分布
        ax = axes[plot_idx, 1]
        valid_log_curv = data['log_curvature'][np.isfinite(data['log_curvature'])]
        ax.hist(valid_log_curv, bins=50, color='#FF6B6B', alpha=0.7, edgecolor='black')
        ax.axvline(np.median(valid_log_curv), color='red', linestyle='--', linewidth=2, label=f'Median (G): {np.median(valid_log_curv):.4f}')
        ax.set_xlabel('log₁₀(Curvature)')
        ax.set_ylabel('Frequency')
        ax.set_title(f'{side.upper()} PSM - Log Curvature Distribution')
        ax.legend()
        ax.grid(alpha=0.3)
        
        # 3. Phi 分布 (直方图)
        ax = axes[plot_idx, 2]
        valid_phi = data['phi'][np.isfinite(data['phi'])]
        ax.hist(valid_phi, bins=50, color='#45B7D1', alpha=0.7, edgecolor='black')
        ax.axvline(np.median(valid_phi), color='red', linestyle='--', linewidth=2, label=f'Median: {np.median(valid_phi):.2e}')
        ax.set_xlabel('Phi (φ)')
        ax.set_ylabel('Frequency')
        ax.set_title(f'{side.upper()} PSM - Phi Distribution')
        ax.legend()
        ax.grid(alpha=0.3)
        
        # 4. Log Phi 分布
        ax = axes[plot_idx, 3]
        valid_log_phi = data['log_phi'][np.isfinite(data['log_phi'])]
        ax.hist(valid_log_phi, bins=50, color='#45B7D1', alpha=0.7, edgecolor='black')
        ax.axvline(np.median(valid_log_phi), color='red', linestyle='--', linewidth=2, label=f'Median (S): {np.median(valid_log_phi):.4f}')
        ax.set_xlabel('log₁₀(Phi)')
        ax.set_ylabel('Frequency')
        ax.set_title(f'{side.upper()} PSM - Log Phi Distribution')
        ax.legend()
        ax.grid(alpha=0.3)
        
        plot_idx += 1
    
    plt.tight_layout()
    
    if save_fig:
        save_path = os.path.join(latest_dir, 'curvature_phi_distribution.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"图片已保存到: {save_path}")
    
    plt.show()
    
    # 打印统计量
    print("\n" + "="*80)
    print("Distribution Statistics")
    print("="*80)
    
    for side, data in results.items():
        if data is None:
            continue
        
        print(f"\n{side.upper()} PSM:")
        print("-" * 60)
        
        print("\nCurvature (κ):")
        stats = data['curvature_stats']
        print(f"  Mean:   {stats['curvature_mean']:.6f}")
        print(f"  Std:    {stats['curvature_std']:.6f}")
        print(f"  Median: {stats['curvature_median']:.6f}")
        print(f"  Range:  [{stats['curvature_min']:.6f}, {stats['curvature_max']:.6f}]")
        print(f"  IQR:    [{stats['curvature_p25']:.6f}, {stats['curvature_p75']:.6f}]")
        print(f"  90%:    [{stats['curvature_p5']:.6f}, {stats['curvature_p95']:.6f}]")
        
        print("\nLog Curvature (G value):")
        stats = data['log_curvature_stats']
        print(f"  Mean:   {stats['log_curvature_mean']:.6f}")
        print(f"  Std:    {stats['log_curvature_std']:.6f}")
        print(f"  Median (G): {stats['log_curvature_median']:.6f}")
        print(f"  Range:  [{stats['log_curvature_min']:.6f}, {stats['log_curvature_max']:.6f}]")
        
        print("\nPhi (φ):")
        stats = data['phi_stats']
        print(f"  Mean:   {stats['phi_mean']:.6e}")
        print(f"  Std:    {stats['phi_std']:.6e}")
        print(f"  Median: {stats['phi_median']:.6e}")
        print(f"  Range:  [{stats['phi_min']:.6e}, {stats['phi_max']:.6e}]")
        
        print("\nLog Phi (S value):")
        stats = data['log_phi_stats']
        print(f"  Mean:   {stats['log_phi_mean']:.6f}")
        print(f"  Std:    {stats['log_phi_std']:.6f}")
        print(f"  Median (S): {stats['log_phi_median']:.6f}")
        print(f"  Range:  [{stats['log_phi_min']:.6f}, {stats['log_phi_max']:.6f}]")
    
    return results

def get_latest_data_dir(data_dir):
    """
    获取data文件夹下最新的数据目录
    按目录名中的数字前缀排序，返回数字最大的目录
    """
    if not os.path.exists(data_dir):
        raise FileNotFoundError(f"数据目录不存在: {data_dir}")
    
    # 获取所有子目录
    subdirs = [d for d in os.listdir(data_dir) 
               if os.path.isdir(os.path.join(data_dir, d)) and d.split('_')[0].isdigit()]
    
    if not subdirs:
        raise FileNotFoundError(f"在 {data_dir} 中未找到数据子目录")
    
    # 按目录名开头的数字排序
    subdirs.sort(key=lambda x: int(x.split('_')[0]), reverse=True)
    latest_dir = os.path.join(data_dir, subdirs[0])
    
    return latest_dir

def cal_GS(data_dir=None, use_left=True, use_right=True):
    """
    计算gracefulness和smoothness指标
    
    Parameters:
    -----------
    data_dir : str, optional
        指定数据目录路径，如果为None则自动使用最新的数据目录
    use_left : bool, default True
        如果为True使用左PSM数据(Lpsm_position)
    use_right : bool, default False
        如果为True使用右PSM数据(Rpsm_position)
        如果use_left和use_right都为True，则分别计算并返回两组值
    """
    current_file_path = os.path.abspath(__file__)
    current_dir = os.path.dirname(current_file_path)
    
    # 确定数据目录
    if data_dir is None:
        data_base_dir = os.path.join(current_dir, 'data')
        latest_dir = get_latest_data_dir(data_base_dir)
        print(f"Using latest data directory: {os.path.basename(latest_dir)}")
    else:
        latest_dir = os.path.join(current_dir, 'data', data_dir)
    
    results = {}
    
    # 加载左PSM数据
    if use_left:
        left_position_path = os.path.join(latest_dir, 'Lpsm_position.npy')
        if not os.path.exists(left_position_path):
            raise FileNotFoundError(f"左PSM位置数据文件不存在: {left_position_path}")
        
        left_positions = np.load(left_position_path, allow_pickle=True)
        
        # 提取位置和时间数据
        # left_positions形状应该是 (n, 4): [x, y, z, timestamp]
        if left_positions.shape[1] >= 4:
            positions = left_positions[:, :3]  # 前3列是位置
            timestamps = left_positions[:, 3]  # 第4列是时间戳
            
            # 将绝对时间戳转换为相对时间（从0开始）
            if timestamps[0] > 1e9:  # 如果是Unix时间戳（很大的数字）
                # 转换为相对时间（秒）
                timestamps = timestamps - timestamps[0]
                # 如果时间戳还是很大，可能是微秒，需要转换
                if timestamps[-1] > 1e6:
                    timestamps = timestamps / 1e6  # 转换为秒
            else:
                # 已经是相对时间，只需从0开始
                timestamps = timestamps - timestamps[0]
            
            # 组合位置和时间数据: [x, y, z, time]
            left_data = np.column_stack([positions, timestamps])
            
            G_left, S_left = calculate_metrics(left_data)
            results['left'] = {'G': G_left, 'S': S_left}
            #print(f"Left PSM - Gracefulness: {G_left:.6f}, Smoothness: {S_left:.6f}")
        else:
            print(f"警告: 左PSM数据格式不正确，期望4列，实际{left_positions.shape[1]}列")
    
    # 加载右PSM数据
    if use_right:
        right_position_path = os.path.join(latest_dir, 'Rpsm_position.npy')
        if not os.path.exists(right_position_path):
            raise FileNotFoundError(f"右PSM位置数据文件不存在: {right_position_path}")
        
        right_positions = np.load(right_position_path, allow_pickle=True)
        
        # 提取位置和时间数据
        if right_positions.shape[1] >= 4:
            positions = right_positions[:, :3]
            timestamps = right_positions[:, 3]
            
            # 将绝对时间戳转换为相对时间
            if timestamps[0] > 1e9:
                timestamps = timestamps - timestamps[0]
                if timestamps[-1] > 1e6:
                    timestamps = timestamps / 1e6
            else:
                timestamps = timestamps - timestamps[0]
            
            right_data = np.column_stack([positions, timestamps])
            
            G_right, S_right = calculate_metrics(right_data)
            results['right'] = {'G': G_right, 'S': S_right}
            print(f"Right PSM - Gracefulness: {G_right:.6f}, Smoothness: {S_right:.6f}")
        else:
            print(f"警告: 右PSM数据格式不正确，期望4列，实际{right_positions.shape[1]}列")
    
    # 返回结果
    if use_left and use_right:
        return 0.5*(results['left']['G'] + results['right']['G']), 0.5*(results['left']['S'] + results['right']['S'])
    elif use_left:
        return results['left']['G'], results['left']['S']
    elif use_right:
        return results['right']['G'], results['right']['S']
    else:
        raise ValueError("use_left 和 use_right 至少有一个必须为 True")

if __name__ == '__main__':
    # # 默认只计算左PSM
    # G, S = cal_GS()
    
    # #如果需要同时计算左右PSM，可以使用：
    # results = cal_GS(use_left=True, use_right=True)
    # print(f"\n左PSM - G值: {results['left']['G']:.6f}, S值: {results['left']['S']:.6f}")
    # print(f"右PSM - G值: {results['right']['G']:.6f}, S值: {results['right']['S']:.6f}")

    visualize_distribution(data_dir='13_data_12-01', use_left=True, use_right=True, save_fig=True)     