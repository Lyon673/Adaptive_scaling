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
    
    # 计算积分（使用梯形法则）
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
    # 默认只计算左PSM
    G, S = cal_GS()
    
    # 如果需要同时计算左右PSM，可以使用：
    # results = cal_GS(use_left=True, use_right=True)
    # print(f"\n左PSM - G值: {results['left']['G']:.6f}, S值: {results['left']['S']:.6f}")
    # print(f"右PSM - G值: {results['right']['G']:.6f}, S值: {results['right']['S']:.6f}")