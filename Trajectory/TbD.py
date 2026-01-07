import numpy as np
import matplotlib.pyplot as plt
from numpy.ma.core import true_divide
from scipy.interpolate import interp1d
from fastdtw import fastdtw
from scipy.spatial.distance import euclidean
from scipy.spatial.transform import Slerp
from sklearn.mixture import GaussianMixture
from load_data import R, load_demonstrations_state, load_demonstrations_label




class TbD_Framework:
    """
    Teaching by Demonstration (TbD) Framework for Surgical Robot Learning
    
    基于论文: Su et al. (2021) "Toward Teaching by Demonstration for 
    Robot-Assisted Minimally Invasive Surgery"
    
    主要步骤:
    1. DTW (Dynamic Time Warping) 对齐 - 论文 Section III.D.1
    2. GMM (Gaussian Mixture Model) 建模 - 论文 Section III.E, Eq.(18)
    3. GMR (Gaussian Mixture Regression) 生成 - 论文 Section III.E, Eq.(19)
    """
    def __init__(self, target_length=300, n_components=6):
        """
        初始化 TbD 框架
        
        Args:
            target_length: DTW对齐后的目标长度 (标准化时间步数)
            n_components: GMM的高斯分量个数 K (论文中建议5-10)
        """
        self.target_length = target_length
        self.n_components = n_components
        self.gmm = None
        self.time_steps = np.linspace(0, 1, target_length)  # 标准化时间变量 ξ_t ∈ [0,1]
        self.aligned_demos = None  # 保存对齐后的数据用于可视化

    # =========================================================
    # 步骤 1: DTW 对齐与预处理 (Preprocessing using DTW)
    # 论文 Section III.D.1: "Multiple demonstrations preprocessing using DTW"
    # =========================================================
    def _resample(self, data, target_length, use_position_only=False):
        """
        辅助函数：使用三次样条插值进行重采样
        
        Args:
            traj: 原始轨迹 (T, D)
            length: 目标长度
        Returns:
            重采样后的轨迹 (length, D)
        """
        if not use_position_only:
            l_pos = data[:, :3]
            l_quat = data[:, 3:7] 
            #l_grip = data[:, 7]
            
            # 右手
            r_pos = data[:, 7:10]
            r_quat = data[:, 10:14]
        #r_grip = data[:, 15]
        else:
            l_pos = data[:, 0:3]
            r_pos = data[:, 3:6]

        combined_traj = np.hstack((l_pos, r_pos))
        
        # 2. 计算在 6D 空间中的每一步位移
        deltas = np.diff(combined_traj, axis=0)
        
        # axis=1 求范数，相当于 sqrt(dx_L^2 + ... + dz_R^2)
        # 这就是比简单相加更科学的"联合距离"
        dists = np.linalg.norm(deltas, axis=1)
        
        # 3. 计算累计进度 (Progress Variable)
        s_cumulative = np.concatenate(([0], np.cumsum(dists)))
        total_dist = s_cumulative[-1]
        
        # 3.5 处理重复值问题：Slerp要求严格递增
        # 找出唯一值的索引（保持顺序）
        _, unique_indices = np.unique(s_cumulative, return_index=True)
        unique_indices = np.sort(unique_indices)  # 确保顺序
        
        # 如果有重复值，使用唯一值创建插值
        s_unique = s_cumulative[unique_indices]
        l_pos_unique = l_pos[unique_indices]
        r_pos_unique = r_pos[unique_indices]
        if not use_position_only:
            l_quat_unique = l_quat[unique_indices]
            r_quat_unique = r_quat[unique_indices]
        # l_grip_unique = l_grip[unique_indices]
        # r_grip_unique = r_grip[unique_indices]
        
        # 4. 生成新的均匀进度网格
        # 使用 linspace 确保精确生成 target_length 个点
        if target_length is not None:
            s_new = np.linspace(0, total_dist, target_length)
        else:
            # 如果没有指定目标长度，使用原始长度
            s_new = s_cumulative
        
        # 如果数据点太少，直接返回原始数据
        if len(s_unique) < 2:
            print("Warning: Not enough unique points for interpolation")
            return data
        
        # 左手插值函数
        f_left = interp1d(s_unique, l_pos_unique, axis=0, bounds_error=False, fill_value='extrapolate')
        new_l_pos = f_left(s_new)
        
        # 右手插值函数
        f_right = interp1d(s_unique, r_pos_unique, axis=0, bounds_error=False, fill_value='extrapolate')
        new_r_pos = f_right(s_new)

        if not use_position_only:
            # 左手旋转插值
            l_rot_obj = R.from_quat(l_quat_unique)
            l_slerp = Slerp(s_unique, l_rot_obj)
            new_l_quat = l_slerp(s_new).as_quat() # 返回插值后的四元数
            
            # 右手旋转插值
            r_rot_obj = R.from_quat(r_quat_unique)
            r_slerp = Slerp(s_unique, r_rot_obj)
            new_r_quat = r_slerp(s_new).as_quat()

        # Gripper插值（最近邻，适合离散的0/1值）
        # f_l_grip = interp1d(s_unique, l_grip_unique, kind='nearest', fill_value='extrapolate')
        # new_l_grip = f_l_grip(s_new)
        
        # f_r_grip = interp1d(s_unique, r_grip_unique, kind='nearest', fill_value='extrapolate')
        # new_r_grip = f_r_grip(s_new)



        # --- 5. 拼接结果 ---
        # 结果顺序: [Time, L_Pos(3), L_Quat(4), L_Grip(1), R_Pos(3), R_Quat(4), R_Grip(1), Delta_T(1)]
        if not use_position_only:
            resampled = np.column_stack((
                new_l_pos, new_l_quat, 
                new_r_pos, new_r_quat, 
            ))
        else:
            resampled = np.column_stack((
                new_l_pos, 
                new_r_pos, 
            ))
        
        return resampled

    def dtw_alignment(self, demos, use_position_only=False):
        """
        使用 DTW 将所有示教轨迹对齐到统一时间基准
        
        论文方法: 
        - 选择一个参考轨迹 (长度中位数)
        - 使用 DTW 计算其他轨迹与参考轨迹的最优匹配路径
        - 根据匹配路径对轨迹进行时间扭曲 (warping)
        
        Args:
            demos: 原始示教轨迹列表 [(T1,D), (T2,D), ..., (TN,D)]
        Returns:
            aligned_data: 对齐后的轨迹 (N_demos, target_length, D)
        """
        n_demos = len(demos)
        n_dims = demos[0].shape[1]
        print(f"\n{'='*60}")
        print(f"[DTW Alignment] Processing {n_demos} demonstrations")
        print(f"[DTW Alignment] Target length: {self.target_length}")
        print(f"[DTW Alignment] Feature dimensions: {n_dims}")
        print(f"{'='*60}")

        # 1. 选取参考轨迹: 选择长度为中位数的轨迹作为基准
        # 论文中提到这种选择可以减少对齐偏差
        lens = [len(d) for d in demos]
        ref_idx = np.argsort(lens)[n_demos // 2]
        print(f"[DTW] Selected reference demo: #{ref_idx} (length: {lens[ref_idx]})")
        
        # 将参考轨迹重采样到目标长度
        reference = self._resample(demos[ref_idx], self.target_length, use_position_only)
        #reference = demos[ref_idx]
        
        aligned_data = []
        dtw_distances = []

        for i, traj in enumerate(demos):
            if i == ref_idx:
                aligned_data.append(reference)
                dtw_distances.append(0.0)
                continue
            
            # 计算 DTW 路径
            # 论文中使用 DTW 来处理不同速度的示教
            distance, path = fastdtw(reference, traj, dist=euclidean)
            dtw_distances.append(distance)
            path = np.array(path)  # path: [(ref_idx, traj_idx), ...]
            
            # 根据 DTW 路径进行时间扭曲
            # 将当前轨迹扭曲以匹配参考轨迹的时间基准
            warped_traj = np.zeros((self.target_length, n_dims))
            
            for t_ref in range(self.target_length):
                # 找到参考轨迹时间步 t_ref 对应的所有当前轨迹索引
                matched_indices = path[path[:, 0] == t_ref, 1]
                
                if len(matched_indices) > 0:
                    # 如果有多个匹配点，取平均值
                    warped_traj[t_ref] = np.mean(traj[matched_indices], axis=0)
                else:
                    # 如果没有匹配点（罕见情况），使用前一个点或起点
                    warped_traj[t_ref] = warped_traj[t_ref-1] if t_ref > 0 else traj[0]
            
            aligned_data.append(warped_traj)
            
            if (i+1) % 10 == 0:
                print(f"[DTW] Aligned {i+1}/{n_demos} demonstrations...")
            
        print(f"[DTW] Alignment completed!")
        print(f"[DTW] Mean DTW distance: {np.mean(dtw_distances):.4f}")
        print(f"[DTW] Std DTW distance: {np.std(dtw_distances):.4f}\n")
        
        self.aligned_demos = np.array(aligned_data)  # Shape: (N_demos, target_length, D)
        return self.aligned_demos

    # =========================================================
    # 步骤 2: GMM 建模 (Gaussian Mixture Model)
    # 论文 Section III.E, Eq.(18): P(ξ) = Σ π_k N(ξ | μ_k, Σ_k)
    # =========================================================
    def train_gmm(self, aligned_demos):
        """
        使用 GMM 对时间-空间联合分布进行建模
        
        论文方法:
        - 将时间 t 和空间坐标 s 作为联合变量 ξ = [t, s]
        - 使用 GMM 建模联合概率分布 P(ξ) = Σ π_k N(ξ | μ_k, Σ_k)
        - 其中 K 个高斯分量可以看作是动作基元 (motion primitives)
        
        Args:
            aligned_demos: 对齐后的轨迹 (N_demos, target_length, D)
        """
        print(f"\n{'='*60}")
        print(f"[GMM Training] Building Gaussian Mixture Model")
        print(f"{'='*60}")
        
        n_demos, length, dim = aligned_demos.shape
        print(f"[GMM] Input: {n_demos} demos × {length} timesteps × {dim} dimensions")
        print(f"[GMM] Number of Gaussian components (K): {self.n_components}")
        
        # 构建训练数据 ξ = [t, s]
        # t: 归一化时间 ∈ [0,1]
        # s: 空间坐标 (位置 + 姿态)
        data_points = []
        for traj in aligned_demos:
            # 时间维度: 归一化到 [0, 1]
            t = self.time_steps.reshape(-1, 1)
            # 拼接时间和空间: ξ = [t, x, y, z, qx, qy, qz, qw, ...]
            augmented_data = np.hstack([t, traj])
            data_points.append(augmented_data)
            
        # 堆叠所有数据点: (N_demos × Length, D+1)
        train_data = np.vstack(data_points)
        print(f"[GMM] Training data shape: {train_data.shape}")
        print(f"[GMM] Data range: t ∈ [{train_data[:,0].min():.3f}, {train_data[:,0].max():.3f}]")
        
        # 训练 GMM
        # 论文中使用 full covariance 以捕捉时间和空间的协方差
        self.gmm = GaussianMixture(
            n_components=self.n_components, 
            covariance_type='full',  # 完整协方差矩阵
            random_state=42,
            max_iter=200,
            n_init=10,  # 多次初始化以获得更好的结果
            verbose=0
        )
        
        self.gmm.fit(train_data)
        
        print(f"[GMM] Training completed!")
        print(f"[GMM] Converged: {self.gmm.converged_}")
        print(f"[GMM] Iterations: {self.gmm.n_iter_}")
        print(f"[GMM] Log-likelihood: {self.gmm.score(train_data):.4f}")
        
        # 显示每个高斯分量的权重
        print(f"[GMM] Component weights (π_k):")
        for k, weight in enumerate(self.gmm.weights_):
            print(f"      Component {k}: {weight:.4f}")
        print()

    # =========================================================
    # 步骤 3: GMR 生成 (Gaussian Mixture Regression)
    # 论文 Section III.E, Eq.(19): E[ξ_s | ξ_t] = Σ β_k(ξ_t) · μ̂_s,k
    # =========================================================
    def generate_trajectory(self):
        """
        使用 GMR 通过条件概率生成标准示教轨迹
        
        论文方法:
        - 对每个时间步 t，计算条件期望 E[s | t]
        - 使用贝叶斯规则计算激活权重 β_k(t) = P(k | t)
        - 条件期望: μ̂_s,k = μ_s,k + Σ_st,k · Σ_tt,k^(-1) · (t - μ_t,k)
        - 最终轨迹: s(t) = Σ β_k(t) · μ̂_s,k
        
        Returns:
            generated_traj: 生成的标准轨迹 (target_length, D)
        """
        print(f"\n{'='*60}")
        print(f"[GMR Generation] Generating standard demonstration")
        print(f"{'='*60}")
        
        # 获取 GMM 参数
        means = self.gmm.means_          # μ_k: (K, D+1)
        covariances = self.gmm.covariances_  # Σ_k: (K, D+1, D+1)
        priors = self.gmm.weights_        # π_k: (K,)
        
        n_dims = means.shape[1] - 1  # 空间维度数量 (去除时间维度)
        generated_traj = np.zeros((self.target_length, n_dims))
        
        print(f"[GMR] Generating {self.target_length} timesteps...")
        print(f"[GMR] Output dimension: {n_dims}")
        
        # 对每个时间步进行 GMR
        for i, t_curr in enumerate(self.time_steps):
            
            # 存储每个高斯分量的贡献
            betas = []  # 激活权重 β_k
            conditional_means = []  # 条件期望 μ̂_s,k
            
            # 遍历所有高斯分量 K
            for k in range(self.n_components):
                # 分离时间和空间的均值
                mu_t = means[k, 0]       # μ_t,k: 时间均值
                mu_s = means[k, 1:]      # μ_s,k: 空间均值
                
                # 协方差矩阵分块 (Block partition):
                #     ┌          ┐
                # Σ = │ Σ_tt  Σ_ts │
                #     │ Σ_st  Σ_ss │
                #     └          ┘
                sigma_tt = covariances[k, 0, 0]      # 时间-时间协方差 (标量)
                sigma_ts = covariances[k, 0, 1:]     # 时间-空间协方差 (1×D)
                sigma_st = covariances[k, 1:, 0]     # 空间-时间协方差 (D×1)
                # sigma_ss = covariances[k, 1:, 1:]  # 空间-空间协方差 (D×D) [GMR不需要]
                
                # 步骤 1: 计算激活权重 β_k(t) = P(k | t)
                # 使用贝叶斯规则: P(k | t) ∝ P(t | k) · P(k)
                # P(t | k) = N(t | μ_t,k, σ_tt,k)
                eps = 1e-8  # 数值稳定性
                prob_t = (1.0 / np.sqrt(2 * np.pi * (sigma_tt + eps))) * \
                         np.exp(-0.5 * ((t_curr - mu_t)**2) / (sigma_tt + eps))
                
                # β_k ∝ π_k · P(t | k)
                beta = priors[k] * prob_t
                betas.append(beta)
                
                # 步骤 2: 计算条件期望 μ̂_s,k (Eq. 19)
                # E[s | t, k] = μ_s,k + Σ_st,k · (Σ_tt,k)^(-1) · (t - μ_t,k)
                conditional_mean = mu_s + sigma_st * (1.0 / (sigma_tt + eps)) * (t_curr - mu_t)
                conditional_means.append(conditional_mean)
            
            # 步骤 3: 归一化激活权重
            betas = np.array(betas)
            sum_beta = np.sum(betas)
            
            if sum_beta > 1e-10:
                # 归一化: β_k = β_k / Σ β_k
                betas = betas / sum_beta
                
                # 步骤 4: 加权组合所有分量的条件期望
                # s(t) = Σ β_k(t) · μ̂_s,k
                generated_traj[i] = np.sum([b * m for b, m in zip(betas, conditional_means)], axis=0)
            else:
                # 极端情况: 所有高斯分量对当前时间的贡献都接近0
                # 使用前一个时间步的结果
                if i > 0:
                    generated_traj[i] = generated_traj[i-1]
                else:
                    # 第一个时间步，使用所有分量的空间均值的平均
                    generated_traj[i] = np.mean([means[k, 1:] for k in range(self.n_components)], axis=0)
            
            if (i+1) % 50 == 0:
                print(f"[GMR] Progress: {i+1}/{self.target_length}")
        
        print(f"[GMR] Generation completed!")
        print(f"[GMR] Output trajectory shape: {generated_traj.shape}\n")
        
        return generated_traj


def load_trajectories(use_position_only=False, demo_group='all'):
    """
    加载并预处理轨迹数据
    
    Args:
        use_position_only: 是否只使用位置信息 (否则使用位置+姿态)
        demo_group: 'all', 'group1' (0-29), 'group2' (28-44,47-50,55-59)
        
    Returns:
        trajectories: 预处理后的轨迹列表
    """
    print(f"\n{'='*60}")
    print(f"[Data Loading] Loading demonstration trajectories")
    print(f"{'='*60}")
    
    # 加载原始数据
    all_demos = load_demonstrations_state(shuffle=False)
    all_labels = load_demonstrations_label(shuffle=False)
    
    # 提取位置和姿态特征
    # 列索引: 0-6 (左手: x,y,z,qx,qy,qz,qw), 7 (左手夹爪), 
    #        8-14 (右手: x,y,z,qx,qy,qz,qw), 15 (右手夹爪)
    # 排除夹爪状态 (7, 15)
    if use_position_only:
        # 只使用位置: 左手xyz + 右手xyz
        all_demos = [demo[:, [0,1,2,8,9,10]] for demo in all_demos]
        print(f"[Data] Using position only (6D)")
    else:
        # 使用位置+姿态: 左手xyz+quat + 右手xyz+quat
        all_demos = [demo[:, [0,1,2,3,4,5,6,8,9,10,11,12,13,14]] for demo in all_demos]
        print(f"[Data] Using position + orientation (14D)")
    
    print(f"[Data] Total demonstrations loaded: {len(all_demos)}")
    
    # 根据标签截取有效片段 (从第一个类别0到最后一个类别5)
    trajectories = []
    incomplete_demos = []
    
    for idx, (demo, labels) in enumerate(zip(all_demos, all_labels)):
        labels_arr = np.array(labels)
        
        # 找到类别0的起始和类别5的结束
        start_candidates = np.where(labels_arr == 0)[0]
        end_candidates = np.where(labels_arr == 5)[0]
        
        # 只处理包含完整 0-5 序列的 demo
        if len(start_candidates) > 0 and len(end_candidates) > 0:
            start_idx = start_candidates[0]  # 第一个类别0
            end_idx = end_candidates[-1]      # 最后一个类别5
            
            if start_idx <= end_idx:
                trajectories.append(demo[start_idx:end_idx + 1])
            else:
                print(f"Warning: demo {idx} has invalid label order, keeping full demo.")
                trajectories.append(demo)
                incomplete_demos.append(idx)
        else:
            # 缺少类别0或5
            missing = []
            if len(start_candidates) == 0:
                missing.append("class 0")
            if len(end_candidates) == 0:
                missing.append("class 5")
            print(f"Warning: demo {idx} is missing {' and '.join(missing)}, keeping full demo.")
            trajectories.append(demo)
            incomplete_demos.append(idx)
    
    if incomplete_demos:
        print(f"[Data] Incomplete demos (kept full): {len(incomplete_demos)} - {incomplete_demos}")
    
    print(f"[Data] Valid demos (class 0-5): {len(trajectories) - len(incomplete_demos)}")
    
    # 根据 demo_group 参数筛选数据
    if demo_group == 'group1':
        group_indices = list(range(0, 30))
        trajectories = [trajectories[i] for i in group_indices if i < len(trajectories)]
        print(f"[Data] Selected Group 1: demos 0-29 ({len(trajectories)} demos)")
    elif demo_group == 'group2':
        group_indices = list(range(28, 45)) + list(range(47, 51)) + list(range(55, 60))
        trajectories = [trajectories[i] for i in group_indices if i < len(trajectories)]
        print(f"[Data] Selected Group 2: demos 28-44,47-50,55-59 ({len(trajectories)} demos)")
    else:
        print(f"[Data] Using all demos ({len(trajectories)} demos)")
    
    # 显示统计信息
    lengths = [len(traj) for traj in trajectories]
    print(f"[Data] Trajectory lengths: min={min(lengths)}, max={max(lengths)}, "
          f"mean={np.mean(lengths):.1f}, median={np.median(lengths):.1f}")
    
    return trajectories

def visualize_results(aligned_demos, standard_traj, save_fig=False):
    """
    可视化 TbD 结果
    
    Args:
        aligned_demos: DTW对齐后的轨迹 (N, T, D)
        standard_traj: GMR生成的标准轨迹 (T, D)
        save_fig: 是否保存图像
    """
    import os
    
    n_dims = standard_traj.shape[1]
    
    # 创建多子图布局
    if n_dims >= 6:  # 双手位置
        fig = plt.figure(figsize=(16, 12))
        
        # 左手 XYZ (列 0,1,2)
        for i, axis in enumerate(['X', 'Y', 'Z']):
            ax = plt.subplot(3, 3, i+1)
            # 绘制所有对齐后的demo
            for demo in aligned_demos:
                ax.plot(demo[:, i], 'gray', alpha=0.2, linewidth=0.5)
            # 绘制标准轨迹
            ax.plot(standard_traj[:, i], 'b-', linewidth=2.5, label='Standard Trajectory')
            ax.set_title(f'Left Hand - {axis} Position')
            ax.set_xlabel('Timestep')
            ax.set_ylabel(f'{axis} (m)')
            ax.grid(True, alpha=0.3)
            if i == 0:
                ax.legend()
        
        # 右手 XYZ (列 3,4,5 或更多)
        offset = 3 if n_dims == 6 else 7  # 如果有姿态，右手位置从索引7开始
        for i, axis in enumerate(['X', 'Y', 'Z']):
            ax = plt.subplot(3, 3, i+4)
            for demo in aligned_demos:
                ax.plot(demo[:, offset+i], 'gray', alpha=0.2, linewidth=0.5)
            ax.plot(standard_traj[:, offset+i], 'r-', linewidth=2.5, label='Standard Trajectory')
            ax.set_title(f'Right Hand - {axis} Position')
            ax.set_xlabel('Timestep')
            ax.set_ylabel(f'{axis} (m)')
            ax.grid(True, alpha=0.3)
            if i == 0:
                ax.legend()
        
        # 3D 轨迹可视化 - 左手
        ax = fig.add_subplot(3, 3, 7, projection='3d')
        for demo in aligned_demos:
            ax.plot(demo[:, 0], demo[:, 1], demo[:, 2], 'gray', alpha=0.2, linewidth=0.5)
        ax.plot(standard_traj[:, 0], standard_traj[:, 1], standard_traj[:, 2], 
                'b-', linewidth=2.5, label='Standard')
        ax.set_title('Left Hand - 3D Trajectory')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.legend()
        
        # 3D 轨迹可视化 - 右手
        ax = fig.add_subplot(3, 3, 8, projection='3d')
        for demo in aligned_demos:
            ax.plot(demo[:, offset], demo[:, offset+1], demo[:, offset+2], 
                    'gray', alpha=0.2, linewidth=0.5)
        ax.plot(standard_traj[:, offset], standard_traj[:, offset+1], standard_traj[:, offset+2], 
                'r-', linewidth=2.5, label='Standard')
        ax.set_title('Right Hand - 3D Trajectory')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.legend()
        
        # 统计信息
        ax = plt.subplot(3, 3, 9)
        ax.axis('off')
        stats_text = f"""
        TbD Statistics:
        
        Demonstrations: {len(aligned_demos)}
        Trajectory Length: {len(standard_traj)}
        Feature Dimensions: {n_dims}
        
        Standard Deviation from Mean:
        Left X:  {np.std(aligned_demos[:,:,0]):.4f} m
        Left Y:  {np.std(aligned_demos[:,:,1]):.4f} m
        Left Z:  {np.std(aligned_demos[:,:,2]):.4f} m
        Right X: {np.std(aligned_demos[:,:,offset]):.4f} m
        Right Y: {np.std(aligned_demos[:,:,offset+1]):.4f} m
        Right Z: {np.std(aligned_demos[:,:,offset+2]):.4f} m
        """
        ax.text(0.1, 0.5, stats_text, fontsize=10, family='monospace',
                verticalalignment='center')
        
    plt.tight_layout()
    
    if save_fig:
        save_dir = os.path.join(os.path.dirname(__file__), 'LSTM_visualization_results')
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, 'tbd_standard_trajectory.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"\n[Visualization] Saved to: {save_path}")
    
    plt.show()

def get_standard_trajectory(demo_group, use_position_only=True):
    demos = load_trajectories(use_position_only=use_position_only, demo_group=demo_group)
    
    # 2. 实例化 TbD 框架
    # target_length: DTW对齐后的统一长度 (论文建议200-300)
    # n_components: GMM高斯分量数 (论文建议5-10，表示动作基元数量)
    tbd = TbD_Framework(target_length=300, n_components=10)
    
    print("\n" + "="*60)
    print(" Starting TbD Pipeline")
    print("="*60)
    
    # 3. 执行 TbD 流程
    # Step 1: DTW 对齐
    aligned_demos = tbd.dtw_alignment(demos,use_position_only=use_position_only)
    
    # Step 2: GMM 建模
    tbd.train_gmm(aligned_demos)
    
    # Step 3: GMR 生成标准轨迹
    standard_traj = tbd.generate_trajectory()
    return standard_traj


def analyze_variance(aligned_demos, standard_traj, save_fig=False):
    """
    分析输入样本与标准示教之间的方差并可视化
    
    Args:
        aligned_demos: DTW对齐后的轨迹 (N, T, D)
        standard_traj: GMR生成的标准轨迹 (T, D)
        save_fig: 是否保存图像
        
    Returns:
        results: 包含各种统计指标的字典
    """
    import os
    import seaborn as sns
    from scipy import stats
    
    n_demos, n_timesteps, n_dims = aligned_demos.shape
    
    print(f"\n{'='*60}")
    print(f"[Variance Analysis] Analyzing variability")
    print(f"{'='*60}")
    
    # =========================================================
    # 1. 计算每个demo与标准轨迹的差异
    # =========================================================
    # deviations: (N, T, D) - 每个demo在每个时间步、每个维度上与标准轨迹的偏差
    deviations = aligned_demos - standard_traj[np.newaxis, :, :]
    
    # 计算欧氏距离 (每个时间步的总偏差)
    euclidean_distances = np.linalg.norm(deviations, axis=2)  # (N, T)
    
    # =========================================================
    # 2. 统计分析
    # =========================================================
    # 2.1 按时间步统计
    mean_deviation_per_timestep = np.mean(euclidean_distances, axis=0)  # (T,)
    std_deviation_per_timestep = np.std(euclidean_distances, axis=0)    # (T,)
    
    # 2.2 按维度统计
    variance_per_dim = np.var(aligned_demos, axis=(0, 1))  # (D,)
    std_per_dim = np.std(aligned_demos, axis=(0, 1))       # (D,)
    
    # 2.3 按demo统计
    mean_distance_per_demo = np.mean(euclidean_distances, axis=1)  # (N,)
    
    # 2.4 全局统计
    global_mean_distance = np.mean(euclidean_distances)
    global_std_distance = np.std(euclidean_distances)
    
    print(f"[Variance] Global mean distance: {global_mean_distance:.6f} m")
    print(f"[Variance] Global std distance: {global_std_distance:.6f} m")
    print(f"[Variance] Max distance: {np.max(euclidean_distances):.6f} m")
    print(f"[Variance] Min distance: {np.min(euclidean_distances):.6f} m")
    
    # 维度标签
    if n_dims == 6:
        dim_labels = ['L_X', 'L_Y', 'L_Z', 'R_X', 'R_Y', 'R_Z']
    elif n_dims == 14:
        dim_labels = ['L_X', 'L_Y', 'L_Z', 'L_qx', 'L_qy', 'L_qz', 'L_qw',
                     'R_X', 'R_Y', 'R_Z', 'R_qx', 'R_qy', 'R_qz', 'R_qw']
    else:
        dim_labels = [f'Dim_{i}' for i in range(n_dims)]
    
    # =========================================================
    # 3. 可视化
    # =========================================================
    fig = plt.figure(figsize=(20, 14))
    
    # -------------------------
    # 3.1 总体偏差随时间变化 (带置信区间)
    # -------------------------
    ax1 = plt.subplot(3, 3, 1)
    ax1.plot(mean_deviation_per_timestep, 'b-', linewidth=2, label='Mean Distance')
    ax1.fill_between(range(n_timesteps),
                     mean_deviation_per_timestep - std_deviation_per_timestep,
                     mean_deviation_per_timestep + std_deviation_per_timestep,
                     alpha=0.3, label='±1 Std Dev')
    ax1.set_xlabel('Timestep')
    ax1.set_ylabel('Distance (m)')
    ax1.set_title('Mean Deviation from Standard Trajectory Over Time')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # -------------------------
    # 3.2 所有demo的距离热力图
    # -------------------------
    ax2 = plt.subplot(3, 3, 2)
    im = ax2.imshow(euclidean_distances, aspect='auto', cmap='hot', interpolation='nearest')
    ax2.set_xlabel('Timestep')
    ax2.set_ylabel('Demonstration ID')
    ax2.set_title('Distance Heatmap (All Demos)')
    plt.colorbar(im, ax=ax2, label='Distance (m)')
    
    # -------------------------
    # 3.3 每个demo的平均距离分布
    # -------------------------
    ax3 = plt.subplot(3, 3, 3)
    ax3.bar(range(n_demos), mean_distance_per_demo, color='steelblue', alpha=0.7)
    ax3.axhline(y=global_mean_distance, color='r', linestyle='--', 
                label=f'Global Mean: {global_mean_distance:.4f}m')
    ax3.set_xlabel('Demonstration ID')
    ax3.set_ylabel('Mean Distance (m)')
    ax3.set_title('Mean Distance per Demonstration')
    ax3.legend()
    ax3.grid(True, alpha=0.3, axis='y')
    
    # -------------------------
    # 3.4 每个维度的方差
    # -------------------------
    ax4 = plt.subplot(3, 3, 4)
    colors = ['blue']*3 + ['orange']*4 + ['green']*3 + ['red']*4 if n_dims == 14 else \
             ['blue']*3 + ['green']*3 if n_dims == 6 else ['gray']*n_dims
    ax4.bar(range(n_dims), variance_per_dim, color=colors[:n_dims], alpha=0.7)
    ax4.set_xlabel('Dimension')
    ax4.set_ylabel('Variance')
    ax4.set_title('Variance per Dimension')
    ax4.set_xticks(range(n_dims))
    ax4.set_xticklabels(dim_labels, rotation=45, ha='right')
    ax4.grid(True, alpha=0.3, axis='y')
    
    # -------------------------
    # 3.5 距离分布直方图
    # -------------------------
    ax5 = plt.subplot(3, 3, 5)
    ax5.hist(euclidean_distances.flatten(), bins=50, color='steelblue', 
             alpha=0.7, edgecolor='black')
    ax5.axvline(x=global_mean_distance, color='r', linestyle='--', linewidth=2,
                label=f'Mean: {global_mean_distance:.4f}m')
    ax5.axvline(x=np.median(euclidean_distances), color='g', linestyle='--', linewidth=2,
                label=f'Median: {np.median(euclidean_distances):.4f}m')
    ax5.set_xlabel('Distance (m)')
    ax5.set_ylabel('Frequency')
    ax5.set_title('Distance Distribution')
    ax5.legend()
    ax5.grid(True, alpha=0.3, axis='y')
    
    # -------------------------
    # 3.6 按维度的标准差随时间变化
    # -------------------------
    ax6 = plt.subplot(3, 3, 6)
    std_per_dim_per_time = np.std(aligned_demos, axis=0)  # (T, D)
    
    # 只显示位置维度
    if n_dims >= 6:
        for i in range(3):
            ax6.plot(std_per_dim_per_time[:, i], label=f'Left {dim_labels[i]}', alpha=0.7)
        offset = 7 if n_dims == 14 else 3
        for i in range(3):
            ax6.plot(std_per_dim_per_time[:, offset+i], label=f'Right {dim_labels[offset+i]}', 
                    alpha=0.7, linestyle='--')
    else:
        for i in range(min(6, n_dims)):
            ax6.plot(std_per_dim_per_time[:, i], label=dim_labels[i], alpha=0.7)
    
    ax6.set_xlabel('Timestep')
    ax6.set_ylabel('Standard Deviation (m)')
    ax6.set_title('Std Dev per Dimension Over Time')
    ax6.legend(fontsize=8, ncol=2)
    ax6.grid(True, alpha=0.3)
    
    # -------------------------
    # 3.7 Box plot - 按时间段分组
    # -------------------------
    ax7 = plt.subplot(3, 3, 7)
    n_segments = 5
    segment_size = n_timesteps // n_segments
    segment_data = []
    segment_labels = []
    
    for i in range(n_segments):
        start_idx = i * segment_size
        end_idx = (i + 1) * segment_size if i < n_segments - 1 else n_timesteps
        segment_data.append(euclidean_distances[:, start_idx:end_idx].flatten())
        segment_labels.append(f'T{start_idx}-{end_idx}')
    
    bp = ax7.boxplot(segment_data, labels=segment_labels, patch_artist=True)
    for patch in bp['boxes']:
        patch.set_facecolor('lightblue')
    ax7.set_xlabel('Time Segment')
    ax7.set_ylabel('Distance (m)')
    ax7.set_title('Distance Distribution Across Time Segments')
    ax7.grid(True, alpha=0.3, axis='y')
    
    # -------------------------
    # 3.8 累积分布函数 (CDF)
    # -------------------------
    ax8 = plt.subplot(3, 3, 8)
    sorted_distances = np.sort(euclidean_distances.flatten())
    cdf = np.arange(1, len(sorted_distances) + 1) / len(sorted_distances)
    ax8.plot(sorted_distances, cdf, linewidth=2, color='darkblue')
    ax8.axhline(y=0.5, color='r', linestyle='--', alpha=0.5, label='50th percentile')
    ax8.axhline(y=0.95, color='orange', linestyle='--', alpha=0.5, label='95th percentile')
    ax8.set_xlabel('Distance (m)')
    ax8.set_ylabel('Cumulative Probability')
    ax8.set_title('Cumulative Distribution Function')
    ax8.legend()
    ax8.grid(True, alpha=0.3)
    
    # -------------------------
    # 3.9 统计信息表格
    # -------------------------
    ax9 = plt.subplot(3, 3, 9)
    ax9.axis('off')
    
    # 计算额外统计量
    percentile_25 = np.percentile(euclidean_distances, 25)
    percentile_50 = np.percentile(euclidean_distances, 50)
    percentile_75 = np.percentile(euclidean_distances, 75)
    percentile_95 = np.percentile(euclidean_distances, 95)
    
    # Shapiro-Wilk 正态性检验 (采样以避免数据过多)
    sample_size = min(5000, euclidean_distances.size)
    sample_indices = np.random.choice(euclidean_distances.size, sample_size, replace=False)
    sample_data = euclidean_distances.flatten()[sample_indices]
    shapiro_stat, shapiro_pval = stats.shapiro(sample_data)
    
    stats_text = f"""
    Variance Analysis Statistics:
    
    Global Metrics:
      Mean Distance:      {global_mean_distance:.6f} m
      Std Dev:            {global_std_distance:.6f} m
      Coefficient of Var: {(global_std_distance/global_mean_distance)*100:.2f}%
      Min Distance:       {np.min(euclidean_distances):.6f} m
      Max Distance:       {np.max(euclidean_distances):.6f} m
      
    Percentiles:
      25th:               {percentile_25:.6f} m
      50th (Median):      {percentile_50:.6f} m
      75th:               {percentile_75:.6f} m
      95th:               {percentile_95:.6f} m
      
    Normality Test (Shapiro-Wilk):
      Statistic:          {shapiro_stat:.4f}
      P-value:            {shapiro_pval:.4e}
      Normal?:            {'Yes' if shapiro_pval > 0.05 else 'No'}
      
    Demonstrations:     {n_demos}
    Timesteps:          {n_timesteps}
    Dimensions:         {n_dims}
    """
    
    ax9.text(0.05, 0.5, stats_text, fontsize=10, family='monospace',
             verticalalignment='center', transform=ax9.transAxes)
    
    plt.tight_layout()
    
    if save_fig:
        save_dir = os.path.join(os.path.dirname(__file__), 'LSTM_visualization_results')
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, 'tbd_variance_analysis.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"\n[Variance Analysis] Saved to: {save_path}")
    
    plt.show()
    
    # =========================================================
    # 4. 返回统计结果
    # =========================================================
    results = {
        'mean_distance': global_mean_distance,
        'std_distance': global_std_distance,
        'variance_per_dim': variance_per_dim,
        'mean_deviation_per_timestep': mean_deviation_per_timestep,
        'std_deviation_per_timestep': std_deviation_per_timestep,
        'mean_distance_per_demo': mean_distance_per_demo,
        'percentiles': {
            '25': percentile_25,
            '50': percentile_50,
            '75': percentile_75,
            '95': percentile_95
        }
    }
    
    return results


# =========================================================
# 主程序：运行 TbD 框架
# =========================================================
if __name__ == "__main__":
    print("\n" + "="*60)
    print(" Teaching by Demonstration (TbD) Framework")
    print(" Based on: Su et al. (2021) IEEE Trans. Automation Sci. Eng.")
    print("="*60)
    
    # 1. 加载数据
    # 参数选项:
    #   use_position_only=True: 只使用位置 (6D)
    #   use_position_only=False: 使用位置+姿态 (14D)
    #   demo_group='all'/'group1'/'group2'

    use_position_only = True

    demos = load_trajectories(use_position_only=use_position_only, demo_group='group2')
    
    # 2. 实例化 TbD 框架
    # target_length: DTW对齐后的统一长度 (论文建议200-300)
    # n_components: GMM高斯分量数 (论文建议5-10，表示动作基元数量)
    tbd = TbD_Framework(target_length=300, n_components=10)
    
    print("\n" + "="*60)
    print(" Starting TbD Pipeline")
    print("="*60)
    
    # 3. 执行 TbD 流程
    # Step 1: DTW 对齐
    aligned_demos = tbd.dtw_alignment(demos,use_position_only=use_position_only)
    
    # Step 2: GMM 建模
    tbd.train_gmm(aligned_demos)
    
    # Step 3: GMR 生成标准轨迹
    standard_traj = tbd.generate_trajectory()
    
    # 4. 结果分析
    print("="*60)
    print(" TbD Results")
    print("="*60)
    print(f"Standard trajectory shape: {standard_traj.shape}")
    print(f"  - Timesteps: {standard_traj.shape[0]}")
    print(f"  - Dimensions: {standard_traj.shape[1]}")
    
    # 计算变异性（所有demo与标准轨迹的平均距离）
    avg_distances = []
    for demo in aligned_demos:
        dist = np.mean(np.linalg.norm(demo - standard_traj, axis=1))
        avg_distances.append(dist)
    
    print(f"\nVariability Analysis:")
    print(f"  Mean distance from standard: {np.mean(avg_distances):.6f} m")
    print(f"  Std distance from standard:  {np.std(avg_distances):.6f} m")
    print(f"  Min distance: {np.min(avg_distances):.6f} m")
    print(f"  Max distance: {np.max(avg_distances):.6f} m")
    
    # 5. 可视化轨迹
    print("\n" + "="*60)
    print(" Generating Trajectory Visualization...")
    print("="*60)
    visualize_results(aligned_demos, standard_traj, save_fig=True)
    
    # 6. 方差分析
    print("\n" + "="*60)
    print(" Performing Variance Analysis...")
    print("="*60)
    variance_results = analyze_variance(aligned_demos, standard_traj, save_fig=True)
    
    print("\n" + "="*60)
    print(" TbD Pipeline Completed Successfully!")
    print("="*60)