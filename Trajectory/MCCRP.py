import numpy as np
from scipy.stats import multivariate_normal
from sklearn.decomposition import PCA
from sklearn.metrics import mutual_info_score
from dtw import accelerated_dtw
import similaritymeasures
import warnings
import matplotlib.pyplot as plt
from load_data import load_demonstrations_state
import pickle
import os
from TSC import TSC
from config import TSC_model_path
from main import load_model
from cluster_visualization import visualize_tsc_results
from tqdm import tqdm
import time

warnings.filterwarnings('ignore', category=FutureWarning)

# class DPCRP_Segmenter:
#     """
#     实现了论文中的DP-CRP（狄利克雷过程 - 中餐馆过程）聚类分割算法。
#     该实现基于吉布斯采样，并假设每个簇的数据服从多元高斯分布。
#     """

#     def __init__(self, alpha=1.0):
#         """
#         初始化
#         :param alpha: CRP的集中度参数。alpha越大，越倾向于生成新的簇。 
#         """
#         self.alpha = alpha
#         self.data = None
#         self.assignments = []
#         self.clusters = {} # key: cluster_id, value: {'mean': ..., 'cov': ..., 'points': ..., 'count': ...}
#         self.n_features = 0

#     def _add_to_cluster(self, point_idx, cluster_id):
#         """将一个点添加到指定的簇"""
#         point = self.data[point_idx]
#         self.assignments[point_idx] = cluster_id
        
#         if cluster_id not in self.clusters:
#             self.clusters[cluster_id] = {
#                 'points': [],
#                 'count': 0,
#                 'mean': np.zeros(self.n_features),
#                 'cov': np.eye(self.n_features)
#             }

#         c = self.clusters[cluster_id]
#         c['points'].append(point_idx)
#         c['count'] += 1
        
#         # 在线更新均值和协方差 (Welford's algorithm can be more stable, but this is simpler)
#         if c['count'] == 1:
#             c['mean'] = point
#         else:
#             old_mean = c['mean']
#             c['mean'] = old_mean + (point - old_mean) / c['count']
#             if c['count'] > 1:
#                  # A simple but potentially unstable way to update covariance
#                 c['cov'] = np.cov(self.data[c['points']].T)

#     def _remove_from_cluster(self, point_idx):
#         """从一个点所属的簇中移除该点"""
#         cluster_id = self.assignments[point_idx]
#         point = self.data[point_idx]
#         c = self.clusters[cluster_id]

#         c['points'].remove(point_idx)
#         c['count'] -= 1
        
#         if c['count'] == 0:
#             del self.clusters[cluster_id]
#         else:
#             # Re-calculate mean and covariance
#             c['mean'] = np.mean(self.data[c['points']], axis=0)
#             if c['count'] > 1:
#                 c['cov'] = np.cov(self.data[c['points']].T)


#     def _log_likelihood(self, point, cluster_info):
#         """计算一个点属于某个簇的对数似然概率"""
#         mean = cluster_info['mean']
#         # 加上一个小的对角矩阵防止协方差矩阵奇异
#         cov = cluster_info['cov'] + 1e-6 * np.eye(self.n_features)
#         return multivariate_normal.logpdf(point, mean=mean, cov=cov)

#     def fit(self, data, max_iter=20, verbose=True):
#         """
#         使用吉布斯采样来拟合数据，实现算法1和算法2的过程。
#         :param data: 运动学数据, shape (n_samples, n_features)
#         :param max_iter: 吉布斯采样的最大迭代次数
#         """
#         self.data = data
#         n_samples, self.n_features = data.shape
#         # 处理一维时间数据的情况
#         if len(data.shape) == 1:
#             self.data = data.reshape(-1, 1)
#             n_samples, self.n_features = self.data.shape

#         self.assignments = [-1] * n_samples
        
#         # --- 初始化 (类似 Algorithm 1) ---
#         for i in range(n_samples):
#             self._add_to_cluster(i, 0) # Start with all points in one cluster

#         # ======================================================================
#         # =================== 修改核心：优化新类别的先验 =======================
#         # ======================================================================
#         # 预先计算整个数据集的均值和协方差，作为新类别的先验
#         data_mean = np.mean(self.data, axis=0)
#         if self.n_features > 1 and n_samples > 1:
#             data_cov = np.cov(self.data.T)
#         else:
#             # 处理一维或样本不足的情况
#             data_cov = np.eye(self.n_features) * max(np.var(self.data), 1e-6)
#         # ======================================================================

#         # --- 吉布斯采样优化 (Algorithm 2) ---
#         for it in range(max_iter):
#             if verbose:
#                 print(f"Gibbs Sampling Iteration: {it + 1}/{max_iter}")
#             for i in range(n_samples):
#                 self._remove_from_cluster(i)
                
#                 # 计算分配到现有簇的概率
#                 log_probs = []
#                 cluster_ids = list(self.clusters.keys())
#                 for cid in cluster_ids:
#                     c = self.clusters[cid]
#                     log_prior = np.log(c['count'])
#                     log_lik = self._log_likelihood(self.data[i], c)
#                     log_probs.append(log_prior + log_lik)

#                 # 计算分配到新簇的概率
#                 log_prior_new = np.log(self.alpha)
#                 # ===================================================================
#                 # =========== 使用基于数据自身分布的先验，而不是固定的先验 ==========
#                 # ===================================================================
#                 log_lik_new = multivariate_normal.logpdf(self.data[i], 
#                                                         mean=data_mean, 
#                                                         cov=data_cov + 1e-6 * np.eye(self.n_features))
#                 # ===================================================================
#                 log_probs.append(log_prior_new + log_lik_new)
                
#                 # 归一化并选择新簇
#                 probs = np.exp(log_probs - np.max(log_probs))
#                 probs /= probs.sum()
                
#                 new_assignment_idx = np.random.choice(len(probs), p=probs)
                
#                 if new_assignment_idx == len(cluster_ids): # A new cluster was chosen
#                     new_cid = max(self.clusters.keys()) + 1 if self.clusters else 0
#                     self._add_to_cluster(i, new_cid)
#                 else: # An existing cluster was chosen
#                     self._add_to_cluster(i, cluster_ids[new_assignment_idx])
            
#             if verbose:
#                 print(f"  Number of clusters: {len(self.clusters)}")

#     def get_segmentation(self):
#         """
#         从聚类结果中提取分段信息。
#         返回一个列表，每个元素是 (start_frame, end_frame, cluster_id)
#         """
#         if not self.assignments:
#             return []
            
#         segments = []
#         current_cluster = self.assignments[0]
#         start_frame = 0
#         for i in range(1, len(self.assignments)):
#             if self.assignments[i] != current_cluster:
#                 segments.append((start_frame, i - 1, current_cluster))
#                 start_frame = i
#                 current_cluster = self.assignments[i]
        
#         segments.append((start_frame, len(self.assignments) - 1, current_cluster))
#         return segments

#     def get_final_clusters(self):
#         """
#         生成用于可视化的final_clusters结果，类似TSC.py中的格式。
#         返回一个字典，包含每个簇的详细信息。
#         """
#         if not self.assignments or not self.clusters:
#             return {}
            
#         final_clusters = {}
        
#         # 为每个簇生成详细信息
#         for cluster_id, cluster_info in self.clusters.items():
#             # 获取属于该簇的所有点
#             cluster_points = cluster_info['points']
#             if not cluster_points: continue
#             cluster_data = self.data[cluster_points]
            
#             # 计算簇中心（均值）
#             cluster_center = cluster_info['mean']
            
#             # 计算时间信息（假设数据是按时间顺序的）
#             time_points = np.array(cluster_points)
            
#             # 生成簇的键名
#             key = f"Cluster_{cluster_id}"
            
#             final_clusters[key] = {
#                 'center': cluster_center,
#                 'mean': cluster_center,  # 保持兼容性
#                 'cov': cluster_info['cov'],
#                 'count': cluster_info['count'],
#                 'points': cluster_data,
#                 'point_indices': cluster_points,
#                 'time_points': time_points,
#                 'cluster_id': cluster_id
#             }
            
#         return final_clusters

class DPCRP_Segmenter:
    """
    实现了论文中的DP-CRP（狄利克雷过程 - 中餐馆过程）聚类分割算法。
    该实现基于吉布斯采样，并假设每个簇的数据服从多元高斯分布。
    """

    def __init__(self, alpha=1.0):
        """
        初始化
        :param alpha: CRP的集中度参数。alpha越大，越倾向于生成新的簇。
        """
        self.alpha = alpha
        self.data = None
        self.assignments = []
        self.clusters = {} # key: cluster_id, value: {'mean': ..., 'cov': ..., 'points': ..., 'count': ...}
        self.n_features = 0

    def _add_to_cluster(self, point_idx, cluster_id):
        """将一个点添加到指定的簇"""
        point = self.data[point_idx]
        self.assignments[point_idx] = cluster_id

        if cluster_id not in self.clusters:
            self.clusters[cluster_id] = {
                'points': [],
                'count': 0,
                'mean': np.zeros(self.n_features),
                'cov': np.eye(self.n_features)
            }

        c = self.clusters[cluster_id]
        c['points'].append(point_idx)
        c['count'] += 1

        # 在线更新均值和协方差 (Welford's algorithm can be more stable, but this is simpler)
        if c['count'] == 1:
            c['mean'] = point
        else:
            old_mean = c['mean']
            c['mean'] = old_mean + (point - old_mean) / c['count']
            if c['count'] > 1:
                 # A simple but potentially unstable way to update covariance
                c['cov'] = np.cov(self.data[c['points']].T)

    def _remove_from_cluster(self, point_idx):
        """从一个点所属的簇中移除该点"""
        cluster_id = self.assignments[point_idx]
        point = self.data[point_idx]
        c = self.clusters[cluster_id]

        c['points'].remove(point_idx)
        c['count'] -= 1

        if c['count'] == 0:
            del self.clusters[cluster_id]
        else:
            # Re-calculate mean and covariance
            c['mean'] = np.mean(self.data[c['points']], axis=0)
            if c['count'] > 1:
                c['cov'] = np.cov(self.data[c['points']].T)


    def _log_likelihood(self, point, cluster_info):
        """计算一个点属于某个簇的对数似然概率"""
        mean = cluster_info['mean']
        # 加上一个小的对角矩阵防止协方差矩阵奇异
        cov = cluster_info['cov'] + 1e-6 * np.eye(self.n_features)
        return multivariate_normal.logpdf(point, mean=mean, cov=cov)

    def fit(self, data, max_iter=20, verbose=True):
        """
        使用吉布斯采样来拟合数据，实现算法1和算法2的过程。
        :param data: 运动学数据, shape (n_samples, n_features)
        :param max_iter: 吉布斯采样的最大迭代次数
        """
        self.data = data
        n_samples, self.n_features = data.shape
        # 处理一维时间数据的情况
        if len(data.shape) == 1:
            self.data = data.reshape(-1, 1)
            n_samples, self.n_features = self.data.shape

        self.assignments = [-1] * n_samples
        self.clusters = {} # 确保从一个空的聚类开始

        # 预先计算整个数据集的均值和协方差，作为新类别的先验
        data_mean = np.mean(self.data, axis=0)
        if self.n_features > 1 and n_samples > 1:
            data_cov = np.cov(self.data.T)
        else:
            # 处理一维或样本不足的情况
            data_cov = np.eye(self.n_features) * max(np.var(self.data), 1e-6)

        # ===================================================================
        # =================== 核心修改：方案B 顺序初始化 ==================
        # ===================================================================
        if verbose:
            print("--- Starting Sequential Initialization (Algorithm 1) ---")

        # 第一个点自成一派
        self._add_to_cluster(0, 0)

        # 依次加入后续的点 - 添加进度条
        if verbose:
            init_pbar = tqdm(range(1, n_samples), desc="Initialization Progress", 
                           bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]')
        else:
            init_pbar = range(1, n_samples)
            
        for i in init_pbar:
            # 更新进度条信息
            if verbose:
                init_pbar.set_postfix({"Point": i, "Clusters": len(self.clusters)})
            
            # 计算点 i 加入每个现有簇的概率
            log_probs = []
            cluster_ids = list(self.clusters.keys())
            for cid in cluster_ids:
                c = self.clusters[cid]
                # CRP 先验概率: p(z_i = k | z_-i) ∝ n_k
                log_prior = np.log(c['count'])
                log_lik = self._log_likelihood(self.data[i], c)
                log_probs.append(log_prior + log_lik)

            # 计算点 i 形成新簇的概率
            # CRP 先验概率: p(z_i = new | z_-i) ∝ alpha
            log_prior_new = np.log(self.alpha)
            log_lik_new = multivariate_normal.logpdf(self.data[i],
                                                    mean=data_mean,
                                                    cov=data_cov + 1e-6 * np.eye(self.n_features))
            log_probs.append(log_prior_new + log_lik_new)

            # 归一化并随机选择一个簇
            # 使用 log-sum-exp 技巧避免数值下溢
            log_probs = np.array(log_probs)
            probs = np.exp(log_probs - np.max(log_probs))
            probs /= probs.sum()

            assignment_idx = np.random.choice(len(probs), p=probs)

            # 执行分配
            if assignment_idx == len(cluster_ids): # A new cluster was chosen
                new_cid = max(self.clusters.keys()) + 1 if self.clusters else 0
                self._add_to_cluster(i, new_cid)
            else: # An existing cluster was chosen
                self._add_to_cluster(i, cluster_ids[assignment_idx])
        
        if verbose:
            print(f"Initialization complete. Found {len(self.clusters)} initial clusters.")
        # ===================================================================
        # ======================= 初始化结束 ========================
        # ===================================================================


        # --- 吉布斯采样优化 (Algorithm 2) ---
        if verbose:
            gibbs_pbar = tqdm(range(max_iter), desc="Gibbs Sampling", 
                            bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]')
        else:
            gibbs_pbar = range(max_iter)
            
        for it in gibbs_pbar:
            if verbose:
                gibbs_pbar.set_postfix({"Iteration": f"{it + 1}/{max_iter}", "Clusters": len(self.clusters)})
            
            # using tqdm 
            for i in tqdm(range(n_samples), desc="Gibbs Sampling", 
                            bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]'):
                self._remove_from_cluster(i)

                # 计算分配到现有簇的概率
                log_probs = []
                cluster_ids = list(self.clusters.keys())
                for cid in cluster_ids:
                    c = self.clusters[cid]
                    log_prior = np.log(c['count'])
                    log_lik = self._log_likelihood(self.data[i], c)
                    log_probs.append(log_prior + log_lik)

                # 计算分配到新簇的概率
                log_prior_new = np.log(self.alpha)
                log_lik_new = multivariate_normal.logpdf(self.data[i],
                                                        mean=data_mean,
                                                        cov=data_cov + 1e-6 * np.eye(self.n_features))
                log_probs.append(log_prior_new + log_lik_new)

                # 归一化并选择新簇
                log_probs = np.array(log_probs)
                probs = np.exp(log_probs - np.max(log_probs))
                probs /= probs.sum()

                new_assignment_idx = np.random.choice(len(probs), p=probs)

                if new_assignment_idx == len(cluster_ids): # A new cluster was chosen
                    new_cid = max(self.clusters.keys()) + 1 if self.clusters else 0
                    self._add_to_cluster(i, new_cid)
                else: # An existing cluster was chosen
                    self._add_to_cluster(i, cluster_ids[new_assignment_idx])

            if verbose:
                print(f"  Number of clusters: {len(self.clusters)}")

    def get_segmentation(self):
        """
        从聚类结果中提取分段信息。
        返回一个列表，每个元素是 (start_frame, end_frame, cluster_id)
        """
        if not self.assignments:
            return []

        segments = []
        current_cluster = self.assignments[0]
        start_frame = 0
        for i in range(1, len(self.assignments)):
            if self.assignments[i] != current_cluster:
                segments.append((start_frame, i - 1, current_cluster))
                start_frame = i
                current_cluster = self.assignments[i]

        segments.append((start_frame, len(self.assignments) - 1, current_cluster))
        return segments

    def get_final_clusters(self):
        """
        生成用于可视化的final_clusters结果，类似TSC.py中的格式。
        返回一个字典，包含每个簇的详细信息。
        """
        if not self.assignments or not self.clusters:
            return {}

        final_clusters = {}

        # 为每个簇生成详细信息
        for cluster_id, cluster_info in self.clusters.items():
            # 获取属于该簇的所有点
            cluster_points = cluster_info['points']
            if not cluster_points: continue
            cluster_data = self.data[cluster_points]

            # 计算簇中心（均值）
            cluster_center = cluster_info['mean']

            # 计算时间信息（假设数据是按时间顺序的）
            time_points = np.array(cluster_points)

            # 生成簇的键名
            key = f"Cluster_{cluster_id}"

            final_clusters[key] = {
                'center': cluster_center,
                'mean': cluster_center,  # 保持兼容性
                'cov': cluster_info['cov'],
                'count': cluster_info['count'],
                'points': cluster_data,
                'point_indices': cluster_points,
                'time_points': time_points,
                'cluster_id': cluster_id
            }

        return final_clusters


# ===================================================================
# =================== 新增的分层聚类功能 ===================
# ===================================================================
class Hierarchical_DPCRP_Segmenter:
    """
    实现分层聚类：首先基于运动学进行聚类，然后在每个运动学簇内基于时间进行二次聚类。
    这个类将调用原有的DPCRP_Segmenter来完成每个阶段的聚类任务。
    """
    def __init__(self, alpha_kinematic=1.0, alpha_time=1.0):
        """
        初始化
        :param alpha_kinematic: 运动学聚类的集中度参数
        :param alpha_time: 时间聚类的集中度参数
        """
        self.alpha_kinematic = alpha_kinematic
        self.alpha_time = alpha_time
        self.final_assignments = []
        self.final_clusters = {}

    def fit(self, kinematic_data, time_data, max_iter=10):
        """
        执行分层聚类。
        :param kinematic_data: 运动学数据
        :param time_data: 时间数据
        :param max_iter: 每次聚类的最大迭代次数
        """
        n_samples = kinematic_data.shape[0]
        self.final_assignments = np.zeros(n_samples, dtype=int)
        
        # --- 第1层: 运动学聚类 ---
        print("\n--- Starting Hierarchical Level 1: Kinematic Clustering ---")
        kinematic_segmenter = DPCRP_Segmenter(alpha=self.alpha_kinematic)
        kinematic_segmenter.fit(kinematic_data, max_iter)
        kinematic_clusters = kinematic_segmenter.get_final_clusters()
        print(f"Level 1 found {len(kinematic_clusters)} kinematic clusters.")

        # --- 第2层: 在每个运动学簇内进行时间聚类 ---
        print("\n--- Starting Hierarchical Level 2: Time Sub-Clustering ---")
        global_cluster_id_counter = 0
        
        # 按簇ID排序，保证处理顺序一致
        sorted_kinematic_clusters = sorted(kinematic_clusters.values(), key=lambda c: c['cluster_id'])

        # 为第二层聚类添加进度条
        level2_pbar = tqdm(enumerate(sorted_kinematic_clusters), 
                          total=len(sorted_kinematic_clusters),
                          desc="Level 2: Time Sub-Clustering",
                          bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]')

        for i, k_cluster_info in level2_pbar:
            point_indices = np.array(k_cluster_info['point_indices'])
            
            # 更新进度条信息
            level2_pbar.set_postfix({
                "Cluster": k_cluster_info['cluster_id'], 
                "Points": len(point_indices),
                "Global_ID": global_cluster_id_counter
            })
            
            print(f"\nProcessing Kinematic Cluster {k_cluster_info['cluster_id']} (contains {len(point_indices)} points)")

            if len(point_indices) < 3: # 样本太少无法聚类
                self.final_assignments[point_indices] = global_cluster_id_counter
                global_cluster_id_counter += 1
                continue

            # 提取该簇对应的时间数据
            time_subset = time_data[point_indices]

            # 对时间子集进行聚类
            time_segmenter = DPCRP_Segmenter(alpha=self.alpha_time)
            # 在子簇中进行聚类，迭代次数可以减少，且不打印内部过程
            time_segmenter.fit(time_subset, max_iter=max(1, max_iter//2), verbose=False)

            # 获取子聚类的分配结果
            sub_assignments = np.array(time_segmenter.assignments)

            # 将局部分配ID映射到全局唯一的ID
            unique_sub_ids = np.unique(sub_assignments)
            print(f"  Sub-clustered into {len(unique_sub_ids)} time-based segments.")

            for sub_id in unique_sub_ids:
                # 找到在子集内部分配为sub_id的点
                mask = (sub_assignments == sub_id)
                # 找到这些点在原始数据集中的索引
                original_indices = point_indices[mask]
                # 为这些点分配一个新的全局ID
                self.final_assignments[original_indices] = global_cluster_id_counter
                global_cluster_id_counter += 1

            time_cluster_info = time_segmenter.get_final_clusters()
            for key, value in time_cluster_info.items():
                global_key = f"KinematicCluster_{k_cluster_info['cluster_id']}-TimeCluster_{value['cluster_id']}"
                value['cluster_id'] = global_key
                self.final_clusters[global_key] = value
        return self.final_clusters

    def get_segmentation(self):
        """
        从最终的分层聚类结果中提取分段信息。
        """
        if len(self.final_assignments) == 0:
            return []

        segments = []
        current_cluster = self.final_assignments[0]
        start_frame = 0
        for i in range(1, len(self.final_assignments)):
            if self.final_assignments[i] != current_cluster:
                segments.append((start_frame, i - 1, current_cluster))
                start_frame = i
                current_cluster = self.final_assignments[i]
        
        segments.append((start_frame, len(self.final_assignments) - 1, current_cluster))
        return segments
# ===================================================================
# =================== 结束新增功能部分 ======================
# ===================================================================

class PDMD_Merger:
    """
    实现了论文中的PDMD后处理算法，用于合并过度分割的片段。
    基于PCA, DFD, MI, DTW四种相似性度量。
    """

    def __init__(self, threshold_tau=0.7, n_pca_components=3, n_mi_bins=10):
        """
        初始化
        :param threshold_tau: 合并阈值，相似度得分高于此值的片段将被合并。 
        :param n_pca_components: PCA相似性分析中使用的主成分数量。
        :param n_mi_bins: 计算互信息时，用于数据离散化的箱数。
        """
        self.tau = threshold_tau
        self.q = n_pca_components
        self.n_mi_bins = n_mi_bins

    def _get_subspace_angle(self, A, B):
        """计算两个子空间之间的角度，用于PCA相似度"""
        A, _ = np.linalg.qr(A)
        B, _ = np.linalg.qr(B)
        M = A.T @ B
        s = np.linalg.svd(M, compute_uv=False)
        angles = np.arccos(np.clip(s, -1.0, 1.0))
        return np.mean(angles)

    def _calculate_pca_similarity(self, seg1_data, seg2_data):
        #
        if len(seg1_data) <= self.q or len(seg2_data) <= self.q: return np.pi / 2
        pca1 = PCA(n_components=self.q).fit(seg1_data)
        pca2 = PCA(n_components=self.q).fit(seg2_data)
        return self._get_subspace_angle(pca1.components_.T, pca2.components_.T)

    def _calculate_dfd_similarity(self, seg1_data, seg2_data):
        # 
        return similaritymeasures.frechet_dist(seg1_data, seg2_data)

    def _calculate_mi_similarity(self, seg1_data, seg2_data):
        # 
        mi = 0
        min_len = min(len(seg1_data), len(seg2_data))
        seg1_data, seg2_data = seg1_data[:min_len], seg2_data[:min_len]
        
        for feat_idx in range(seg1_data.shape[1]):
            # 离散化
            all_vals = np.concatenate((seg1_data[:, feat_idx], seg2_data[:, feat_idx]))
            bins = np.linspace(all_vals.min(), all_vals.max(), self.n_mi_bins + 1)
            s1_binned = np.digitize(seg1_data[:, feat_idx], bins)
            s2_binned = np.digitize(seg2_data[:, feat_idx], bins)
            mi += mutual_info_score(s1_binned, s2_binned)
        return mi / seg1_data.shape[1] # Average MI over all features

    def _calculate_dtw_similarity(self, seg1_data, seg2_data):
        # ===================================================================
        # ======================= 此处为核心修改 ==========================
        # ===================================================================
        # 1. 使用您提供的dtw.py中的 accelerated_dtw 函数，效率更高。
        # 2. 距离参数 dist 直接使用字符串 'euclidean'。
        # 3. 函数返回一个元组，我们只取第一个元素（最小距离）。
        dist, _, _, _ = accelerated_dtw(seg1_data, seg2_data, dist='euclidean')
        # ===================================================================
        return dist

    def _normalize_scores(self, scores_dict):
        """根据论文公式(20)和(21)进行归一化"""
        norm_scores = {
            'pca': [], 'dfd': [], 'dtw': [], 'mi': []
        }
        
        # For negative correlations (lower is better)
        for key in ['pca', 'dfd', 'dtw']:
            s = np.array(scores_dict[key])
            if len(s) == 0: continue
            mean_s, min_s = np.mean(s), np.min(s)
            # Avoid division by zero
            if mean_s == min_s:
                norm_scores[key] = np.zeros_like(s)
                continue
            norm = np.where(s >= mean_s, 0, (mean_s - s) / (mean_s - min_s))
            norm_scores[key] = np.clip(norm, 0, 1)

        # For positive correlation (higher is better)
        s_mi = np.array(scores_dict['mi'])
        if len(s_mi) > 0:
            mean_s, max_s = np.mean(s_mi), np.max(s_mi)
            if max_s == mean_s:
                norm_scores['mi'] = np.zeros_like(s_mi)
            else:
                norm = np.where(s_mi <= mean_s, 0, (s_mi - mean_s) / (max_s - mean_s))
                norm_scores['mi'] = np.clip(norm, 0, 1)

        return norm_scores

    def merge(self, data, initial_segments):
        """
        执行PDMD合并过程，实现算法3。 
        :param data: 完整的运动学数据
        :param initial_segments: DP-CRP输出的初始分段列表
        :return: 合并后的分段列表
        """
        segments = list(initial_segments)
        
        while True:
            if len(segments) <= 1:
                break

            # 1. 计算所有相邻片段的相似度
            all_scores = {'pca': [], 'dfd': [], 'mi': [], 'dtw': []}
            for i in range(len(segments) - 1):
                s1_start, s1_end, _ = segments[i]
                s2_start, s2_end, _ = segments[i+1]
                
                seg1_data = data[s1_start : s1_end + 1]
                seg2_data = data[s2_start : s2_end + 1]
                
                if seg1_data.shape[0] == 0 or seg2_data.shape[0] == 0: continue

                all_scores['pca'].append(self._calculate_pca_similarity(seg1_data, seg2_data))
                all_scores['dfd'].append(self._calculate_dfd_similarity(seg1_data, seg2_data))
                all_scores['mi'].append(self._calculate_mi_similarity(seg1_data, seg2_data))
                all_scores['dtw'].append(self._calculate_dtw_similarity(seg1_data, seg2_data))
            
            if not all_scores['pca']: # 如果没有可以比较的片段，则停止
                break

            # 2. 归一化并聚合
            norm_scores = self._normalize_scores(all_scores)
            
            # 聚合 (公式22) 
            T = np.sqrt(
                (np.array(norm_scores['pca'])**2 + np.array(norm_scores['dfd'])**2 + np.array(norm_scores['mi'])**2 + np.array(norm_scores['dtw'])**2) / 4
            )
            
            # 3. 找到最相似的片段并判断是否合并
            max_similarity = np.max(T)
            if max_similarity < self.tau:
                print(f"Max similarity {max_similarity:.4f} is below threshold {self.tau}. Stopping.")
                break # 没有需要合并的了
            
            merge_idx = np.argmax(T)
            print(f"Merging segment {merge_idx} and {merge_idx+1} with similarity {max_similarity:.4f}")
            
            # 4. 执行合并
            s1_start, _, s1_label = segments[merge_idx]
            _, s2_end, _ = segments[merge_idx + 1]
            
            # 合并后的片段继承第一个片段的标签
            merged_segment = (s1_start, s2_end, s1_label)
            
            # 更新片段列表
            segments.pop(merge_idx)
            segments[merge_idx] = merged_segment
            
        return segments

    def get_final_clusters(self, data, final_segments):
        """
        从最终的分段结果生成用于可视化的final_clusters结果。
        :param data: 完整的运动学数据
        :param final_segments: 合并后的分段列表
        :return: 字典，包含每个簇的详细信息
        """
        final_clusters = {}
        
        for i, (start_frame, end_frame, cluster_id) in enumerate(final_segments):
            # 获取该分段的数据
            segment_data = data[start_frame:end_frame + 1]
            time_points = np.arange(start_frame, end_frame + 1)
            
            # 计算该分段的统计信息
            segment_mean = np.mean(segment_data, axis=0)
            segment_cov = np.cov(segment_data.T) if len(segment_data) > 1 else np.eye(segment_data.shape[1])
            
            # 生成簇的键名
            key = f"Segment_{i}_Cluster_{cluster_id}"
            
            final_clusters[key] = {
                'center': segment_mean,
                'mean': segment_mean,  # 保持兼容性
                'cov': segment_cov,
                'count': len(segment_data),
                'points': segment_data,
                'point_indices': np.arange(start_frame, end_frame + 1),
                'time_points': time_points,
                'cluster_id': cluster_id,
                'start_frame': start_frame,
                'end_frame': end_frame,
                'segment_id': i
            }
            
        return final_clusters


# --- 主函数：演示 ---
if __name__ == '__main__':

    file_path = os.path.join(os.path.dirname(__file__), 'cluster', TSC_model_path)
    # 假设 tsc_model_state 是 tsc_model 的一个属性
    tsc_model_state = load_model(TSC, file_path)


    # 1. 加载真实数据集
    print("--- Loading Real Dataset ---")

    kinematic_data = np.vstack(tsc_model_state.remaining_states)
    time_data = np.vstack(tsc_model_state.remaining_times)

    print(f"Loaded kinematic data shape: {kinematic_data.shape}")
    print(f"Loaded time data shape: {time_data.shape}")

    # ===================================================================
    # =================== 使用新的分层聚类器 ======================
    # ===================================================================
    print("\n--- Starting Hierarchical DP-CRP Segmentation ---")
    # ===================================================================
    # ===================== 修改核心：调整超参数 ======================
    # ===================================================================
    # 尝试增大 alpha 值以鼓励模型创建更多的簇
    hierarchical_segmenter = Hierarchical_DPCRP_Segmenter(alpha_kinematic=0.1, alpha_time=0.1)
    # ===================================================================
    final_clusters = hierarchical_segmenter.fit(kinematic_data, time_data, max_iter=10)
    initial_segments = hierarchical_segmenter.get_segmentation()
    
    print("\n--- Initial Segmentation (Before PDMD) ---")
    print(f"Found {len(initial_segments)} segments from hierarchical clustering:")
    for seg in initial_segments:
        print(f"  Segment from frame {seg[0]} to {seg[1]} (Label: {seg[2]})")


    # ===================================================================
    # ============ 后续的合并流程保持不变，使用原有的类 ============
    # ===================================================================
    # print("\n--- Starting PDMD Post-processing ---")
    # ===================================================================
    # ===================== 修改核心：调整超参数 ======================
    # ===================================================================
    # 增大 tau 值 will 使得合并条件更严格，
    # merger = PDMD_Merger(threshold_tau=1)
    # # ===================================================================
    # final_segments = merger.merge(kinematic_data, initial_segments)

    # print("\n--- Final Segmentation (After PDMD) ---")
    # print(f"Found {len(final_segments)} segments:")
    # for seg in final_segments:
    #     print(f"  Segment from frame {seg[0]} to {seg[1]} (Label: {seg[2]})")

    # # 4. 生成用于可视化的final_clusters结果
    # print("\n--- Generating Final Clusters for Visualization ---")
    # final_clusters = merger.get_final_clusters(kinematic_data, final_segments)


    print(f"\nFinal PDMD clusters: {len(final_clusters)}")
    for key, cluster_info in final_clusters.items():
        print(f"  {key}: {cluster_info['count']} points, center shape: {cluster_info['center'].shape}")
        print(f"  Cluster ID: {cluster_info['cluster_id']}")
    
    print("\n--- Hierarchical DP-CRP + PDMD Processing Complete ---")
    print("Final clusters are ready for visualization!")
    visualize_tsc_results(final_clusters)