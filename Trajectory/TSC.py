import numpy as np
from sklearn.mixture import BayesianGaussianMixture
from dtw import dtw
import warnings
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from load_data import load_demonstrations_state
import pickle
import os
from config import p, fp, delta, n_regimes, n_state_clusters, n_time_clusters, TSC_model_path
from sklearn.preprocessing import StandardScaler

# 忽略来自 scikit-learn 的关于收敛性的警告
warnings.filterwarnings("ignore", category=UserWarning)

class TSC:
    """
    Transition State Clustering (TSC) 算法的 Python 实现。
    该算法旨在从多个示教轨迹中无监督地分割出有意义的片段。

    参考文献:
    Krishnan, S., Garg, A., Patil, S., Lea, C., Hager, G., Abbeel, P., & Goldberg, K. (2017).
    Transition state clustering: Unsupervised surgical trajectory segmentation for robot learning.

    参数:
    p (float): 剪枝参数，用于移除罕见动态模式相关的转移。
               一个模式(regime)如果出现在少于 p * N (N为总示教数) 个示教中，则被认为是罕见的。 
    delta (float): 压缩参数，用于合并循环(looping)动作。
                     如果两个连续且类型相同的转移段之间的DTW L2距离小于 delta，它们将被合并。 [cite: 207, 217]
    n_regimes (int): 用于识别动态模式的DP-GMM的最大聚类数。
    n_state_clusters (int): 用于状态空间聚类的DP-GMM的最大聚类数。
    n_time_clusters (int): 用于时间聚类的DP-GMM的最大聚类数。
    """
    def __init__(self, p=0.8, fp=0.8, delta=0.1, n_regimes=20, n_state_clusters=20, n_time_clusters=10):
        self.p = p
        self.fp = fp
        self.delta = delta
        
        # 为DP-GMM模型设置较高的组件数
        self.dpgmm_regime = BayesianGaussianMixture(
            n_components=n_regimes, weight_concentration_prior_type='dirichlet_process',weight_concentration_prior=0.01,
            random_state=42, max_iter=300
        )
        self.dpgmm_state = BayesianGaussianMixture(
            n_components=n_state_clusters, weight_concentration_prior_type='dirichlet_process',weight_concentration_prior=0.01,
            random_state=42, max_iter=300
        )
        self.dpgmm_time = BayesianGaussianMixture(
            n_components=n_time_clusters, weight_concentration_prior_type='dirichlet_process',weight_concentration_prior=0.1,
            random_state=42, max_iter=300
        )
        
        # 新增：保存时间聚类模型
        self.time_cluster_models = {}
        
        # 新增：保存训练数据
        self.remaining_states = None
        self.remaining_times = None
        self.remaining_indices = None
        self.remaining_regime_labels = None
    
    def fit(self, demonstrations):
        """训练TSC模型"""
        print("step 1/6: prepare augmented states n(t)...")
        augmented_states, demo_indices = self._prepare_augmented_states(demonstrations)
        
        print("step 2/6: identify dynamic modes (DP-GMM)...")
        regime_labels = self.dpgmm_regime.fit_predict(augmented_states)
        
        print("step 3/6: identify and prune transition states...")
        transition_data = self._identify_and_prune_transitions(demonstrations, regime_labels, demo_indices)

        print("step 4/6: compact transition states (handle looping)...")
        compacted_transitions = self._compact_transitions(demonstrations, transition_data)
        
        print("step 5/6: cluster transition states in state space...")
        self.remaining_states = np.array([t['state'] for t in compacted_transitions])
        self.remaining_times = np.array([t['time'] for t in compacted_transitions]).reshape(-1, 1)
        self.remaining_indices = np.array([t['demo_idx'] for t in compacted_transitions])
        
        state_cluster_labels = self.dpgmm_state.fit_predict(self.remaining_states)
        
        print("step 6/6: cluster transition states in time space...")
        final_clusters = self._cluster_time_within_state_clusters(
            self.remaining_states, self.remaining_times, state_cluster_labels, self.remaining_indices
        )
        
        print("\nTSC algorithm has been executed successfully.")
        self.final_clusters = final_clusters
        return final_clusters
    
    def _prepare_augmented_states(self, demonstrations):
        """构建增广状态向量 n(t) = [x(t+1), x(t)] 用于模式识别。 [cite: 189]"""
        augmented_states_list = []
        demo_indices = []
        for i, demo in enumerate(demonstrations):
            # aug_states 的维度是 [T, 2 * D]
            aug_states = np.hstack([demo[1:], demo[:-1]])
            augmented_states_list.append(aug_states)
            demo_indices.extend([i] * len(aug_states))
        return np.vstack(augmented_states_list), np.array(demo_indices)

    def _identify_and_prune_transitions(self, demonstrations, regime_labels, demo_indices):
        """识别所有转移，并根据参数 p 进行剪枝。"""
        # 1. 统计每个模式(regime)的支持示教数量
        num_demos = len(demonstrations)
        regime_support = {}
        unique_regimes = np.unique(regime_labels) # [0,1,2,...,k] not demo label 
        for r in unique_regimes:
            demos_with_regime = np.unique(demo_indices[regime_labels == r]) # every regime contains which demos
            regime_support[r] = len(demos_with_regime)

        # 2. 识别需要剪枝的模式
        min_support = self.p * num_demos
        regimes_to_prune = {r for r, count in regime_support.items() if count < min_support}
        self.remaining_regime_labels = [r for r in unique_regimes if r not in regimes_to_prune]
        print(f"  - found {len(unique_regimes)} dynamic modes.")
        print(f"  - prune {len(regimes_to_prune)} modes according to pruning parameter p={self.p}.")
        
        # 3. 识别转移并应用剪枝
        transition_data = []
        start_idx = 0
        for i, demo in enumerate(demonstrations):
            demo_len = len(demo) - 1
            labels = regime_labels[start_idx : start_idx + demo_len]
            
            for t in range(demo_len - 1):
                if labels[t] != labels[t+1]:
                    # 检查转移是否涉及到被剪枝的模式
                    if labels[t] in regimes_to_prune or labels[t+1] in regimes_to_prune:
                        continue
                    
                    transition_data.append({
                        'demo_idx': i,
                        'time': t,
                        'state': demo[t],
                        'from_regime': labels[t],
                        'to_regime': labels[t+1]
                    })
            start_idx += demo_len
            
        print(f"  - after pruning, remaining {len(transition_data)} transition states.")
        return transition_data

    def _compact_transitions(self, demonstrations, transition_data):
        """根据 delta 参数，通过 DTW 压缩循环动作。"""
        compacted_transitions = []
        
        for i in range(len(demonstrations)):
            demo_transitions = sorted([t for t in transition_data if t['demo_idx'] == i], key=lambda x: x['time'])
            
            if not demo_transitions:
                continue

            is_compacted = [False] * len(demo_transitions)
            # 修正：我们将保留要添加的转移，而不是在循环中直接添加
            # 这样可以正确处理最后一个元素
            transitions_to_keep = [True] * len(demo_transitions)

            for j in range(len(demo_transitions) - 1):
                t1 = demo_transitions[j]
                t2 = demo_transitions[j+1]

                is_repeated_transition = (t1['from_regime'] == t2['from_regime'] and t1['to_regime'] == t2['to_regime'])

                if is_repeated_transition:
                    demo = demonstrations[i]
                    start1, end1 = t1['time'], t2['time']
                    
                    end2 = len(demo) -1
                    if j + 2 < len(demo_transitions):
                        end2 = demo_transitions[j+2]['time']
                    
                    segment1 = demo[start1:end1]
                    segment2 = demo[end1:end2]

                    if len(segment1) > 0 and len(segment2) > 0:
                        # --- 错误修正处 ---
                        # 之前: dist = dtw(segment1, segment2)[0] / ...
                        # 修正后: 明确传入距离计算方法
                        distfunc = lambda u, v: np.linalg.norm(u - v)
                        dtw_result = dtw(segment1, segment2, dist=distfunc)
                        
                        dist = dtw_result[0] / (len(segment1) + len(segment2))
                        # --- 修正结束 ---
                        
                        if dist < self.delta:
                            # 距离小于阈值，标记第二个转移为不保留
                            transitions_to_keep[j+1] = False
            
            # 根据标记添加最终的转移
            for j, keep in enumerate(transitions_to_keep):
                if keep:
                    compacted_transitions.append(demo_transitions[j])
            
        print(f"  - after compacting, remaining {len(compacted_transitions)} transition states.")
        return compacted_transitions
        
    def _cluster_time_within_state_clusters(self, states, times, state_cluster_labels, demo_indices):
        """修改后的时间聚类方法，保存每个状态簇的时间聚类模型"""
        final_clusters = {}
        if state_cluster_labels.size == 0:
            return final_clusters

        unique_state_labels = np.unique(state_cluster_labels)
        print(f"  - found {len(unique_state_labels)} main state clusters.")
        print(f"unique_state_labels: {unique_state_labels}")
        
        for state_label in unique_state_labels:
            mask = (state_cluster_labels == state_label)
            
            n_samples_in_cluster = np.sum(mask)
            if n_samples_in_cluster == 0: 
                continue

            times_in_cluster = times[mask]
            states_in_cluster = states[mask]
            demos_in_cluster = demo_indices[mask]
            
            if n_samples_in_cluster <= 1:
                time_labels = np.zeros(n_samples_in_cluster, dtype=int)
                self.time_cluster_models[state_label] = None  # 保存None表示不需要聚类
            else:
                # 为每个状态簇创建独立的时间聚类模型
                temp_dpgmm_time = BayesianGaussianMixture(
                    n_components=min(self.dpgmm_time.n_components, n_samples_in_cluster),
                    weight_concentration_prior_type='dirichlet_process',
                    random_state=42, max_iter=300
                )
                time_labels = temp_dpgmm_time.fit_predict(times_in_cluster)
                # print(f"state_label: {state_label}'s DP-GMM模型信息:")
                # print(f"  n_components_: {temp_dpgmm_time.n_components}")
                # print(f"  weights_: {temp_dpgmm_time.weights_}")
                # print(f"  means_ shape: {temp_dpgmm_time.means_.shape}")
                
                # 保存这个状态簇的时间聚类模型
                self.time_cluster_models[state_label] = temp_dpgmm_time
            
            # 剪枝逻辑
            unique_time_labels = np.unique(time_labels)
            print(f"state_label: {state_label}, time labels count: {len(unique_time_labels)}")
            print(f"time labels: {unique_time_labels}")
            valid_time_labels = []
            for time_label in unique_time_labels:
                time_mask = (time_labels == time_label)
                count = np.sum(time_mask)

                # 统计该时间簇中包含的示教轨迹数量
                demos_in_time_cluster = np.unique(demos_in_cluster[time_mask])
                num_demos_in_time_cluster = len(demos_in_time_cluster)

                # 计算示教轨迹数量占比
                proportion = num_demos_in_time_cluster / len(np.unique(demo_indices))
                if proportion < self.fp:
                    print(f"    - StateCluster_{state_label}-TimeCluster_{time_label} is pruned (the proportion of demos in the time cluster {proportion:.2f} < {self.fp}).")
                    continue

                # 保留未被剪枝的簇
                valid_time_labels.append(time_label)
                cluster_center_state = np.mean(states_in_cluster[time_mask], axis=0)
                cluster_center_time = np.mean(times_in_cluster[time_mask])
                print(f"state shape: {states_in_cluster[time_mask].shape}")
                print(f"time shape: {times_in_cluster[time_mask].shape}")
                key = f"StateCluster_{state_label}-TimeCluster_{time_label}"
                final_clusters[key] = {
                    'state_center': cluster_center_state,
                    'time_center': cluster_center_time,
                    'count': count,
                    'points': states_in_cluster[time_mask],
                    'time_points': times_in_cluster[time_mask],
                    'demo_points': demos_in_cluster[time_mask],
                    'demos': demos_in_time_cluster
                }

            # 重新分配时间簇标签
            # valid_time_labels.sort()
            # time_label_mapping = {old_label: new_label for new_label, old_label in enumerate(valid_time_labels)}
            # for old_key in list(final_clusters.keys()):
            #     if f"StateCluster_{state_label}-TimeCluster_" in old_key:
            #         old_time_label = int(old_key.split('-TimeCluster_')[-1])
            #         if old_time_label in time_label_mapping:
            #             new_time_label = time_label_mapping[old_time_label]
            #             new_key = f"StateCluster_{state_label}-TimeCluster_{new_time_label}"
            #             final_clusters[new_key] = final_clusters.pop(old_key)

        return final_clusters
    
    def save_model(self, filepath):
        """保存TSC模型到文件"""
        # 获取TSC.py文件所在目录
        current_dir = os.path.dirname(__file__)
        # 构建完整路径
        full_path = os.path.join(current_dir,'cluster', filepath)
        
        model_data = {
            'p': self.p,
            'fp': self.fp,
            'delta': self.delta,
            'dpgmm_regime': self.dpgmm_regime,
            'dpgmm_state': self.dpgmm_state,
            'dpgmms_time': self.time_cluster_models,
            'remaining_states': self.remaining_states,
            'remaining_times': self.remaining_times,
            'remaining_indices': self.remaining_indices,
            'remaining_regime_labels': self.remaining_regime_labels,
            'final_clusters': self.final_clusters
        }
        
        # 确保目录存在
        os.makedirs(os.path.dirname(full_path), exist_ok=True)
        
        with open(full_path, 'wb') as f:
            pickle.dump(model_data, f)
        
        print(f"TSC model has been saved to {full_path} successfully")
    

if __name__ == '__main__':

    # 1. 生成合成数据
    # print("正在生成合成数据...")
    # num_demos = 20
    # num_steps = 100
    # demos = generate_synthetic_data(num_demos=num_demos, noise_std=0.01, loop_prob=0.05)
    # switch_points = [num_steps // 2] * num_demos  # 每条轨迹的切换点

    demos = load_demonstrations_state()

    # 90% of demos act as training data
    demos = demos[:122]


    scaler = StandardScaler()

    all_data_stacked = np.vstack(demos)
    scaler.fit(all_data_stacked)

    # b) 现在，对列表中的每个演示独立进行 transform
    #    结果是一个新的列表，包含了标准化后的数据，结构和原来一样
    scaled_demos = [scaler.transform(arr) for arr in demos]


    
    # 2. 初始化并运行 TSC 算法
    # 根据论文，p=0.8 是一个合理的默认值 [cite: 306]
    # delta 的值需要根据具体任务的尺度来经验性地设置 [cite: 217]
    tsc_model = TSC(p=p, fp=fp, delta=delta, n_regimes=n_regimes, n_state_clusters=n_state_clusters, n_time_clusters=n_time_clusters)
    clusters = tsc_model.fit(scaled_demos)

    tsc_model.save_model(TSC_model_path)
    
    # 3. 打印结果
    print("\nfound final transition clusters:")
    # 按时间顺序排序并打印
    sorted_clusters = sorted(clusters.items(), key=lambda item: item[1]['time_center'])
    for name, data in sorted_clusters:
        state_str = np.array2string(data['state_center'], precision=2, floatmode='fixed')
        print(f"  - cluster: {name}")
        print(f"    - average time: {data['time_center']:.2f}")
        print(f"    - state center: {state_str}")
        print(f"    - number of points: {data['count']}")

    # 4. 可视化 (可选)
    #visualize_tsc_results

