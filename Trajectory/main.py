import numpy as np
from sklearn.mixture import BayesianGaussianMixture
from dtw import dtw
import warnings
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from load_data import load_demonstrations_state
from TSC import TSC
import rospy
import numpy as np
from std_msgs.msg import Float64MultiArray  
import os
import pickle
from config import state_probability_threshold, time_probability_threshold, TSC_model_path



tsc_model = None
data_buffer = []  # 存储所有接收到的数据
max_buffer_size = 1000  # 最大缓冲区大小
transition_threshold = 0.8  # 转移状态检测阈值
potential_state_list = []
regime_buffer = []



class KinematicDataSubscriber:
    def __init__(self):
        rospy.init_node('kinematic_data_subscriber')
        self.data = None
        
        # 订阅话题
        self.sub = rospy.Subscriber('kinematic_data', Float64MultiArray, self.callback)
        
    def callback(self, msg):
        # 将消息数据转换为numpy数组
        self.data = np.array(msg.data).reshape(1, -1)  # 确保是1行14列
        potential_state =realtime_TSC(self.data)
        if potential_state is not None:
            potential_state_list.append(potential_state)
        
    def get_latest_data(self):
        return self.data


def realtime_TSC(data):
    """
    实时TSC处理函数
    
    参数:
    - data: 当前时刻的状态向量 (14,)
    """
    global tsc_model, data_buffer, regime_buffer, potential_state_list, transition_threshold, max_buffer_size
    print("--------------------------------")
    
    # 检查模型是否已加载
    if tsc_model is None:
        print("error: TSC model is not loaded, please train or load the model")
        return None
    
    # 1. 将新数据添加到缓冲区
    data_buffer.append(data.copy())
    
    # 限制缓冲区大小
    if len(data_buffer) > max_buffer_size:
        data_buffer.pop(0)  # 移除最旧的数据
    
    print(f"current buffer size: {len(data_buffer)}")
    
    # 2. 检查是否有足够的数据进行增广向量构造
    if len(data_buffer) < 2:
        print("First data point is received")
        return None
    
    # 3. 构造增广向量 n(t) = [x(t+1), x(t)] = [current, previous]
    current_state = data_buffer[-1]  # 当前状态
    previous_state = data_buffer[-2]  # 前一个状态
    augmented_state = np.hstack([current_state, previous_state])  # 形状: (28,)
    
    
    # 4. 使用regime DP-GMM判断是否为transition state
    regime_probabilities = tsc_model.dpgmm_regime.predict_proba(augmented_state.reshape(1, -1))[0]
    max_regime_prob = np.max(regime_probabilities)
    
    print(f"Regime probabilities: {np.round(regime_probabilities, 3)}")
    print(f"Max Regime probability: {max_regime_prob:.3f}")

    regime_buffer.append(np.argmax(regime_probabilities))
    if len(regime_buffer) < 2:
        print("Second data point is received")
        return None
    
    
    
    # 5. 判断是否为transition state
    if regime_buffer[-1] != regime_buffer[-2] and regime_buffer[-1] in tsc_model.remaining_regime_labels and regime_buffer[-2] in tsc_model.remaining_regime_labels:
        print("detect a transition state!")
        
        # 6. 使用state DP-GMM进行空间聚类
        state_probabilities = tsc_model.dpgmm_state.predict_proba(current_state.reshape(1, -1))[0]
        state_cluster = np.argmax(state_probabilities)
        state_confidence = np.max(state_probabilities)

        if state_confidence < state_probability_threshold:
            print(f"State confidence {state_confidence:.3f} is less than the threshold {state_probability_threshold:.3f}, return None")
            return None
        
        print(f"State probabilities: {np.round(state_probabilities, 3)}")
        print(f"Predicted state cluster: {state_cluster}, confidence: {state_confidence:.3f}")
        
        # 7. 使用对应的时间聚类模型
        if state_cluster in tsc_model.dpgmms_time and tsc_model.dpgmms_time[state_cluster] is not None:
            # current time is the index
            current_time = len(data_buffer)
            time_probabilities = tsc_model.dpgmms_time[state_cluster].predict_proba([[current_time]])[0]
            time_cluster = np.argmax(time_probabilities)
            time_confidence = np.max(time_probabilities)

            if time_confidence < time_probability_threshold:
                print(f"Time confidence {time_confidence:.3f} is less than the threshold {time_probability_threshold:.3f}, return None")
                return None
            
            print(f"Time probabilities: {np.round(time_probabilities, 3)}")
            print(f"Predicted time cluster: {time_cluster}, confidence: {time_confidence:.3f}")
        else:
            time_cluster = 0
            time_confidence = 1.0
            print(f"State cluster {state_cluster} has no time cluster model, error!")
            return None
        
        # 8. 构造最终的簇名称
        predicted_cluster = f"StateCluster_{state_cluster}-TimeCluster_{time_cluster}"
        combined_confidence = (state_confidence + time_confidence) / 2
        
        print(f"Predicted cluster: {predicted_cluster}")
        print(f"Combined confidence: {combined_confidence:.3f}")
        
        # 9. 检查簇是否存在
        if predicted_cluster in tsc_model.final_clusters:
            cluster_info = tsc_model.final_clusters[predicted_cluster]
            print(f"Cluster information:")
            print(f"  State center: {cluster_info['state_center']}")
            print(f"  Time center: {cluster_info['time_center']}")
            print(f"  Count: {cluster_info['count']}")
            print(f"  Demos: {cluster_info['demos']}")
            
            return {
                'cluster': predicted_cluster,
                'state_cluster': state_cluster,
                'time_cluster': time_cluster,
                'state_confidence': state_confidence,
                'time_confidence': time_confidence,
                'combined_confidence': combined_confidence,
                'cluster_info': cluster_info,
                'data_index': len(data_buffer) - 1  # 当前数据在缓冲区中的索引
            }
        else:
            print(f"predicted cluster {predicted_cluster} is not in the final cluster list")
            return None
    else:
        print(f"non-transition state")
        return None

def get_data_buffer():
    """获取当前数据缓冲区"""
    return data_buffer.copy()

def clear_data_buffer():
    """清空数据缓冲区"""
    global data_buffer
    data_buffer.clear()
    print("数据缓冲区已清空")

def get_buffer_statistics():
    """获取缓冲区统计信息"""
    if len(data_buffer) == 0:
        return {"size": 0, "mean": None, "std": None}
    
    data_array = np.array(data_buffer)
    return {
        "size": len(data_buffer),
        "mean": np.mean(data_array, axis=0),
        "std": np.std(data_array, axis=0),
        "shape": data_array.shape
    }
    
def print_tsc_results(potential_state_list):
    """打印TSC结果"""
    if not potential_state_list:
        print("\n\nNo potential states to print")
        print(f"length of potential_state_list: {len(potential_state_list)}")
        return
    print("--------------------------------------------------------")
    print(f"Total potential states: {len(potential_state_list)}")
    for potential_state in potential_state_list:
        print("--------------------------------")
        print(f"Time: {potential_state['data_index']}")
        print(f"Cluster: {potential_state['cluster']}")
        print(f"State cluster: {potential_state['state_cluster']}")
        print(f"Time cluster: {potential_state['time_cluster']}")
        print(f"State confidence: {potential_state['state_confidence']:.3f}")
        print(f"Time confidence: {potential_state['time_confidence']:.3f}")
        print(f"Combined confidence: {potential_state['combined_confidence']:.3f}")
        print("--------------------------------")

def load_model(cls, filepath):
        """从文件加载TSC模型"""
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
        
        # 创建TSC实例
        tsc = cls(
            p=model_data['p'],
            fp=model_data['fp'],
            delta=model_data['delta'],
            n_regimes=model_data['dpgmm_regime'].n_components,
            n_state_clusters=model_data['dpgmm_state'].n_components
        )
        
        # 恢复所有模型和数据
        tsc.dpgmm_regime = model_data['dpgmm_regime']
        tsc.dpgmm_state = model_data['dpgmm_state']
        tsc.dpgmms_time = model_data['dpgmms_time']
        tsc.remaining_states = model_data['remaining_states']
        tsc.remaining_times = model_data['remaining_times']
        tsc.remaining_indices = model_data['remaining_indices']
        tsc.remaining_regime_labels = model_data['remaining_regime_labels']
        tsc.final_clusters = model_data['final_clusters']
        
        print(f"TSC model has been loaded from {filepath} successfully")
        return tsc

if __name__ == '__main__':

    file_path = os.path.join(os.path.dirname(__file__), 'cluster', TSC_model_path)
    tsc_model = load_model(TSC, file_path)

    # check the dpgmms_time models parameters
    # for state_label in tsc_model.dpgmms_time:
    #     print(f"state_label: {state_label}'s DP-GMM模型信息:")
    #     print(f"  n_components_: {tsc_model.dpgmms_time[state_label].n_components}")
    #     print(f"  weights_: {tsc_model.dpgmms_time[state_label].weights_}")
    #     print(f"  means_ shape: {tsc_model.dpgmms_time[state_label].means_.shape}")
    


    subscriber = KinematicDataSubscriber()

    try:
        rospy.spin()
    except KeyboardInterrupt:
        print("subscriber terminated")

    print_tsc_results(potential_state_list)                 

