import numpy as np
from sklearn.mixture import BayesianGaussianMixture
from dtw import dtw
import warnings
import matplotlib.pyplot as plt
from matplotlib.widgets import Button, RadioButtons
import numpy as np
from load_data import load_demonstrations_state
from TSC import TSC
import rospy
import numpy as np
from std_msgs.msg import Float64MultiArray  
import os
import pickle
from config import state_probability_threshold, time_probability_threshold, TSC_model_path
from main import realtime_TSC, load_model
from load_data import cal_transition_time


tsc_model = None
data_buffer = []  # 存储所有接收到的数据
max_buffer_size = 1000  # 最大缓冲区大小
transition_threshold = 0.8  # 转移状态检测阈值
potential_state_list = []
regime_buffer = []

_T_MEANS = None
_T_STDS = None

def set_phase_stats(means, stds):
	"""设置时间节点统计并打印：第一行均值，第二行标准差"""
	import numpy as np
	global _T_MEANS, _T_STDS
	_T_MEANS = np.asarray(means, dtype=float).ravel()
	_T_STDS = np.asarray(stds, dtype=float).ravel()
	print(np.array2string(_T_MEANS, precision=6, separator=', '))
	print(np.array2string(_T_STDS, precision=6, separator=', '))

def visualize_phase():
    demos = load_demonstrations_state()
    demos = demos[138:]

    file_path = os.path.join(os.path.dirname(__file__), 'cluster', TSC_model_path)
    tsc_model = load_model(TSC, file_path)
    
    for demo in demos:
        for frame in demo:
            potential_state =realtime_TSC(frame)
            if potential_state is not None:
                potential_state_list.append(potential_state)
        # set global variables to zreo
        data_buffer = []
        regime_buffer = []
        potential_state_list = []  

def visualize_tsc_results(final_clusters=None):
    """可视化TSC聚类结果"""

    if final_clusters is None:
        file_path = os.path.join(os.path.dirname(__file__), 'cluster', TSC_model_path)
        tsc_model = load_model(TSC, file_path)

        """
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
        """
        final_clusters = tsc_model.final_clusters
    
    # 创建图形 - 增大尺寸
    fig, ax = plt.subplots(figsize=(14, 10))
    fig.suptitle('TSC cluster visualization', fontsize=16)
    
    # 创建控制面板 - 移到右侧图例下方
    ax_dim1 = plt.axes([0.78, 0.38, 0.2, 0.35])
    ax_dim2 = plt.axes([0.78, 0, 0.2, 0.35])
    
    # 维度选择按钮 - 包含时间维度
    dimension_names = [f"Dim_{i}" for i in range(14)] + ["Time"]
    radio_dim1 = RadioButtons(ax_dim1, dimension_names, active=0)
    radio_dim2 = RadioButtons(ax_dim2, dimension_names, active=1)
    
    # 添加标签
    ax_dim1.set_title('Dimension 1', fontsize=12, fontweight='bold')
    ax_dim2.set_title('Dimension 2', fontsize=12, fontweight='bold')
    
    # # 刷新按钮 - 移到右侧
    # ax_refresh = plt.axes([0.5, 0.00, 0.1, 0.05])
    # btn_refresh = Button(ax_refresh, 'Refresh')
    
    # 当前选择的维度
    selected_dims = [0, 1]
    
    def update_plot():
        """更新图形"""
        ax.clear()
        
        # 获取选中的维度
        dim1, dim2 = selected_dims
        
        # 为每个簇分配颜色
        colors = plt.cm.tab20(np.linspace(0, 1, len(final_clusters)))
        
        # 绘制每个簇
        for i, (cluster_name, cluster_data) in enumerate(final_clusters.items()):
            points = cluster_data['points']
            time_points = cluster_data['time_points']
            
            # 根据选择的维度提取数据
            if dim1 == 14:  # Time维度
                x_data = time_points.flatten()
            else:
                x_data = points[:, dim1]
            
            if dim2 == 14:  # Time维度
                y_data = time_points.flatten()
            else:
                y_data = points[:, dim2]
            
            # 绘制散点图
            ax.scatter(x_data, y_data, 
                      c=[colors[i]], 
                      label=cluster_name,
                      alpha=0.7,
                      s=50,
                      edgecolors='black',
                      linewidth=0.5)
            
            # 绘制簇中心
            if dim1 == 14:  # Time维度
                center_x = cluster_data['time_center']
            else:
                center_x = cluster_data['state_center'][dim1]
            
            if dim2 == 14:  # Time维度
                center_y = cluster_data['time_center']
            else:
                center_y = cluster_data['state_center'][dim2]
            
            ax.scatter(center_x, center_y, 
                      c=[colors[i]], 
                      marker='x', 
                      s=200, 
                      linewidth=3,
                      edgecolors='red')
        
        # 设置图形属性
        dim1_label = "Time" if dim1 == 14 else f"Dimension {dim1}"
        dim2_label = "Time" if dim2 == 14 else f"Dimension {dim2}"
        
        ax.set_xlabel(dim1_label, fontsize=12)
        ax.set_ylabel(dim2_label, fontsize=12)
        ax.set_title(f'TSC cluster visualization ({dim1_label} vs {dim2_label})', fontsize=14)
        ax.grid(True, alpha=0.3)
        ax.legend(bbox_to_anchor=(1.05, 1.05), loc='upper left')
        
        # 调整布局
        _annotate_time(ax, dim1 == 14, dim2 == 14)
        plt.tight_layout()
        plt.draw()
    
    def on_dim1_change(label):
        """维度1选择改变"""
        if label == "Time":
            selected_dims[0] = 14
        else:
            selected_dims[0] = int(label.split('_')[1])
        update_plot()
    
    def on_dim2_change(label):
        """维度2选择改变"""
        if label == "Time":
            selected_dims[1] = 14
        else:
            selected_dims[1] = int(label.split('_')[1])
        update_plot()
    
    def refresh_plot(event):
        """刷新图形"""
        update_plot()
    
    # 绑定事件
    radio_dim1.on_clicked(on_dim1_change)
    radio_dim2.on_clicked(on_dim2_change)
    # btn_refresh.on_clicked(refresh_plot)
    
    # 绘制初始图形
    update_plot()
    
    # 显示图形
    plt.show()

def _annotate_time(ax, time_on_x, time_on_y):
	if _T_MEANS is None or _T_STDS is None: return
	means, stds = _T_MEANS, _T_STDS
	xmin, xmax = ax.get_xlim(); ymin, ymax = ax.get_ylim()
	if time_on_x:
		for i, (m, s) in enumerate(zip(means, stds), 1):
			ax.axvline(m, color='gray', ls='--', lw=1.2, alpha=0.85)
			ax.axvspan(m - s, m + s, color='gray', alpha=0.15)
			ax.text(m, ymax, f'T{i}', color='gray', fontsize=9, ha='center', va='top')
	if time_on_y:
		for i, (m, s) in enumerate(zip(means, stds), 1):
			ax.axhline(m, color='gray', ls='--', lw=1.2, alpha=0.85)
			ax.axhspan(m - s, m + s, color='gray', alpha=0.15)
			ax.text(xmax, m, f'T{i}', color='gray', fontsize=9, ha='left', va='center')

if __name__ == '__main__':
    average_transition_time, std_transition_time = cal_transition_time()
    set_phase_stats(average_transition_time, std_transition_time)
    visualize_tsc_results()
         

    