import message_filters
import rospy
import random
import numpy as np
from cv_bridge import CvBridge
import sensor_msgs.point_cloud2 as pc2
from ambf_msgs.msg import RigidBodyState
from sensor_msgs.msg import Image
from sensor_msgs.msg import PointCloud2
import sensor_msgs.point_cloud2 as pc2
from sensor_msgs.msg import CompressedImage
from ambf_msgs.msg import RigidBodyState
from ambf_msgs.msg import ActuatorState
from ambf_msgs.msg import CameraState
from geometry_msgs.msg import PoseStamped
from geometry_msgs.msg import Twist
from geomagic_control.msg import DeviceButtonEvent
from os.path import join
import atexit
import time
import sys
import os
import struct
import math
from collections import namedtuple
import ctypes
import roslib.message
from sensor_msgs.msg import PointField
from ambf_client import Client
from numpy.linalg import inv
import functools
from scipy.spatial.transform import Rotation as R
from std_msgs.msg import Float32MultiArray
from collections import deque
import ipa as ipa_t
from std_msgs.msg import Float32  
import datetime
import threading
import time
from pynput import keyboard
import sys
import platform
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
from gracefulness import get_latest_data_dir

import params.config as config




# set resolution
sidesize = 22
resolution_x = 640
resolution_y = 360

stereo_l_img = []
stereo_r_img = []
segment_l_img = []
segment_r_img = []
psm_ghost_pose = []
Lpsm_pose_list = []
Rpsm_pose_list = []
Lpsm_velocity_list = []
Rpsm_velocity_list = []
gazepoint_list = []
gaze_list = []

GP_distance_list = []
psms_distance_list = []
ipaL_data_list = []
ipaR_data_list = []
pupilL_list = []
pupilR_list = []
scale_list = []

velocity_deque_L = deque(maxlen=8)
velocity_deque_R = deque(maxlen=8)

pupilL = deque(maxlen=128)
pupilR = deque(maxlen=128)
gazePoint = np.array([320,180])
scale = deque(maxlen=10)
velocity_deque = deque(maxlen=8)
psmPosePre = np.zeros(4)
psmPosePreL = np.zeros(4)
psmPosePreR = np.zeros(4)
flag = False
ipaL_data = 2
ipaR_data = 2
last_button_state = 0
last_button_stateL = 0
last_button_stateR = 0


clutch_times_list = [0, 0]
total_distance_list = [0]
total_time_list = [0]
test_scale = True

_DATATYPES = {}
_DATATYPES[PointField.INT8]	= ('b', 1)
_DATATYPES[PointField.UINT8]   = ('B', 1)
_DATATYPES[PointField.INT16]   = ('h', 2)
_DATATYPES[PointField.UINT16]  = ('H', 2)
_DATATYPES[PointField.INT32]   = ('i', 4)
_DATATYPES[PointField.UINT32]  = ('I', 4)
_DATATYPES[PointField.FLOAT32] = ('f', 4)
_DATATYPES[PointField.FLOAT64] = ('d', 8)


start_time = time.time()


latest_gaze_timestamp = 0.0
latest_gaze_point = [0.5, 0.5] 

"""
one-handed mode params:
tau_d tau_p tau_v  **A_d A_p A_v**  Y_base W_d W_p W_v W_dp W_dv W_pv W_dpv 

two-handed mode params:
tau_d tau_p tau_v tau_s  **A_d A_p A_v A_s**  Y_base W_d W_p W_v W_dps W_dvs W_pvs W_dpv W_dpvs
"""
init_params = {

	'd_min': config.feature_bound['d_min'] , 'd_max': config.feature_bound['d_max'],
	'p_min': config.feature_bound['p_min'] , 'p_max': config.feature_bound['p_max'], 
	'v_min': config.feature_bound['v_min'] , 'v_max': config.feature_bound['v_max'],
	's_min': config.feature_bound['s_min'] , 's_max': config.feature_bound['s_max'],

	'tau_d': 0.7,  
	'tau_p': 0.8,  
	'tau_v': 0.6,  
	'tau_s': 0.5,  

	'A_d': 1.5,
	'A_p': 4.0,
	'A_v': 8.0,
	'A_s': 2.0,

	'Y_base': 0.1,  
	'W_d': 1.0,	 
	'W_p': 1.0,	  
	'W_v': 1.0,	 
	'W_s': 1.0,
	'W_dp': 0.5,	
	'W_dv': 0.5,	 
	'W_pv': 0.5,	
	'W_dpv': 0.5,	
	'W_dps': 0.5,	
	'W_dvs': 0.5,	
	'W_pvs': 0.5,	
	'W_dpvs': 0.5, 

	'C_offset': config.feature_bound['C_offset'],

	'fixed_scale': 1.0,  
	'AFflag': 0 # 0-adaptive, 1-fixed
}

class AttentionHeatmapGenerator:
	def __init__(self, screen_width=640, screen_height=360, heatmap_size=(64, 36)):
		"""
		Initialize Attention Heatmap Generator

		Parameters:
			screen_width: screen width
			screen_height: screen height  
			heatmap_size: heatmap size (reduced resolution for performance)
		"""
		self.screen_width = screen_width
		self.screen_height = screen_height
		self.heatmap_size = heatmap_size

		# Initialize heatmap and historical data
		self.realtime_heatmap = np.zeros(heatmap_size)  # 实时更新的热图
		self.gaze_history = []  # 存储所有历史注视点 (timestamp, x, y)
		self.filtered_gaze_history = []  # 存储过滤后的注视点

		# For visualization
		self.all_realtime_heatmaps = []
		self.all_scale_factors = []
		self.filtered_indices = []
		self.outlier_indices = []
		self.all_timestamps = []
		self.all_gaze_points = []
		self.all_hand_positions = []
        # 添加三维坐标存储
		self.all_gaze_points_3d = []  # 存储三维注视点坐标 [(x, y, z), ...]
		self.all_hand_positions_3d = []  # 存储三维手部坐标 [[left_x, left_y, left_z], [right_x, right_y, right_z]]

		# 添加加权距离历史
		self.weighted_distances_left = []  # 左手加权距离历史
		self.weighted_distances_right = []  # 右手加权距离历史		

		# Scaling factor parameters
		self.scale_params = {
			'd_min': 0,      # Minimum distance
			'd_max': 500,    # Maximum distance
			'min_scale': 0.1,  # Minimum scaling factor
			'max_scale': 4.0,  # Maximum scaling factor
			'tau_d': 0.7,    # Distance weight parameter
			'Y_base': 1.0    # Base scaling factor
		}

		# Outlier filtering parameters - more focused on significant outliers
		self.filter_params = {
			'attention_threshold': 0.4,     # 注意力值
			'window_seconds': 3.0,          # 时间窗口
			'jump_threshold': 0.1,          # 跳跃检测阈值 (屏幕比例)
			'min_neighbors': 3,             # 有效点周围需要的最小有效邻居数
			'velocity_threshold': 0.06,      # 速度阈值，用于检测异常移动
		}

		# Fixed window configuration 
		self.fixed_window_config = {
			'enabled': True,           # 是否启用固定窗口
			'activation_offset': 2.5   # 在结束前多少秒启用固定窗口
		}

		# 初始化上一帧的缩放因子
		self.prev_scale_factor = [1.0, 1.0]

		# Create save directory
		
		self.save_dir = None
		

		# Set matplotlib to use English and specify font
		plt.rcParams['font.family'] = 'DejaVu Sans'
		plt.rcParams['axes.unicode_minus'] = False

	def set_save_dir(self, dir_name):
		if dir_name is None:
			current_file_path = os.path.abspath(__file__)
			current_dir = os.path.dirname(current_file_path)   
			data_base_dir = os.path.join(current_dir, 'data')
			latest_dir = get_latest_data_dir(data_base_dir)
			
			self.save_dir = latest_dir
		else:
			self.save_dir = dir_name
		print(f"gaze heatmap results will be saved to: {os.path.abspath(self.save_dir)}")

	def update_realtime_heatmap(self, current_time, total_duration=None, use_filtered_points=False):
		"""
		更新实时热图
		
		Parameters:
			current_time: 当前时间
			total_duration: 总采集时间（如果已知）
			use_filtered_points: 是否使用过滤后的点
		"""
		# 清空热图
		self.realtime_heatmap = np.zeros(self.heatmap_size)
		
		window_mode = "moving window"
		
		# 计算窗口开始时间：max(0, current_time - window_seconds)
		window_start = max(0, current_time - self.filter_params['window_seconds'])
		window_end = current_time
		
		# 如果提供了总采集时间且启用了固定窗口，检查是否进入固定窗口阶段
		if (total_duration is not None and 
			self.fixed_window_config['enabled']):
			
			# 计算固定窗口的起始时间
			fixed_window_start_time = total_duration - self.fixed_window_config['activation_offset']
			
			if current_time >= fixed_window_start_time:
				# 使用固定窗口：叠加最后5秒数据 (total_duration-5 ~ total_duration)
				window_start = total_duration - self.filter_params['window_seconds']
				window_end = total_duration
				window_mode = "fixed window"
		
		# 收集窗口内的所有注视点
		window_points = []
		
		if use_filtered_points:
			# 使用过滤后的点
			for timestamp, gx, gy in self.filtered_gaze_history:
				if window_start <= timestamp <= window_end:
					# Convert to heatmap coordinates
					x_heatmap = int(gx * (self.heatmap_size[1] - 1))
					y_heatmap = int(gy * (self.heatmap_size[0] - 1))
					
					# Ensure coordinates are valid
					x_heatmap = max(0, min(x_heatmap, self.heatmap_size[1] - 1))
					y_heatmap = max(0, min(y_heatmap, self.heatmap_size[0] - 1))
					
					window_points.append((x_heatmap, y_heatmap))
		else:
			# 使用所有点
			for timestamp, gx, gy in self.gaze_history:
				if window_start <= timestamp <= window_end:
					# Convert to heatmap coordinates
					x_heatmap = int(gx * (self.heatmap_size[1] - 1))
					y_heatmap = int(gy * (self.heatmap_size[0] - 1))
					
					# Ensure coordinates are valid
					x_heatmap = max(0, min(x_heatmap, self.heatmap_size[1] - 1))
					y_heatmap = max(0, min(y_heatmap, self.heatmap_size[0] - 1))
					
					window_points.append((x_heatmap, y_heatmap))
		
		if not window_points:
			return
		
		# to do 
		sigma = 2.5
		for x, y in window_points:
			# Create Gaussian kernel
			size = int(3 * sigma) + 1
			x_grid, y_grid = np.meshgrid(np.arange(-size, size+1), np.arange(-size, size+1))
			gaussian = np.exp(-(x_grid**2 + y_grid**2) / (2 * sigma**2))
			
			# Apply to heatmap
			x_start = max(0, x - size)
			x_end = min(self.heatmap_size[1], x + size + 1)
			y_start = max(0, y - size)
			y_end = min(self.heatmap_size[0], y + size + 1)
			
			gauss_x_start = max(0, size - (x - x_start))
			gauss_x_end = min(2*size+1, size + (x_end - x))
			gauss_y_start = max(0, size - (y - y_start))
			gauss_y_end = min(2*size+1, size + (y_end - y))
			
			self.realtime_heatmap[y_start:y_end, x_start:x_end] += gaussian[gauss_y_start:gauss_y_end, gauss_x_start:gauss_x_end]
		
		# Normalize
		max_val = np.max(self.realtime_heatmap)
		if max_val > 0:
			self.realtime_heatmap = self.realtime_heatmap / max_val
		
		return 

	def detect_jump(self, current_point, prev_point):
		"""
		检测明显的跳跃（突变点）
		"""
		if prev_point is None:
			return False
			
		dx = abs(current_point[0] - prev_point[0])
		dy = abs(current_point[1] - prev_point[1])
		distance = np.sqrt(dx**2 + dy**2)
		
		return distance > self.filter_params['jump_threshold']
	
	def detect_velocity_outlier(self, current_point, prev_point, time_diff):
		"""
		基于速度检测异常点
		"""
		if prev_point is None or time_diff <= 0:
			return False
			
		dx = abs(current_point[0] - prev_point[0])
		dy = abs(current_point[1] - prev_point[1])
		velocity = np.sqrt(dx**2 + dy**2) / time_diff
		
		return velocity > self.filter_params['velocity_threshold']	    
	
	def has_sufficient_neighbors(self, point, recent_points):
		"""
		检查一个点是否有足够多的邻居（在热图上有足够的支持）
		"""
		if not recent_points:
			return True
			
		# 将点转换到热图坐标
		x_heatmap = int(point[0] * (self.heatmap_size[1] - 1))
		y_heatmap = int(point[1] * (self.heatmap_size[0] - 1))
		
		# 检查热图上该点周围区域的值
		neighborhood_size = 3
		x_start = max(0, x_heatmap - neighborhood_size)
		x_end = min(self.heatmap_size[1], x_heatmap + neighborhood_size + 1)
		y_start = max(0, y_heatmap - neighborhood_size)
		y_end = min(self.heatmap_size[0], y_heatmap + neighborhood_size + 1)
		
		neighborhood = self.realtime_heatmap[y_start:y_end, x_start:x_end]
		
		# 计算高注意力值的邻居数量
		high_attention_count = np.sum(neighborhood > self.filter_params['attention_threshold'] * 2)
		
		return high_attention_count >= self.filter_params['min_neighbors']
	
	def filter_outliers_focused(self, current_idx, gaze_points, timestamps):
		"""
		专注于过滤明显异常点的函数 - 适配实时模式
		"""
		if current_idx < 0 or current_idx >= len(gaze_points):
			return False, 0.0
			
		current_point = gaze_points[current_idx]
		
		# 1. 基于热图的过滤 - 只过滤非常低注意力值的点
		x_heatmap = int(current_point[0] * (self.heatmap_size[1] - 1))
		y_heatmap = int(current_point[1] * (self.heatmap_size[0] - 1))
		x_heatmap = max(0, min(x_heatmap, self.heatmap_size[1] - 1))
		y_heatmap = max(0, min(y_heatmap, self.heatmap_size[0] - 1))
		
		attention_value = self.realtime_heatmap[y_heatmap, x_heatmap]
		heatmap_valid = attention_value >= self.filter_params['attention_threshold']
		
		# 2. 检测明显的跳跃
		jump_detected = False
		if current_idx > 0:
			prev_point = gaze_points[current_idx - 1]
			jump_detected = self.detect_jump(current_point, prev_point)
		
		# 3. 检查是否有足够的邻居支持
		has_neighbors = True
		if current_idx > 10:  # 只有在有足够历史数据时才检查
			recent_points = gaze_points[max(0, current_idx-10):current_idx]
			has_neighbors = self.has_sufficient_neighbors(current_point, recent_points)

		# 4. 基于速度的异常检测
		velocity_outlier = False
		if current_idx > 0:
			try:
				prev_point = gaze_points[current_idx - 1]
				prev_time = timestamps[current_idx - 1]
				time_diff = timestamps[current_idx] - prev_time
				velocity_outlier = self.detect_velocity_outlier(current_point, prev_point, time_diff)
			except (IndexError, TypeError):
				velocity_outlier = False            
		
		# 综合条件：当点在低注意力区域，或缺乏邻居支持，或速度异常，或是明显的跳跃时，被过滤
		is_outlier = not heatmap_valid or not has_neighbors or velocity_outlier or jump_detected
		
		return not is_outlier, attention_value
	
	# def calculate_weighted_distance(self, hand_pos):
	# 	"""
	# 	Calculate weighted distance between attention heatmap and hand position
	# 	"""
	# 	if np.sum(self.realtime_heatmap) < 1e-6:
	# 		return (self.scale_params['d_max'] + self.scale_params['d_min']) / 2
		
	# 	y_indices, x_indices = np.indices(self.heatmap_size)
	# 	screen_x = x_indices * self.screen_width / self.heatmap_size[1]
	# 	screen_y = y_indices * self.screen_height / self.heatmap_size[0]
		
	# 	hand_screen_x = hand_pos[0] * self.screen_width
	# 	hand_screen_y = hand_pos[1] * self.screen_height
		
		
	# 	distances = np.sqrt((screen_x - hand_screen_x)**2 + (screen_y - hand_screen_y)**2)
	# 	normalized_heatmap = self.realtime_heatmap / (np.sum(self.realtime_heatmap) + 1e-10)
	# 	weighted_distance = np.sum(normalized_heatmap * distances)
		
	# 	return weighted_distance
	def calculate_weighted_distance(self, gaze_position_3d, hand_position_3d):
		"""
		计算单个注视点与单个手部位置的三维距离
		gaze_position_3d: 单个注视点的三维坐标 [x, y, z]
		hand_position_3d: 单个手部的三维坐标 [x, y, z]
		"""
		try:
			# 安全检查
			if gaze_position_3d is None or hand_position_3d is None:
				return (self.scale_params['d_max'] + self.scale_params['d_min']) / 2
			
			# 调试输出
			# print(f"gaze_position_3d: {gaze_position_3d}, type: {type(gaze_position_3d)}")
			# print(f"hand_position_3d: {hand_position_3d}, type: {type(hand_position_3d)}")
			
			# 确保输入是有效的三维坐标
			if (not hasattr(gaze_position_3d, '__len__') or len(gaze_position_3d) < 3 or
				not hasattr(hand_position_3d, '__len__') or len(hand_position_3d) < 3):
				return (self.scale_params['d_max'] + self.scale_params['d_min']) / 2
			
			# 直接计算三维欧几里得距离
			distance = np.sqrt(
				(float(gaze_position_3d[0]) - float(hand_position_3d[0]))**2 +
				(float(gaze_position_3d[1]) - float(hand_position_3d[1]))**2 +
				(float(gaze_position_3d[2]) - float(hand_position_3d[2]))**2
			)
			
			return distance
			
		except (TypeError, IndexError, ValueError) as e:
			print(f"Error calculating 3D distance: {e}")
			return (self.scale_params['d_max'] + self.scale_params['d_min']) / 2

	def calculate_temporal_weights(self, n_points):
		"""
		计算时间衰减权重（指数衰减）
		"""
		if n_points == 0:
			return []
		
		# 指数衰减：最近的注视点权重最高
		decay_rate = 0.8
		weights = [decay_rate ** i for i in range(n_points-1, -1, -1)]
		
		# 归一化
		total_weight = sum(weights)
		if total_weight > 0:
			weights = [w / total_weight for w in weights]
		
		return weights
	
	def visualize_results(self, timestamps, gaze_points, hand_positions, 
						all_realtime_heatmaps, all_scale_factors, 
						filtered_indices, outlier_indices):
		"""
		Visualize results with filtering comparison
		"""
		timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
		
		# 创建2x2的子图布局
		fig, axes = plt.subplots(2, 2, figsize=(15, 12))
		
		# 1. 全时间热图
		# 创建全时间的热图，叠加所有注视点
		full_time_heatmap = np.zeros(self.heatmap_size)
		
		# 叠加所有过滤后的注视点
		for i in filtered_indices:
			if i < len(gaze_points):
				gaze_point = gaze_points[i]
				x_heatmap = int(gaze_point[0] * (self.heatmap_size[1] - 1))
				y_heatmap = int(gaze_point[1] * (self.heatmap_size[0] - 1))
				x_heatmap = max(0, min(x_heatmap, self.heatmap_size[1] - 1))
				y_heatmap = max(0, min(y_heatmap, self.heatmap_size[0] - 1))
				
				# 为每个点添加高斯分布
				sigma = 2.5
				size = int(3 * sigma) + 1
				x_grid, y_grid = np.meshgrid(np.arange(-size, size+1), np.arange(-size, size+1))
				gaussian = np.exp(-(x_grid**2 + y_grid**2) / (2 * sigma**2))
				
				# Apply to heatmap
				x_start = max(0, x_heatmap - size)
				x_end = min(self.heatmap_size[1], x_heatmap + size + 1)
				y_start = max(0, y_heatmap - size)
				y_end = min(self.heatmap_size[0], y_heatmap + size + 1)
				
				gauss_x_start = max(0, size - (x_heatmap - x_start))
				gauss_x_end = min(2*size+1, size + (x_end - x_heatmap))
				gauss_y_start = max(0, size - (y_heatmap - y_start))
				gauss_y_end = min(2*size+1, size + (y_end - y_heatmap))
				
				full_time_heatmap[y_start:y_end, x_start:x_end] += gaussian[gauss_y_start:gauss_y_end, gauss_x_start:gauss_x_end]
		
		# Normalize full time heatmap
		max_val = np.max(full_time_heatmap)
		if max_val > 0:
			full_time_heatmap = full_time_heatmap / max_val
		
		# 设置640x360的显示比例
		target_width = 640
		target_height = 360
		aspect_ratio = target_width / target_height
		
		# 显示全时间热图
		im = axes[0, 0].imshow(full_time_heatmap, cmap='hot', interpolation='nearest',
							extent=[0, target_width, target_height, 0], 
							aspect=aspect_ratio, origin='upper')
		
		# Plot all gaze points (转换为目标分辨率)
		gaze_x = [p[0] * target_width for p in gaze_points]
		gaze_y = [p[1] * target_height for p in gaze_points]
		
		# Plot filtered points (green) and outliers (red)
		filtered_x = [gaze_x[i] for i in filtered_indices if i < len(gaze_x)]
		filtered_y = [gaze_y[i] for i in filtered_indices if i < len(gaze_y)]
		outlier_x = [gaze_x[i] for i in outlier_indices if i < len(gaze_x)]
		outlier_y = [gaze_y[i] for i in outlier_indices if i < len(gaze_y)]
		
		# # Plot hand positions (转换为目标分辨率)
		# hand_x = []
		# hand_y = []
		# for p in hand_positions:
		# 	if len(p) > 0:
		# 		if isinstance(p, (list, np.ndarray)) and len(p) >= 2:
		# 			hand_x.append(p[0] * target_width)
		# 			hand_y.append(p[1] * target_height)
		
		axes[0, 0].scatter(gaze_x, gaze_y, c='blue', marker='o', s=20, alpha=0.3, label='All Gaze')
		if filtered_x and filtered_y:
			axes[0, 0].scatter(filtered_x, filtered_y, c='green', marker='o', s=40, label='Filtered Gaze')
		if outlier_x and outlier_y:
			axes[0, 0].scatter(outlier_x, outlier_y, c='red', marker='x', s=60, label='Outliers')
		#axes[0, 0].scatter(hand_x, hand_y, c='purple', marker='s', s=80, label='Hand Position')
		axes[0, 0].set_title('Full Time Heatmap (All Filtered Gaze Points) - 640x360')
		axes[0, 0].set_xlim(0, target_width)
		axes[0, 0].set_ylim(target_height, 0)  # 保持原点在左上角
		axes[0, 0].legend()
		plt.colorbar(im, ax=axes[0, 0])

		# 2. Weighted distance over time
		if (timestamps and 
			hasattr(self, 'all_gaze_points_3d') and 
			hasattr(self, 'all_hand_positions_3d') and
			len(self.all_gaze_points_3d) > 0 and 
			len(self.all_hand_positions_3d) > 0):
			
			times = np.array(timestamps) - timestamps[0]
			
			# 确保数据长度一致
			min_length = min(len(times), len(self.all_gaze_points_3d), 
							len(self.all_hand_positions_3d),
							len(self.weighted_distances_left),
							len(self.weighted_distances_right))
			
			if min_length > 0:
				times_trimmed = times[:min_length]
				left_distances = self.weighted_distances_left[:min_length]
				right_distances = self.weighted_distances_right[:min_length]
				
				# 绘制左右手的三维距离
				axes[0, 1].plot(times_trimmed, left_distances, 'b-', linewidth=2, label='Left Hand 3D Distance')
				axes[0, 1].plot(times_trimmed, right_distances, 'r-', linewidth=2, label='Right Hand 3D Distance')
				
				axes[0, 1].set_title('3D Weighted Distance Between Hands and Gaze Point')
				axes[0, 1].set_xlabel('Time (s)')
				axes[0, 1].set_ylabel('3D Distance (meters)')
				axes[0, 1].grid(True)
				axes[0, 1].legend()
				
				# 添加统计信息
				avg_left = np.mean(left_distances)
				avg_right = np.mean(right_distances)
				axes[0, 1].axhline(y=avg_left, color='b', linestyle='--', alpha=0.7, 
									label=f'Left Avg: {avg_left:.3f}m')
				axes[0, 1].axhline(y=avg_right, color='r', linestyle='--', alpha=0.7,
									label=f'Right Avg: {avg_right:.3f}m')
				
			else:
				axes[0, 1].text(0.5, 0.5, 'No 3D distance data', 
							ha='center', va='center', transform=axes[0, 1].transAxes)
				axes[0, 1].set_title('3D Weighted Distance Over Time')
		else:
			axes[0, 1].text(0.5, 0.5, 'No 3D coordinate data available', 
						ha='center', va='center', transform=axes[0, 1].transAxes)
			axes[0, 1].set_title('3D Weighted Distance Over Time')

		# # 2. Scale factor over time 
		# if timestamps and all_scale_factors:
		# 	min_length = min(len(timestamps), len(all_scale_factors))
		# 	times = np.array(timestamps[:min_length]) - timestamps[0]
		# 	scale_factors = all_scale_factors[:min_length]
			
		# 	axes[0, 1].plot(times, scale_factors, 'r-', linewidth=2)
		# 	axes[0, 1].set_title('Scale Factor Over Time')
		# 	axes[0, 1].set_xlabel('Time (s)')
		# 	axes[0, 1].set_ylabel('Scale Factor')
		# 	axes[0, 1].grid(True)
		# else:
		# 	axes[0, 1].text(0.5, 0.5, 'No scale factor data', 
		# 				ha='center', va='center', transform=axes[0, 1].transAxes)
		# 	axes[0, 1].set_title('Scale Factor Over Time')
	
		# 3. Original gaze coordinates over time 
		if timestamps and gaze_x and gaze_y:
			times = np.array(timestamps) - timestamps[0]
			min_length = min(len(times), len(gaze_x), len(gaze_y))
			times_trimmed = times[:min_length]
			gaze_x_trimmed = gaze_x[:min_length]
			gaze_y_trimmed = gaze_y[:min_length]
			
			axes[1, 0].plot(times_trimmed, gaze_x_trimmed, 'b-', label='X Coordinate', alpha=0.7)
			axes[1, 0].plot(times_trimmed, gaze_y_trimmed, 'g-', label='Y Coordinate', alpha=0.7)
			axes[1, 0].set_title('Original Gaze Coordinates Over Time')
			axes[1, 0].set_xlabel('Time (s)')
			axes[1, 0].set_ylabel('Screen Coordinates (pixels)')
			axes[1, 0].legend()
			axes[1, 0].grid(True)
		else:
			axes[1, 0].text(0.5, 0.5, 'No gaze coordinate data', 
						ha='center', va='center', transform=axes[1, 0].transAxes)
			axes[1, 0].set_title('Original Gaze Coordinates Over Time')
		
		# 4. Filtered gaze coordinates over time 
		if filtered_indices and timestamps and gaze_x and gaze_y:
			times = np.array(timestamps) - timestamps[0]
			filtered_times = [times[i] for i in filtered_indices if i < len(times)]
			filtered_x_coords = [gaze_x[i] for i in filtered_indices if i < len(gaze_x)]
			filtered_y_coords = [gaze_y[i] for i in filtered_indices if i < len(gaze_y)]
			
			min_length = min(len(filtered_times), len(filtered_x_coords), len(filtered_y_coords))
			if min_length > 0:
				filtered_times = filtered_times[:min_length]
				filtered_x_coords = filtered_x_coords[:min_length]
				filtered_y_coords = filtered_y_coords[:min_length]
				
				axes[1, 1].plot(filtered_times, filtered_x_coords, 'b-', label='X Coordinate', alpha=0.7)
				axes[1, 1].plot(filtered_times, filtered_y_coords, 'g-', label='Y Coordinate', alpha=0.7)
				axes[1, 1].set_title('Filtered Gaze Coordinates Over Time')
				axes[1, 1].set_xlabel('Time (s)')
				axes[1, 1].set_ylabel('Screen Coordinates (pixels)')
				axes[1, 1].legend()
				axes[1, 1].grid(True)
			else:
				axes[1, 1].text(0.5, 0.5, 'No filtered gaze data', 
							ha='center', va='center', transform=axes[1, 1].transAxes)
				axes[1, 1].set_title('Filtered Gaze Coordinates Over Time')
		else:
			axes[1, 1].text(0.5, 0.5, 'No filtered gaze data', 
						ha='center', va='center', transform=axes[1, 1].transAxes)
			axes[1, 1].set_title('Filtered Gaze Coordinates Over Time')
		
		plt.tight_layout()
		plt.savefig(os.path.join(self.save_dir, f'gaze_filter_results.png'), 
				dpi=300, bbox_inches='tight')
		
		# 检查是否在有 GUI 的环境中运行
		# import matplotlib
		# if matplotlib.get_backend().lower() not in ['agg', 'svg', 'pdf', 'ps']:
		#     plt.show()
		# else:
		#     print("图表已保存，但在非 GUI 环境下无法显示")
		plt.close()
	
	def save_visualization_data(self):
		"""Save visualization data for later analysis"""
		data = {
			'timestamps': self.all_timestamps,
			'gaze_points': self.all_gaze_points,
			'hand_positions': self.all_hand_positions,
			'realtime_heatmaps': self.all_realtime_heatmaps,
			'scale_factors': self.all_scale_factors,
			'filtered_indices': self.filtered_indices,
			'outlier_indices': self.outlier_indices,
			'gaze_history': self.gaze_history,
			'filtered_gaze_history': self.filtered_gaze_history
		}
		np.save(os.path.join(self.save_dir, f'gaze_filter_data.npy'), data)

class DataCollector:
	def __init__(self):
		# Your original initialization code
		self.bridge = CvBridge()
		self.collecting = False
		self.collection_complete = False
		self.params = None

		# Initialize ROS
		rospy.init_node('main', anonymous=True)
		self.scale_pub = rospy.Publisher('/scale', Float32MultiArray, queue_size=10)
		rospy.Subscriber("/gaze_data", Float32MultiArray, gaze_data_cb)

		# acquire touch data

		LPSM_POSE = message_filters.Subscriber("/ambf/env/psm1/toolyawlink/State", RigidBodyState)
		RPSM_POSE = message_filters.Subscriber("/ambf/env/psm2/toolyawlink/State", RigidBodyState)
		#PSM_POSE = message_filters.Subscriber("/Geomagic/pose", PoseStamped)
		#LPSM_TWIST = message_filters.Subscriber("/Geomagic/twist", Twist)
		DEPTH = message_filters.Subscriber("/ambf/env/cameras/cameraR/DepthData", PointCloud2)
		FRAME = message_filters.Subscriber("/ambf/env/CameraFrame/State", RigidBodyState)
		CAMERA = message_filters.Subscriber("/ambf/env/cameras/cameraR/State", CameraState)
		#rospy.Subscriber("/gaze_data", Float32MultiArray, set_gaze_data)
		#CAMERA_DEPTHDATA = message_filters.Subscriber("/ambf/env/cameras/cameraL/DepthData", PointCSubscriber("/ambf/env/cameras/cameraL/DepthData", PointCloud2)loud2)
		rospy.Subscriber("/Geomagic_Left/button", DeviceButtonEvent, cal_Lpsm_clutch_times)
		rospy.Subscriber("/Geomagic_Right/button", DeviceButtonEvent, cal_Rpsm_clutch_times)
		# create approximate time synchronizer
		ts = message_filters.ApproximateTimeSynchronizer([LPSM_POSE,RPSM_POSE,DEPTH,CAMERA,FRAME], queue_size=20, slop=1.2, allow_headerless=True)
		#ts = message_filters.ApproximateTimeSynchronizer([DEPTH,CAMERA,FRAME], queue_size=10, slop=0.6, allow_headerless=True)
		ts.registerCallback(self.maincb)	

		# Initialize attention heatmap generator
		
		self.attention_heatmap_generator = AttentionHeatmapGenerator(screen_width=resolution_x, screen_height=resolution_y)
		# --- KEYBOARD LISTENER INTEGRATION START ---
		# In the class's __init__ function, start the keyboard listener
		self.start_keyboard_listener()
		# --- KEYBOARD LISTENER INTEGRATION END ---
		
	# --- KEYBOARD LISTENER INTEGRATION START ---
	# The following three functions are newly added to handle keyboard listening
	def reset_globals(self):
		"""Reset all global variables used for data collection"""
		global Lpsm_pose_list, Rpsm_pose_list, Lpsm_velocity_list, Rpsm_velocity_list, gazepoint_list, gaze_list
		global GP_distance_list, psms_distance_list, ipaL_data_list, ipaR_data_list
		global pupilL_list, pupilR_list, scale_list
		global velocity_deque_L, velocity_deque_R, pupilL, pupilR
		global gazePoint, scale, velocity_deque
		global psmPosePre, psmPosePreL, psmPosePreR, flag
		global ipaL_data, ipaR_data, last_button_state, last_button_stateL, last_button_stateR
		global clutch_times_list, total_distance_list, total_time_list, start_time
		global latest_gaze_timestamp, latest_gaze_point

		
		# Reset all lists
		Lpsm_pose_list = []
		Rpsm_pose_list = []
		Lpsm_velocity_list = []
		Rpsm_velocity_list = []
		gazepoint_list = []
		gaze_list = []
		GP_distance_list = []
		psms_distance_list = []
		ipaL_data_list = []
		ipaR_data_list = []
		pupilL_list = []
		pupilR_list = []
		scale_list = []
		
		# Reset deques
		velocity_deque_L = deque(maxlen=8)
		velocity_deque_R = deque(maxlen=8)
		pupilL = deque(maxlen=128)
		pupilR = deque(maxlen=128)
		scale = deque(maxlen=10)
		velocity_deque = deque(maxlen=8)
		
		# Reset state variables
		psmPosePre = np.zeros(4)
		psmPosePreL = np.zeros(4)
		psmPosePreR = np.zeros(4)
		flag = False
		ipaL_data = 2
		ipaR_data = 2
		last_button_state = 0
		last_button_stateL = 0
		last_button_stateR = 0
		
		# Reset statistical variables
		clutch_times_list = [0,0]
		total_distance_list = [0]
		total_time_list = [0]
		
		# Reset other variables
		gazePoint = np.array([320, 180])
		start_time = time.time()

		latest_gaze_timestamp = 0.0
		latest_gaze_point = [0.5, 0.5]

		# Reset attention heatmap
		
		self.attention_heatmap_generator.gaze_history.clear()
		self.attention_heatmap_generator.filtered_gaze_history.clear()
		self.attention_heatmap_generator.realtime_heatmap = np.zeros(self.attention_heatmap_generator.heatmap_size)
		self.attention_heatmap_generator.all_realtime_heatmaps.clear()
		self.attention_heatmap_generator.all_scale_factors.clear()
		self.attention_heatmap_generator.filtered_indices.clear()
		self.attention_heatmap_generator.outlier_indices.clear()
		self.attention_heatmap_generator.all_timestamps.clear()
		self.attention_heatmap_generator.all_gaze_points.clear()
		self.attention_heatmap_generator.all_hand_positions.clear()
		self.attention_heatmap_generator.prev_scale_factor=[1.0, 1.0]	
		
		print("global variables reset, ready to start new data collection...")

	def on_press(self, key):
		"""Callback function for key press"""
		# We set the 's' key (stop) to stop collection
		if hasattr(key, 'char') and key.char == 's':
			if self.collecting:
				print(f"\nstop collecting... (press 's' to stop)")
				self.stop_collection()
	
	def listener_thread_target(self):
		"""Target function for the listener thread"""
		# The 'with' statement ensures the listener is properly cleaned up on exit
		with keyboard.Listener(on_press=self.on_press) as listener:
			listener.join()

	def start_keyboard_listener(self):
		"""Create a background thread to run the keyboard listener"""
		listener_thread = threading.Thread(target=self.listener_thread_target, daemon=True)
		listener_thread.start()
	# --- KEYBOARD LISTENER INTEGRATION END ---

	def start_collection(self):
		"""Start data collection"""
		self.collecting = True
		self.collection_complete = False
		print("started collecting PSM 1&2 baselink, stereo image, and segmentation image ...")
		# --- MODIFICATION START ---
		# Modify the prompt message to inform the user of the new key
		print("data collecting... (press 's' to stop)")
		# --- MODIFICATION END ---
		
	def stop_collection(self):
		"""Stop data collection"""
		# Ensure messages are printed only when collection is in progress
		if self.collecting:
			print("collecting task stopped.")
			self.collecting = False
			self.collection_complete = True

			
	def wait_for_completion(self):
		"""
		Wait for collection to complete.
		--- MODIFICATION START ---
		This function is now very simple and no longer needs a try/except block.
		It just waits for the self.collecting flag to be changed by the keyboard listener thread.
		--- MODIFICATION END ---
		"""
		while self.collecting and not rospy.is_shutdown():
			rospy.sleep(0.1)

	def gaze_heatmap_visualization(self):
		"""Generate and save visualization results using the existing methods"""
		
		if self.attention_heatmap_generator.all_timestamps:  # 确保有数据可可视化
			
			# 准备可视化所需的数据
			timestamps = self.attention_heatmap_generator.all_timestamps
			gaze_points = self.attention_heatmap_generator.all_gaze_points
			hand_positions = self.attention_heatmap_generator.all_hand_positions
			all_realtime_heatmaps = self.attention_heatmap_generator.all_realtime_heatmaps
			all_scale_factors = self.attention_heatmap_generator.all_scale_factors
			filtered_indices = self.attention_heatmap_generator.filtered_indices
			outlier_indices = self.attention_heatmap_generator.outlier_indices
			
			
			self.attention_heatmap_generator.visualize_results(
				timestamps,
				gaze_points, 
				hand_positions,
				all_realtime_heatmaps,
				all_scale_factors,
				filtered_indices,
				outlier_indices
			)
			
			
			self.attention_heatmap_generator.save_visualization_data()
			
		else:
			print("no data to visualize")

	def maincb(self, Lpsm, Rpsm, camera_depthdata, cameraR, cameraFrame):
		if self.collecting:
			sys.stdout.write('\r-- Time past: %02.1f' % float(time.time() - start_time))
			sys.stdout.flush()

		Lpsm_position3 = self.get_position(Lpsm.pose)
		Rpsm_position3 = self.get_position(Rpsm.pose)
		Lpsm_velocity6 = self.get_velocity(Lpsm.twist)
		Rpsm_velocity6 = self.get_velocity(Rpsm.twist)
		Lpsm_timestamp = Lpsm.header.stamp.secs + Lpsm.header.stamp.nsecs*1e-9
		Rpsm_timestamp = Rpsm.header.stamp.secs + Rpsm.header.stamp.nsecs*1e-9

		gazepoint_position3 = get_gazepoint_position3(camera_depthdata, cameraR.pose, cameraFrame.pose, gazePoint[0], gazePoint[1])

		# 存储三维坐标到注意力热图生成器
		self.attention_heatmap_generator.all_gaze_points_3d.append(gazepoint_position3)
		self.attention_heatmap_generator.all_hand_positions_3d.append([Lpsm_position3, Rpsm_position3])

		# 计算三维加权距离
		weighted_dist_left_3d = self.attention_heatmap_generator.calculate_weighted_distance(
			gazepoint_position3, Lpsm_position3
		)
		weighted_dist_right_3d = self.attention_heatmap_generator.calculate_weighted_distance(
			gazepoint_position3, Rpsm_position3
		)
		
		self.attention_heatmap_generator.weighted_distances_left.append(weighted_dist_left_3d)
		self.attention_heatmap_generator.weighted_distances_right.append(weighted_dist_right_3d)		

		global latest_gaze_timestamp, latest_gaze_point
		
		current_time = latest_gaze_timestamp
		current_gaze_point = latest_gaze_point
		
		# 检查时间戳是否有效
		if current_time <= 0:
			current_time = time.time()
		
		# 添加到历史数据
		self.attention_heatmap_generator.gaze_history.append((current_time, current_gaze_point[0], current_gaze_point[1]))
		
		# 更新热图用于过滤
		self.attention_heatmap_generator.update_realtime_heatmap(current_time, use_filtered_points=False)
		
		# 存储到可视化数据
		self.attention_heatmap_generator.all_timestamps.append(current_time)
		self.attention_heatmap_generator.all_gaze_points.append(current_gaze_point)
		
		# 使用3D坐标作为手部位置数据
		normalized_Lpsm = self.normalize_3d_position(Lpsm_position3)
		normalized_Rpsm = self.normalize_3d_position(Rpsm_position3)
		self.attention_heatmap_generator.all_hand_positions.append([normalized_Lpsm, normalized_Rpsm])
		self.attention_heatmap_generator.all_realtime_heatmaps.append(self.attention_heatmap_generator.realtime_heatmap.copy())

		current_idx = len(self.attention_heatmap_generator.all_timestamps) - 1

		# 调用过滤方法
		try:
			is_valid, attention_value = self.attention_heatmap_generator.filter_outliers_focused(
				current_idx,
				self.attention_heatmap_generator.all_gaze_points,
				self.attention_heatmap_generator.all_timestamps
			)
		except Exception as e:
			is_valid = True

		if not is_valid:
			# 如果是异常点，从主要数据列表中移除
			self.attention_heatmap_generator.all_timestamps.pop()
			self.attention_heatmap_generator.all_gaze_points.pop()
			self.attention_heatmap_generator.all_hand_positions.pop()
			self.attention_heatmap_generator.all_realtime_heatmaps.pop()
			# 被过滤的点不参与后续计算
			#print(f"过滤掉异常点: {current_gaze_point}, 注意力值: {attention_value:.3f}")
		else:
			# 有效点：添加到过滤历史并更新过滤后热图
			self.attention_heatmap_generator.filtered_indices.append(current_idx)
			self.attention_heatmap_generator.filtered_gaze_history.append((current_time, current_gaze_point[0], current_gaze_point[1]))
			# 更新基于过滤后点的热图（用于缩放计算）
			self.attention_heatmap_generator.update_realtime_heatmap(current_time, use_filtered_points=True)

		Ldistance, Lpsm_velocity = cal_metrics(Lpsm_position3, Lpsm_velocity6, Lpsm_timestamp, gazepoint_position3)
		Rdistance, Rpsm_velocity = cal_metrics(Rpsm_position3, Rpsm_velocity6, Rpsm_timestamp, gazepoint_position3)
		distance_psms = np.sqrt((Lpsm_position3[0]-Rpsm_position3[0])**2 + (Lpsm_position3[1]-Rpsm_position3[1])**2 + (Lpsm_position3[2]-Rpsm_position3[2])**2)
		
		global ipaL_data, ipaR_data
		if(len(pupilL)>=128 and len(pupilR)>=128):
			_ipaL_data = ipa_t.ipa_cal(pupilL)
			_ipaR_data = ipa_t.ipa_cal(pupilR)
			if(_ipaL_data!=0):
				ipaL_data = _ipaL_data
			if(_ipaR_data!=0):
				ipaR_data = _ipaR_data
		else:
			ipaL_data = 1
			ipaR_data = 1

		psm_position = [Lpsm_position3, Rpsm_position3]
		distance = [Ldistance, Rdistance]
		psm_velocity = [Lpsm_velocity, Rpsm_velocity]

		scale = self.calculate_scale(gazepoint_position3, psm_position, psm_velocity, ipaL_data, ipaR_data, distance_psms, is_valid)	
		
		try:  
			self.scale_pub.publish(scale)
		except rospy.ROSException as e:
			rospy.logwarn(f"Failed to publish scale: {e}")

		scale_list.append(scale.data)
		self.attention_heatmap_generator.all_scale_factors.append(np.mean(scale.data))

		cal_total_distance(Lpsm_position3, Rpsm_position3)
		total_time_list[0] = float(time.time() - start_time)

		# # 可选：定期输出调试信息
		# if self.collecting and current_idx % 30 == 0:
		# 	valid_count = len(self.attention_heatmap_generator.filtered_indices)
		# 	total_count = current_idx + 1
		# 	valid_ratio = valid_count / total_count if total_count > 0 else 0
		# 	print(f"\n实时过滤统计: {valid_count}/{total_count} 有效点 ({valid_ratio:.1%})")
		# 	print(f"当前热图模式: {window_mode}")
		# 	print(f"当前点状态: {'有效' if is_valid else '过滤'}")
		# 	print(f"左右缩放因子: {scale.data[0]:.3f}, {scale.data[1]:.3f}")

		if self.collecting:
			print("\n")
			print("=" * 50)
			print(f"{'PSM Left 3d Position':<25}: [{psm_position[0][0]:.3f}, {psm_position[0][1]:.3f}, {psm_position[0][2]:.3f}]")
			print(f"{'PSM Right 3d Position':<25}: [{psm_position[1][0]:.3f}, {psm_position[1][1]:.3f}, {psm_position[1][2]:.3f}]")
			print(f"{'Gazing Point ':<25}: [{gazePoint[0]:.3f}, {gazePoint[1]:.3f}]")
			print(f"{'PSM Left Gazing Point Distance':<25}: {distance[0]:.3f}")
			print(f"{'PSM Right Gazing Point Distance':<25}: {distance[1]:.3f}")
			print(f"{'PSM Left Velocity (x, y, z)':<25}: [{psm_velocity[0]:.3f}]")
			print(f"{'PSM Right Velocity (x, y, z)':<25}: [{psm_velocity[1]:.3f}]")
			print(f"{'IPA Left Data':<25}: [{ipaL_data:.3f}]")
			print(f"{'IPA Right Data':<25}: [{ipaR_data:.3f}]")
			print(f"{'Scale Left':<25}: {scale.data[0]:.8f}")
			print(f"{'Scale Right':<25}: {scale.data[1]:.8f}")
			print(f"{'Point Status':<25}: {'Valid' if is_valid else 'Filtered'}")
			print("=" * 50)
			print("\n")

		GP_distance_list.append(distance)
		psms_distance_list.append(distance_psms)
		Lpsm_pose_list.append(np.append(Lpsm_position3, Lpsm_timestamp))
		Rpsm_pose_list.append(np.append(Rpsm_position3, Rpsm_timestamp))
		Lpsm_velocity_list.append(Lpsm_velocity6)
		Rpsm_velocity_list.append(Rpsm_velocity6)
		if(ipaL_data != 0):
			ipaL_data_list.append(ipaL_data)
		if(ipaR_data != 0):
			ipaR_data_list.append(ipaR_data)		

	# def maincb(self,Lpsm,Rpsm,camera_depthdata, cameraR, cameraFrame):
	# 	if self.collecting:
	# 		sys.stdout.write('\r-- Time past: %02.1f' % float(time.time() - start_time))
	# 		sys.stdout.flush()

	# 	Lpsm_position3 = self.get_position(Lpsm.pose)
	# 	Rpsm_position3 = self.get_position(Rpsm.pose)
	# 	Lpsm_velocity6 = self.get_velocity(Lpsm.twist) # linear3 + angular3
	# 	Rpsm_velocity6 = self.get_velocity(Rpsm.twist)
	# 	Lpsm_timestamp = Lpsm.header.stamp.secs + Lpsm.header.stamp.nsecs*1e-9
	# 	Rpsm_timestamp = Rpsm.header.stamp.secs + Rpsm.header.stamp.nsecs*1e-9

	# 	gazepoint_position3 = get_gazepoint_position3(camera_depthdata, cameraR.pose, cameraFrame.pose, gazePoint[0], gazePoint[1])
	
	# 	# Metrics calculation
	# 	Ldistance, Lpsm_velocity= cal_metrics(Lpsm_position3, Lpsm_velocity6, Lpsm_timestamp, gazepoint_position3)
	# 	Rdistance, Rpsm_velocity = cal_metrics(Rpsm_position3, Rpsm_velocity6, Rpsm_timestamp, gazepoint_position3)
	# 	distance_psms = np.sqrt((Lpsm_position3[0]-Rpsm_position3[0])**2 + (Lpsm_position3[1]-Rpsm_position3[1])**2 + (Lpsm_position3[2]-Rpsm_position3[2])**2)
	# 	# IPA calculation
	# 	global ipaL_data
	# 	global ipaR_data

	# 	if(len(pupilL)>=128 and len(pupilR)>=128):
	# 		_ipaL_data = ipa_t.ipa_cal(pupilL)
	# 		_ipaR_data = ipa_t.ipa_cal(pupilR)
	# 		if(_ipaL_data!=0):
	# 			ipaL_data = _ipaL_data
	# 		if(_ipaR_data!=0):
	# 			ipaR_data = _ipaR_data
	# 	else:

	# 		ipaL_data = 1
	# 		ipaR_data = 1

	# 	psm_position = [Lpsm_position3, Rpsm_position3]
	# 	distance = [Ldistance, Rdistance]
	# 	psm_velocity = [Lpsm_velocity, Rpsm_velocity]

	# 	# Update attention heatmap with current gaze data
		
	# 	#current_time = time.time()
	# 	current_time = latest_gaze_timestamp
	# 	current_gaze_point = latest_gaze_point
		
	# 	# 归一化坐标计算
	# 	#current_gaze_point = [gazePoint[0] / resolution_x, gazePoint[1] / resolution_y]
		
	# 	self.attention_heatmap_generator.gaze_history.append((current_time, current_gaze_point[0], current_gaze_point[1]))
	# 	window_mode = self.attention_heatmap_generator.update_realtime_heatmap(current_time, use_filtered_points=False)
		    
	# 	# 使用3D坐标作为手部位置数据
	# 	normalized_Lpsm = self.normalize_3d_position(Lpsm_position3)
	# 	normalized_Rpsm = self.normalize_3d_position(Rpsm_position3)
			
	# 	# 存储到可视化数据
	# 	self.attention_heatmap_generator.all_timestamps.append(current_time)
	# 	self.attention_heatmap_generator.all_gaze_points.append(current_gaze_point)
	# 	self.attention_heatmap_generator.all_hand_positions.append([normalized_Lpsm, normalized_Rpsm])
	# 	self.attention_heatmap_generator.all_realtime_heatmaps.append(self.attention_heatmap_generator.realtime_heatmap.copy())

	# 	current_idx = len(self.attention_heatmap_generator.all_timestamps) - 1

	# 	# 调用过滤方法
	# 	try:
	# 		is_valid, attention_value = self.attention_heatmap_generator.filter_outliers_focused(
	# 			current_idx,
	# 			self.attention_heatmap_generator.all_gaze_points,
	# 			self.attention_heatmap_generator.all_timestamps
	# 		)
	# 	except Exception as e:
	# 		# print(f"过滤错误: {e}")  # 错误输出
	# 		is_valid = True

	# 	if not is_valid:
	# 		self.attention_heatmap_generator.all_timestamps.pop()
	# 		self.attention_heatmap_generator.all_gaze_points.pop()
	# 		self.attention_heatmap_generator.all_hand_positions.pop()
	# 		self.attention_heatmap_generator.all_realtime_heatmaps.pop()
			
	# 		# print(f"过滤掉异常点: {current_gaze_point}, 注意力值: {attention_value:.3f}")
	# 	else:
	# 		self.attention_heatmap_generator.filtered_indices.append(current_idx)
	# 		self.attention_heatmap_generator.filtered_gaze_history.append((current_time, current_gaze_point[0], current_gaze_point[1]))
	# 		self.attention_heatmap_generator.update_realtime_heatmap(current_time, use_filtered_points=True)
	# 	# calculate and publish scale
		
	# 	#print(f"<LYON> Using parameters: {params}")
	# 	#scale= scale_cal(distance, psm_velocity, ipaL_data, ipaR_data, self.params, distance_psms)
	# 	scale = self.calculate_scale(gazepoint_position3,psm_position, psm_velocity, ipaL_data, ipaR_data, distance_psms, is_valid)	
	# 	try:  
	# 		self.scale_pub.publish(scale)
	# 	except rospy.ROSException as e:
	# 		rospy.logwarn(f"Failed to publish scale: {e}")

	# 	scale_list.append(scale.data)
	# 	self.attention_heatmap_generator.all_scale_factors.append(np.mean(scale.data))

	# 	cal_total_distance(Lpsm_position3, Rpsm_position3)
	# 	# calculate total time
	# 	total_time_list[0] = float(time.time() - start_time)

	# 	# print the data
	# 	if self.collecting:
	# 		print("\n")
	# 		print("=" * 50)
	# 		print(f"{'PSM Left 3d Position':<25}: [{psm_position[0][0]:.3f}, {psm_position[0][1]:.3f}, {psm_position[0][2]:.3f}]")
	# 		print(f"{'PSM Right 3d Position':<25}: [{psm_position[1][0]:.3f}, {psm_position[1][1]:.3f}, {psm_position[1][2]:.3f}]")
	# 		print(f"{'Gazing Point ':<25}: [{gazePoint[0]:.3f}, {gazePoint[1]:.3f}]")
	# 		print(f"{'PSM Left Gazing Point Distance':<25}: {distance[0]:.3f}")
	# 		print(f"{'PSM Right Gazing Point Distance':<25}: {distance[1]:.3f}")
	# 		print(f"{'PSM Left Velocity (x, y, z)':<25}: [{psm_velocity[0]:.3f}]")
	# 		print(f"{'PSM Right Velocity (x, y, z)':<25}: [{psm_velocity[1]:.3f}]")
	# 		print(f"{'IPA Left Data':<25}: [{ipaL_data:.3f}]")
	# 		print(f"{'IPA Right Data':<25}: [{ipaR_data:.3f}]")
	# 		print(f"{'Scale Left':<25}: {scale.data[0]:.8f}")
	# 		print(f"{'Scale Right':<25}: {scale.data[1]:.8f}")
	# 		print("=" * 50)
	# 		print("\n")

	# 	# store the data
	# 	GP_distance_list.append(distance) # distance = [Ldistance, Rdistance]
	# 	psms_distance_list.append(distance_psms) # distance_psms = np.sqrt((Lpsm_position3[0]-Rpsm_position3[0])**2 + (Lpsm_position3[1]-Rpsm_position3[1])**2 + (Lpsm_position3[2]-Rpsm_position3[2])**2)
	# 	Lpsm_pose_list.append(np.append(Lpsm_position3, Lpsm_timestamp)) # add timestamp to Lpsm_position3
	# 	Rpsm_pose_list.append(np.append(Rpsm_position3, Rpsm_timestamp)) # add timestamp to Rpsm_position3	
	# 	Lpsm_velocity_list.append(Lpsm_velocity6)
	# 	Rpsm_velocity_list.append(Rpsm_velocity6)
	# 	if(ipaL_data != 0):
	# 		ipaL_data_list.append(ipaL_data)
	# 	if(ipaR_data != 0):
	# 		ipaR_data_list.append(ipaR_data)


	def normalize_3d_position(self, position_3d):
		"""
		将3D坐标归一化到[0,1]范围用于热图计算
		这里需要根据工作空间范围调整
		"""
		# 假设工作空间范围（需要根据实际情况调整）
		# to do 
		workspace_min = [-0.5, -0.5, 0.0]   # 最小坐标
		workspace_max = [0.5, 0.5, 0.5]     # 最大坐标

		normalized = []
		for i in range(3):
			# 将每个维度归一化到[0,1]
			norm_val = (position_3d[i] - workspace_min[i]) / (workspace_max[i] - workspace_min[i])
			normalized.append(np.clip(norm_val, 0, 1))

		return normalized

	def calculate_scale(self, gaze_position_3d, psm_position, psm_velocity, IPAL, IPAR, distance_psms, is_valid):
		scaleArray = Float32MultiArray()
		
		if not is_valid:
			# 被过滤的点：使用上一帧的缩放因子
			scaleArray.data = self.attention_heatmap_generator.prev_scale_factor
			return scaleArray

		# 使用热图加权距离替换原有的距离特征
		# 将3D坐标归一化用于热图计算
		normalized_Lpsm = self.normalize_3d_position(psm_position[0])
		normalized_Rpsm = self.normalize_3d_position(psm_position[1])

		# 计算热图加权距离
		weighted_dist_L = self.attention_heatmap_generator.calculate_weighted_distance(gaze_position_3d,normalized_Lpsm)
		weighted_dist_R = self.attention_heatmap_generator.calculate_weighted_distance(gaze_position_3d,normalized_Rpsm)

		# 保持原有的缩放因子计算方式，但用热图加权距离替换距离特征
		IPA_m = (IPAL + IPAR) / 2

		# 使用热图加权距离作为距离特征
		N_d_L = normalize(weighted_dist_L, self.params['d_min'], self.params['d_max'], 1)
		N_d_R = normalize(weighted_dist_R, self.params['d_min'], self.params['d_max'], 1)
		N_p = normalize(IPA_m, self.params['p_min'], self.params['p_max'], 1)
		N_v_L = normalize(psm_velocity[0], self.params['v_min'], self.params['v_max'], 1)
		N_v_R = normalize(psm_velocity[1], self.params['v_min'], self.params['v_max'], 1)
		N_s = normalize(distance_psms, self.params['s_min'], self.params['s_max'], 1) 

		# 原有的tan函数计算方式
		T_d_L = np.tan(0.49 * np.pi * self.params['tau_d'] * N_d_L)
		T_d_R = np.tan(0.49 * np.pi * self.params['tau_d'] * N_d_R)
		T_p = np.tan(0.49 * np.pi * self.params['tau_p'] * N_p)
		T_v_L = np.tan(0.49 * np.pi * self.params['tau_v'] * N_v_L)
		T_v_R = np.tan(0.49 * np.pi * self.params['tau_v'] * N_v_R)
		T_s = np.tan(0.49 * np.pi * self.params['tau_s'] * N_s)

		weighted_sum_L = (
			self.params['W_d'] * T_d_L +
			self.params['W_p'] * T_p +
			self.params['W_v'] * T_v_L +
			self.params['W_s'] * T_s +
			self.params['W_dps'] * T_d_L * T_p * T_s +
			self.params['W_dvs'] * T_d_L * T_v_L * T_s +
			self.params['W_pvs'] * T_p * T_v_L * T_s +
			self.params['W_dpv'] * T_d_L * T_p * T_v_L +
			self.params['W_dpvs'] * T_d_L * T_p * T_v_L * T_s +
			self.params['C_offset']
		)

		output_L = self.params['Y_base'] * weighted_sum_L

		weighted_sum_R = (
			self.params['W_d'] * T_d_R +
			self.params['W_p'] * T_p +
			self.params['W_v'] * T_v_R +
			self.params['W_s'] * T_s +
			self.params['W_dps'] * T_d_R * T_p * T_s +
			self.params['W_dvs'] * T_d_R * T_v_R * T_s +
			self.params['W_pvs'] * T_p * T_v_R * T_s +
			self.params['W_dpv'] * T_d_R * T_p * T_v_R +
			self.params['W_dpvs'] * T_d_R * T_p * T_v_R * T_s +
			self.params['C_offset']
		)

		output_R = self.params['Y_base'] * weighted_sum_R

		scaleArray.data = [np.clip(output_L, 0.1, 4), np.clip(output_R, 0.1, 4)]

		self.attention_heatmap_generator.prev_scale_factor = scaleArray.data.copy()

		return scaleArray

	def get_position(self,pose):
		return np.array([pose.position.x,pose.position.y,pose.position.z])
	
	def get_velocity(self,twist):
		return np.array([twist.linear.x,twist.linear.y,twist.linear.z])

	def orientation2numpy(pose):
		return np.array([pose.orientation.x,pose.orientation.y,pose.orientation.z,pose.orientation.w])


	def print_statistics(self):
		#print msg container info
		print("\n")
		print(f"\n{'='*50}")
		print(f"Features collected:")
		print(f"{'LPSM Pose List Length':<25}: {len(Lpsm_pose_list)}")
		print(f"{'RPSM Pose List Length':<25}: {len(Rpsm_pose_list)}")
		print(f"{'LPSM Velocity List Length':<25}: {len(Lpsm_velocity_list)}")
		print(f"{'RPSM Velocity List Length':<25}: {len(Rpsm_velocity_list)}")
		print(f"{'Gaze and PSM distance List Length':<25}: {len(GP_distance_list)}")
		print(f"{'PSMs distance List Length':<25}: {len(psms_distance_list)}")
		print(f"{'LPSM Velocity List Length':<25}: {len(Lpsm_velocity_list)}")
		print(f"{'RPSM Velocity List Length':<25}: {len(Rpsm_velocity_list)}")
		print(f"{'Gazepoint List Length':<25}: {len(gazepoint_list)}")
		print(f"{'Gaze List Length':<25}: {len(gaze_list)}")
		print(f"{'IPA Left Data List Length':<25}: {len(ipaL_data_list)}")
		print(f"{'IPA Right Data List Length':<25}: {len(ipaR_data_list)}")
		print(f"{'Pupil Left Data List Length':<25}: {len(pupilL_list)}")
		print(f"{'Pupil Right Data List Length':<25}: {len(pupilR_list)}")
		print(f"pupilL_list Length:{len(pupilL)}")
		print(f"pupilL_list Length:{len(pupilR)}")

		print(f"\n{'='*50}")
		print(f"Gaze filtering results:")
		print(f"{'Total gaze points':<25}: {len(self.attention_heatmap_generator.all_gaze_points)}")
		print(f"{'Valid gaze points':<25}: {len(self.attention_heatmap_generator.filtered_indices)}")
		print(f"{'Outlier gaze points':<25}: {len(self.attention_heatmap_generator.outlier_indices)}")
		print(f"{'Valid ratio':<25}: {len(self.attention_heatmap_generator.filtered_indices)/len(self.attention_heatmap_generator.all_gaze_points):.1%}")
		print(f"{'Outlier ratio':<25}: {len(self.attention_heatmap_generator.outlier_indices)/len(self.attention_heatmap_generator.all_gaze_points):.1%}")

		print(f"\n{'='*50}")
		print(f"Global factors:")
		print(f"  {'Total time':<12} : {total_time_list[0]:.6f}")
		# to fix: bimanual mode print
		print(f"  {'Total distance':<12} : {total_distance_list[0]:.6f}")
		print(f"  {'Clutch times':<12} : [{clutch_times_list[0]}, {clutch_times_list[1]}]")
		print(f"{'='*50}")


	def load_params(self):
		"""
		Load the parameters.
		"""
		params = init_params.copy()

		current_file_path = os.path.abspath(__file__)
		current_dir = os.path.dirname(current_file_path)

		filename = os.path.join(current_dir, 'params', 'params.txt')
		# Load parameters from the file if it exists
		try:
			with open(filename, 'r') as f:
				for line in f:
					key, value = line.strip().split('=')
					params[key] = float(value)
		except FileNotFoundError:
			if self.collecting:
				print("params.txt not found, using default parameters.")

		return params

	def create_save_dir(self):
		path = './data'
		num_dirs = sum(1 for name in os.listdir(path) if os.path.isdir(os.path.join(path, name)))
		s1 = datetime.datetime.today().strftime("%m-%d")   
		dir_name = f'./data/{num_dirs}_data_{s1}'
		os.makedirs(dir_name, exist_ok=True)
		self.attention_heatmap_generator.set_save_dir(dir_name)
		
	def run(self):
		"""Main run loop (your function remains unchanged)"""
		while not rospy.is_shutdown():
			# First, reset the data to be used
			self.reset_globals()
			self.params = self.load_params()
			
			# Start data collection
			self.start_collection()
			self.wait_for_completion()
			self.create_save_dir()
			self.gaze_heatmap_visualization()
			self.print_statistics()
			
			
			# Save data
			save_data_cb()
			flush_input()
			# Ask whether to continue
			try:
				#print(f"\n当前使用的参数: {self.params}")
				ok = input("是否进行下一次数据收集？(y or n): ")
				if ok.lower() != "y":
					break
			except KeyboardInterrupt:
				# Using Ctrl+C here to exit the entire program is reasonable
				print("\n程序退出")
				break

def flush_input():
	"""Clear the standard input buffer according to the operating system"""
	try:
		# For Linux and macOS
		if platform.system() in ["Linux", "Darwin"]:
			import termios
			termios.tcflush(sys.stdin, termios.TCIFLUSH)
		# For Windows
		elif platform.system() == "Windows":
			import msvcrt
			while msvcrt.kbhit():
				msvcrt.getch()
	except (ImportError, OSError) as e:
		# May fail in some special environments (e.g., non-interactive scripts)
		print(f"清空输入缓冲区时出错 (可以忽略): {e}")


def cal_metrics(psm_position3,psm_velocity6,timestamp,gazepoint_position3):
	"""
	calculate the features
	- distance between psm and gazing point
	- psm linear velocity
	"""
	distance = np.sqrt((psm_position3[0]-gazepoint_position3[0])**2 + (psm_position3[1]-gazepoint_position3[1])**2 + (psm_position3[2]-gazepoint_position3[2])**2)
	psm_linear_velocity = np.sqrt(psm_velocity6[0]**2 + psm_velocity6[1]**2 + psm_velocity6[2]**2)
	return distance, psm_linear_velocity

# gaze point related

def get_gazepoint_position3(point_cloud,cameraL_pose, cameraFrame_pose,u,v):

	u = int(u)
	v = int(v)
	T_FL = get_transformation_matrix(position2numpy(cameraL_pose),orientation2numpy(cameraL_pose)) # cameraL to cameraFrame
	T_WF = get_transformation_matrix(position2numpy(cameraFrame_pose),orientation2numpy(cameraFrame_pose)) # cameraFrame to world
		
	point = next(read_points(point_cloud, field_names=("x","y","z"), skip_nans=False, uvs=[(u,v)]))
	T_WL = T_WF.dot(T_FL)
	point = T_WL.dot(np.array(list(point)+[1]).T)
	gazepoint_list.append([u,v])
	return point

def read_points(cloud, field_names=None, skip_nans=False, uvs=[]):
	assert isinstance(cloud, roslib.message.Message) and cloud._type == 'sensor_msgs/PointCloud2', 'cloud is not a sensor_msgs.msg.PointCloud2'
	fmt = _get_struct_fmt(cloud.is_bigendian, cloud.fields, field_names)
	width, height, point_step, row_step, data, isnan = cloud.width, cloud.height, cloud.point_step, cloud.row_step, cloud.data, math.isnan
	unpack_from = struct.Struct(fmt).unpack_from

	if skip_nans:
		if uvs:
			for u, v in uvs:
				p = unpack_from(data, (row_step * v) + (point_step * u))
				has_nan = False
				for pv in p:
					if isnan(pv):
						has_nan = True
						break
				if not has_nan:
					yield p
		else:
			for v in range(height):
				offset = row_step * v
				for u in range(width):
					p = unpack_from(data, offset)
					has_nan = False
					for pv in p:
						if isnan(pv):
							has_nan = True
							break
					if not has_nan:
						yield p
					offset += point_step
	else:
		if uvs:
			for u, v in uvs:
				yield unpack_from(data, np.clip((720 * point_step * v) + (point_step * u),0,720*480*point_step-12))
		else:
			for v in range(height):
				offset = row_step * v
				for u in range(width):
					yield unpack_from(data, offset)
					offset += point_step

def _get_struct_fmt(is_bigendian, fields, field_names=None):
	fmt = '>' if is_bigendian else '<'

	offset = 0
	for field in (f for f in sorted(fields, key=lambda f: f.offset) if field_names is None or f.name in field_names):
		if offset < field.offset:
			fmt += 'x' * (field.offset - offset)
			offset = field.offset
		if field.datatype not in _DATATYPES:
			print('Skipping unknown PointField datatype [%d]' % field.datatype, file=sys.stderr)
		else:
			datatype_fmt, datatype_length = _DATATYPES[field.datatype]
			fmt    += field.count * datatype_fmt
			offset += field.count * datatype_length

	return fmt

def get_transformation_matrix(position, quaternion):
	
	
	r = R.from_quat(quaternion)  
	rotation_matrix = r.as_matrix()  

	
	transformation_matrix = np.eye(4)  
	transformation_matrix[:3, :3] = rotation_matrix  
	transformation_matrix[:3, 3] = position  

	return transformation_matrix

# -------------------------------------------------------------


def normalize(value, min_val, max_val, corr):
	if corr == 1:
		normalized_value = np.clip((value - min_val) / (max_val - min_val),0,1)
	else: 
		normalized_value = np.clip((-value + max_val) / (max_val - min_val),0,1)

	return normalized_value


def scale_cal(distance_GP, velocity_psm, IPAL, IPAR, params, distance_psms=None):	

	scaleArray = Float32MultiArray()

	# to fix: bimanual mode's scale
	#scale = k_s * (np.exp((min(1, (distance_GP)/(d_GP_max)) * min(1, velocity_psm/v_psm_max+0.01) * \
	#	max(0, (ipa_max-IPAL)/ipa_max) * max(0, (ipa_max-IPAR)/ipa_max) )**0.2 *2-1) /2.35*9-0.4089)		

	# the left one is the left touch, the right one is the right touch
	# IPA_m = (IPAL+IPAR)/2
	# N_d_L = normalize(distance_GP[0], params['d_min'], params['d_max'],1)
	# N_d_R = normalize(distance_GP[1], params['d_min'], params['d_max'],1)
	# N_p = normalize(IPA_m, params['p_min'], params['p_max'],1)
	# N_v_L = normalize(velocity_psm[0], params['v_min'], params['v_max'],1)
	# N_v_R = normalize(velocity_psm[1], params['v_min'], params['v_max'],1)
	# N_s = normalize(distance_psms, params['s_min'], params['s_max'],1) 

	# # # (G_x)
	# G_d_L = params['A_d'] if N_d_L >= params['tau_d'] else 1.0
	# G_d_R = params['A_d'] if N_d_R >= params['tau_d'] else 1.0
	# G_p = params['A_p'] if N_p >= params['tau_p'] else 1.0
	# G_v_L = params['A_v'] if N_v_L >= params['tau_v'] else 1.0
	# G_v_R = params['A_v'] if N_v_R >= params['tau_v'] else 1.0
	# G_s = params['A_s'] if N_s >= params['tau_s'] else 1.0

	# weighted_sum_L = (
	#	 params['W_d'] * N_d_L +
	#	 params['W_p'] * N_p +
	#	 params['W_v'] * N_v_L +
	#	 params['W_dps'] * N_d_L * N_p * N_s +
	#	 params['W_dvs'] * N_d_L * N_v_L * N_s +
	#	 params['W_pvs'] * N_p * N_v_L * N_s +
	#	 params['W_dpv'] * N_d_L * N_p * N_v_L +
	#	 params['W_dpvs'] * N_d_L * N_p * N_v_L * N_s +
	#	 params['C_offset']
	# )
	
	# output_L = params['Y_base'] * G_d_L * G_p * G_v_L * G_s* weighted_sum_L

	# # right 
	# weighted_sum_R = (
	#	 params['W_d'] * N_d_R +
	#	 params['W_p'] * N_p +
	#	 params['W_v'] * N_v_R +
	#	 params['W_dps'] * N_d_R * N_p * N_s  +
	#	 params['W_dvs'] * N_d_R * N_v_R * N_s  +
	#	 params['W_pvs'] * N_p * N_v_R * N_s  +
	#	 params['W_dpv'] * N_d_R * N_p * N_v_R +
	#	 params['W_dpvs'] * N_d_R * N_p * N_v_R * N_s +
	#	 params['C_offset']
	# )
	
	# output_R = params['Y_base'] * G_d_R * G_p * G_v_R * weighted_sum_R		

	""" tan func"""
	IPA_m = (IPAL+IPAR)/2
	N_d_L = normalize(distance_GP[0], params['d_min'], params['d_max'],1)
	N_d_R = normalize(distance_GP[1], params['d_min'], params['d_max'],1)
	N_p = normalize(IPA_m, params['p_min'], params['p_max'],1)
	N_v_L = normalize(velocity_psm[0], params['v_min'], params['v_max'],1)
	N_v_R = normalize(velocity_psm[1], params['v_min'], params['v_max'],1)
	N_s = normalize(distance_psms, params['s_min'], params['s_max'],1) 

	T_d_L = np.tan(0.49* np.pi * params['tau_d'] * N_d_L)
	T_d_R = np.tan(0.49* np.pi * params['tau_d'] * N_d_R)
	T_p = np.tan(0.49* np.pi * params['tau_p'] * N_p)
	T_v_L = np.tan(0.49* np.pi * params['tau_v'] * N_v_L)
	T_v_R = np.tan(0.49* np.pi * params['tau_v'] * N_v_R)
	T_s = np.tan(0.49* np.pi * params['tau_s'] * N_s)

	weighted_sum_L = (
		params['W_d'] * T_d_L +
		params['W_p'] * T_p +
		params['W_v'] * T_v_L +
		params['W_s'] * T_s +
		params['W_dps'] * T_d_L * T_p * T_s +
		params['W_dvs'] * T_d_L * T_v_L * T_s +
		params['W_pvs'] * T_p * T_v_L * T_s +
		params['W_dpv'] * T_d_L * T_p * T_v_L +
		params['W_dpvs'] * T_d_L * T_p * T_v_L * T_s +
		params['C_offset']
	)
	
	output_L = params['Y_base'] * weighted_sum_L

	weighted_sum_R = (
		params['W_d'] * T_d_R +
		params['W_p'] * T_p +
		params['W_v'] * T_v_R +
		params['W_s'] * T_s +
		params['W_dps'] * T_d_R * T_p * T_s  +
		params['W_dvs'] * T_d_R * T_v_R * T_s  +
		params['W_pvs'] * T_p * T_v_R * T_s  +
		params['W_dpv'] * T_d_R * T_p * T_v_R +
		params['W_dpvs'] * T_d_R * T_p * T_v_R * T_s +
		params['C_offset']
	)
	
	output_R = params['Y_base'] * weighted_sum_R

	scaleArray.data = [np.clip(output_L,0.1,4), np.clip(output_R,0.1,4)]


	return scaleArray

def sigmoid(x):
	return 1 / (1 + np.exp(-x))

"""
The functions for calculating features
"""
def cal_total_distance(Lpsm_position3,Rpsm_position3):
	# calculate total distance
	global psmPosePreL
	global psmPosePreR
	global flag
	if(flag):
		total_distance_list[0] += ((psmPosePreL[0]-Lpsm_position3[0])**2+(psmPosePreL[1]-Lpsm_position3[1])**2+(psmPosePreL[2]-Lpsm_position3[2])**2)**(1/2)
		total_distance_list[0] += ((psmPosePreR[0]-Rpsm_position3[0])**2+(psmPosePreR[1]-Rpsm_position3[1])**2+(psmPosePreR[2]-Rpsm_position3[2])**2)**(1/2)
	flag = True
	psmPosePreL = Lpsm_position3
	psmPosePreR = Rpsm_position3

def cal_Lpsm_clutch_times(button):
	if button.grey_button == 1:
		clutch_times_list[0] += 1
def cal_Rpsm_clutch_times(button):
	if button.grey_button == 1:
		clutch_times_list[1] += 1


"""
if left gaze point and right are both available, we set gaze point as their medium point.If only one, set as that.If none, no change.
"""
def gaze_data_cb(gaze):
	#print(f"<LYON> Gaze data received: {gaze.data}")
	w = resolution_x
	h = resolution_y

	gazedata = list(gaze.data)
	gazedata[1] = np.clip(gazedata[1],0,1)
	gazedata[2] = np.clip(gazedata[2],0,1)
	gazedata[4] = np.clip(gazedata[4],0,1)
	gazedata[5] = np.clip(gazedata[5],0,1)

	gaze_timestamp = gazedata[8] * 1e-6

	if (len(pupilL)>0 and gazedata[8]*1e-6 == pupilL[-1].get_timestamp()):
		return
	if(gazedata[0] == True):
		if(gazedata[3] == True):
			gazePoint[0] = np.round(((gazedata[1]+gazedata[4])/2)*w).astype(int)
			gazePoint[1] = h-np.round(((gazedata[2]+gazedata[5])/2)*h).astype(int)

		else:
			gazePoint[0] = np.round((gazedata[1])*w).astype(int)
			gazePoint[1] = h-np.round((gazedata[2])*h).astype(int)
	elif(gazedata[3] == True):
		gazePoint[0] = np.round((gazedata[4])*w).astype(int)
		gazePoint[1] = h-np.round((gazedata[5])*h).astype(int)
	if (gazedata[6] != 0):
		pupilL.append(ipa_t.Pupil(gazedata[6], gazedata[8]*1e-6))
		pupilL_list.append(ipa_t.Pupil(gazedata[6], gazedata[8]*1e-6))
		
	else:
		if pupilL:
			last_pupilL = pupilL[-1]
			pupilL.append(last_pupilL)
			pupilL_list.append(last_pupilL)
	if (gazedata[7] != 0):
		pupilR.append(ipa_t.Pupil(gazedata[7], gazedata[8]*1e-6))
		pupilR_list.append(ipa_t.Pupil(gazedata[7], gazedata[8]*1e-6))
		
	else:
		if pupilR:
			last_pupilR = pupilR[-1]
			pupilR.append(last_pupilR)
			pupilR_list.append(last_pupilR)
	gazepoint_list.append([gazePoint[0], gazePoint[1],gazedata[8]*1e-6])
	gaze_list.append(gazedata)

	# 更新最新的gaze时间戳
	global latest_gaze_timestamp, latest_gaze_point
	latest_gaze_timestamp = gaze_timestamp
	latest_gaze_point = [gazePoint[0] / resolution_x, gazePoint[1] / resolution_y]
	
	return gaze.data


"""
Utility functions
"""

# change pose.position to numpy array
def position2numpy(pose):
	return np.array([pose.position.x,pose.position.y,pose.position.z])


# change pose.orientation to numpy array
def orientation2numpy(pose):
	return np.array([pose.orientation.x,pose.orientation.y,pose.orientation.z,pose.orientation.w])


def get_transformation_matrix(position, quaternion):
	
	
	r = R.from_quat(quaternion)  
	rotation_matrix = r.as_matrix()  

	
	transformation_matrix = np.eye(4)  
	transformation_matrix[:3, :3] = rotation_matrix  
	transformation_matrix[:3, 3] = position  

	return transformation_matrix


def save_data_cb():
	"""
	data storage callback, invoked when script terminates
	"""
	current_file_path = os.path.abspath(__file__)
	
	current_dir = os.path.dirname(current_file_path)
	s1 = datetime.datetime.today().strftime("%m-%d")   
	data_base_dir = os.path.join(current_dir, 'data')
	latest_dir = get_latest_data_dir(data_base_dir)

	print(f"saving data to {latest_dir}...")
	np.save(join(latest_dir, 'Lpsm_position.npy'), Lpsm_pose_list)
	np.save(join(latest_dir, 'Rpsm_position.npy'), Rpsm_pose_list)
	np.save(join(latest_dir, 'Lpsm_velocity.npy'), Lpsm_velocity_list)
	np.save(join(latest_dir, 'Rpsm_velocity.npy'), Rpsm_velocity_list)
	np.save(join(latest_dir, 'gazepoint_position_data.npy'), gazepoint_list)
	np.save(join(latest_dir, 'gaze_data.npy'), gaze_list)
	  
	np.save(join(latest_dir, 'GP_distance_data.npy'), GP_distance_list)
	np.save(join(latest_dir, 'psms_distance_data.npy'), psms_distance_list)
	np.save(join(latest_dir, 'ipaL_data.npy'), ipaL_data_list)
	np.save(join(latest_dir, 'ipaR_data.npy'), ipaR_data_list)
	  
	np.save(join(latest_dir, 'clutch_times.npy'), clutch_times_list)
	np.save(join(latest_dir, 'total_distance.npy'), total_distance_list)
	np.save(join(latest_dir, 'total_time.npy'), total_time_list)


	np.save(join(latest_dir, 'pupilL_data.npy'), pupilL_list)
	np.save(join(latest_dir, 'pupilR_data.npy'), pupilR_list)
	np.save(join(latest_dir, 'scale_data.npy'), scale_list)
	print("done saving...")


if __name__ == '__main__':
	args = sys.argv[1:]

	if len(args) > 0 :
		exflag = float(args[0])

	print("start data collection...")

	collector = DataCollector()
	collector.run()
