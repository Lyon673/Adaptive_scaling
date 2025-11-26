from params import config
import numpy as np
import os
import matplotlib.pyplot as plt
from gracefulness import get_latest_data_dir


class AttentionHeatmapGenerator:
	def __init__(self, screen_width=config.resolution_x, screen_height=config.resolution_y, heatmap_size=(config.resolution_x/10, config.resolution_y/10)):
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
		self.all_gaze_points = []  # 存储所有历史注视点 (timestamp, x, y)
		self.all_timestamps = []

		# For visualization
		self.filtered_indices = []
		self.outlier_indices = []
		self.filtered_timestamps = []
		self.filtered_gaze_points = []
        # 添加三维坐标存储
		self.all_gaze_points_3d = []  # 存储三维注视点坐标 [(x, y, z), ...]
		self.all_hand_positions_3d = []  # 存储三维手部坐标 [[left_x, left_y, left_z], [right_x, right_y, right_z]]
		self.gaze_points_3d_with_time = []

		# 添加加权距离历史
		self.weighted_distances_left = []  # 左手加权距离历史
		self.weighted_distances_right = []  # 右手加权距离历史		

		# Scaling factor parameters
		self.scale_params = config.gaze_filter_params['scale_params']


		# Outlier filtering parameters - more focused on significant outliers
		self.filter_params = config.gaze_filter_params['filter_params']	

		# Fixed window configuration 
		self.fixed_window_config = config.gaze_filter_params['fixed_window_config']

		# 初始化上一帧的缩放因子
		self.prev_scale_factor = config.gaze_filter_params['prev_scale_factor']
		self.prev_valid_gaze = None

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
		#print(f"gaze heatmap results will be saved to: {os.path.abspath(self.save_dir)}")

	def update_realtime_heatmap(self, current_time, total_duration=None, use_filtered_points=True):
		"""
		更新实时热图
		
		Parameters:
			current_time: 当前时间
			total_duration: 总采集时间（如果已知）
			use_filtered_points: 是否使用过滤后的点
		"""
		filtered_gaze_points = self.filtered_gaze_points.copy()
		# 清空热图
		self.realtime_heatmap = np.zeros(self.heatmap_size)
		
		window_mode = "moving window"
		
		# 计算窗口开始时间：max(0, current_time - window_seconds)
		window_start = max(0, current_time - self.filter_params['window_seconds'])
		window_end = current_time
		#print(f"window_start: {window_start}, window_end: {window_end}")
		
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
			for timestamp, gx, gy in filtered_gaze_points:
				if window_start <= timestamp < window_end:
					# Convert to heatmap coordinates
					x_heatmap = int(gx * (self.heatmap_size[1] - 1))
					y_heatmap = int(gy * (self.heatmap_size[0] - 1))
					
					# Ensure coordinates are valid
					x_heatmap = max(0, min(x_heatmap, self.heatmap_size[1] - 1))
					y_heatmap = max(0, min(y_heatmap, self.heatmap_size[0] - 1))
					
					window_points.append((x_heatmap, y_heatmap))
		else:
			# 使用所有点
			for timestamp, gx, gy in self.all_gaze_points:
				if window_start <= timestamp <= window_end:
					# Convert to heatmap coordinates
					x_heatmap = int(gx * (self.heatmap_size[1] - 1))
					y_heatmap = int(gy * (self.heatmap_size[0] - 1))
					
					# Ensure coordinates are valid
					x_heatmap = max(0, min(x_heatmap, self.heatmap_size[1] - 1))
					y_heatmap = max(0, min(y_heatmap, self.heatmap_size[0] - 1))
					
					window_points.append((x_heatmap, y_heatmap))
		#print(f"window_points: {window_points}")
		#print(f"filtered_gaze_points: {filtered_gaze_points}")
		if not window_points:
			return
		
		# to do 
		sigma = config.gaze_filter_params['gaussian_kernel_sigma']
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
	
	def filter_outliers_focused(self, gaze_points, timestamps):
		"""
		专注于过滤明显异常点的函数 - 适配实时模式
		"""
		# gaze_points 格式是 [[timestamp, x, y], ...] 或 [(timestamp, x, y), ...] x,y ratio
		# 提取 [x, y] 部分用于后续处理
		if len(gaze_points) == 0:
			return False, 0.0
		
		# 将 gaze_points 转换为只包含 [x, y] 的列表
		gaze_points_xy = []
		for p in gaze_points:
			if len(p) >= 3:  # 确保是 [timestamp, x, y] 或 (timestamp, x, y) 格式
				gaze_points_xy.append([p[1], p[2]])  # 提取 x, y
			elif len(p) >= 2:  # 如果已经是 [x, y] 格式
				gaze_points_xy.append([p[0], p[1]])
			else:
				gaze_points_xy.append([0.5, 0.5])  # 默认值
		

		current_point = gaze_points_xy[-1]
		
		# 1. 基于热图的过滤 - 只过滤非常低注意力值的点
		x_heatmap = int(current_point[0] * (self.heatmap_size[1] - 1))
		y_heatmap = int(current_point[1] * (self.heatmap_size[0] - 1))
		x_heatmap = max(0, min(x_heatmap, self.heatmap_size[1] - 1))
		y_heatmap = max(0, min(y_heatmap, self.heatmap_size[0] - 1))
		
		attention_value = self.realtime_heatmap[y_heatmap, x_heatmap]
		heatmap_valid = attention_value >= self.filter_params['attention_threshold']
		#print(f"attention_value: {attention_value}")
		# 2. 检测明显的跳跃
		jump_detected = False
		if len(gaze_points_xy) > 1:
			prev_point = gaze_points_xy[-2]
			jump_detected = self.detect_jump(current_point, prev_point)
		#print(f"jump_detected: {jump_detected}")
		# 3. 检查是否有足够的邻居支持
		has_neighbors = True
		if len(gaze_points_xy) > 10:  # 只有在有足够历史数据时才检查
			recent_points = gaze_points_xy[max(0, len(gaze_points_xy)-10):len(gaze_points_xy)]
			has_neighbors = self.has_sufficient_neighbors(current_point, recent_points)
		#print(f"has_neighbors: {has_neighbors}")
		# 4. 基于速度的异常检测
		velocity_outlier = False
		if len(gaze_points_xy) > 1:
			try:
				prev_point = gaze_points_xy[-2]
				prev_time = timestamps[-2]
				time_diff = timestamps[-1] - prev_time
				velocity_outlier = self.detect_velocity_outlier(current_point, prev_point, time_diff)
			except (IndexError, TypeError):
				velocity_outlier = False            
		#print(f"velocity_outlier: {velocity_outlier}")
		# 综合条件：当点在低注意力区域，或缺乏邻居支持，或速度异常，或是明显的跳跃时，被过滤
		is_outlier = not heatmap_valid or not has_neighbors or velocity_outlier or jump_detected
		problem_id = [not heatmap_valid, not has_neighbors, velocity_outlier, jump_detected]
		
		return not is_outlier, problem_id
	
	def calculate_weighted_distance(self, gaze_positions_3d, hand_position_3d):
		"""
		计算加权距离（基于时间窗口内的多个注视点）
		gaze_positions_3d: 时间窗口内的三维注视点坐标列表 [(x1,y1,z1), (x2,y2,z2), ...] 或单个坐标 [x, y, z]
		hand_position_3d: 单个手的三维坐标 [x, y, z]
		"""
		try:
			# 如果传入的是单个坐标而不是列表，转换为列表
			if hasattr(gaze_positions_3d, '__len__') and len(gaze_positions_3d) > 0:
				# 检查是否是单个3D坐标（如果第一个元素不是数组/列表）
				if not hasattr(gaze_positions_3d[0], '__len__'):
					gaze_positions_3d = [gaze_positions_3d]
			else:
				raise ValueError("gaze_positions_3d is not a list")
				return (self.scale_params['d_max'] + self.scale_params['d_min']) / 2
			
			# 安全检查
			if hand_position_3d is None:
				raise ValueError("hand_position_3d is None")
				return (self.scale_params['d_max'] + self.scale_params['d_min']) / 2
			
			# 确保手部坐标有效
			if (not hasattr(hand_position_3d, '__len__') or len(hand_position_3d) < 3):
				raise ValueError("hand_position_3d is not a list")
				return (self.scale_params['d_max'] + self.scale_params['d_min']) / 2
			
			# 过滤无效的注视点并计算距离
			valid_distances = []
			for gaze_pos in gaze_positions_3d:
				if (gaze_pos is not None and 
					hasattr(gaze_pos, '__len__') and 
					len(gaze_pos) >= 3 and
					not np.any(np.isnan(gaze_pos))):  # 修复：使用 np.any() 而不是 any()
					
					# 3D欧几里得距离
					direct_distance = np.sqrt(
						(float(gaze_pos[0]) - float(hand_position_3d[0]))**2 +
						(float(gaze_pos[1]) - float(hand_position_3d[1]))**2 +
						(float(gaze_pos[2]) - float(hand_position_3d[2]))**2
					)
					valid_distances.append(direct_distance)
				else:
					raise ValueError("gaze_pos is not a list")
			
			if  len(valid_distances) == 0:
				raise ValueError("valid_distances is empty")
			
			if not valid_distances:
				return (self.scale_params['d_max'] + self.scale_params['d_min']) / 2
			
			# 使用时间衰减权重（最近的注视点权重更高）
			weights = self.calculate_temporal_weights(len(valid_distances))
			
			# 计算加权平均距离
			weighted_dist = np.average(valid_distances, weights=weights)
			return weighted_dist
			
		except (TypeError, IndexError, ValueError) as e:
			print(f"Error calculating weighted 3D distance: {e}")
			return (self.scale_params['d_max'] + self.scale_params['d_min']) / 2

	def calculate_temporal_weights(self, n_points):
		"""
		计算时间衰减权重（指数衰减）
		"""
		if n_points == 0:
			return []
		
		# 指数衰减：最近的注视点权重最高
		decay_rate = config.gaze_filter_params['temporal_exponent_decay']
		weights = [decay_rate ** i for i in range(n_points-1, -1, -1)]
		
		# 归一化
		total_weight = sum(weights)
		if total_weight > 0:
			weights = [w / total_weight for w in weights]
		
		return weights
	
	def visualize_results(self, timestamps, gaze_points, 
						filtered_indices, outlier_indices):
		"""
		Visualize results with filtering comparison
		"""
		# 在函数开始时立即复制数据，确保数据在函数执行过程中不会被修改
		# 这样可以避免回调函数在可视化过程中继续添加数据
		timestamps = list(timestamps) if hasattr(timestamps, '__iter__') else timestamps
		gaze_points = list(gaze_points) if hasattr(gaze_points, '__iter__') else gaze_points
		filtered_indices = list(filtered_indices) if hasattr(filtered_indices, '__iter__') else filtered_indices
		outlier_indices = list(outlier_indices) if hasattr(outlier_indices, '__iter__') else outlier_indices
		
		# 记录初始长度用于调试
		initial_gaze_points_len = len(gaze_points)
		
		# 创建2x2的子图布局
		fig, axes = plt.subplots(2, 2, figsize=(15, 12))
		
		# 1. 全时间热图
		# 创建全时间的热图，叠加所有注视点
		full_time_heatmap = np.zeros(self.heatmap_size)
		
		# 叠加所有过滤后的注视点
		for i in filtered_indices:
			if i < len(gaze_points):
				gaze_point = gaze_points[i]
				# gaze_point 是 (timestamp, x, y) 元组
				x_heatmap = int(gaze_point[1] * (self.heatmap_size[1] - 1))
				y_heatmap = int(gaze_point[2] * (self.heatmap_size[0] - 1))
				x_heatmap = max(0, min(x_heatmap, self.heatmap_size[1] - 1))
				y_heatmap = max(0, min(y_heatmap, self.heatmap_size[0] - 1))
				
				# 为每个点添加高斯分布
				sigma = config.gaze_filter_params['gaussian_kernel_sigma']
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
		
		
		target_width = config.resolution_x
		target_height = config.resolution_y
		aspect_ratio = target_width / target_height
		
		# 显示全时间热图
		im = axes[0, 0].imshow(full_time_heatmap, cmap='hot', interpolation='nearest',
							extent=[0, target_width, target_height, 0], 
							aspect=aspect_ratio, origin='upper')
		
		# Plot all gaze points (转换为目标分辨率)
		# gaze_points 是 [(timestamp, x, y), ...] 格式
		# 再次检查长度，确保数据没有被修改
		current_gaze_points_len = len(gaze_points)
		if current_gaze_points_len != initial_gaze_points_len:
			print(f"<WARNING> gaze_points length changed from {initial_gaze_points_len} to {current_gaze_points_len}!")
		
		gaze_x = np.array([p[1] * target_width for p in gaze_points])
		gaze_y = np.array([p[2] * target_height for p in gaze_points])

		
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
			axes[0, 0].scatter(filtered_x, filtered_y, c='green', marker='o', s=20, label='Filtered Gaze')
		if outlier_x and outlier_y:
			axes[0, 0].scatter(outlier_x, outlier_y, c='red', marker='x', s=20, label='Outliers')
		#axes[0, 0].scatter(hand_x, hand_y, c='purple', marker='s', s=80, label='Hand Position')
		axes[0, 0].set_title(f'Full Time Heatmap (All Filtered Gaze Points) - {config.resolution_x}x{config.resolution_y}')
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
			min_length = min(len(times), len(GP_distance_list))
			
			if min_length > 0:
				times_trimmed = times[:min_length]
				GP_distance_array = np.array(GP_distance_list)
				left_distances = GP_distance_array[:min_length, 0]
				right_distances = GP_distance_array[:min_length, 1]
				
				# 绘制左右手的三维距离
				axes[0, 1].plot(times_trimmed, left_distances, '#3498db', linewidth=2, label='Left Hand 3D Distance')
				axes[0, 1].plot(times_trimmed, right_distances, '#e74c3c', linewidth=2, label='Right Hand 3D Distance')
				
				axes[0, 1].set_title('3D Weighted Distance Between Hands and Gaze Point')
				axes[0, 1].set_xlabel('Time (s)')
				axes[0, 1].set_ylabel('3D Distance (meters)')
				axes[0, 1].grid(True)
				axes[0, 1].legend()
				
				# 添加统计信息
				avg_left = np.mean(left_distances)
				avg_right = np.mean(right_distances)
				axes[0, 1].axhline(y=avg_left, color='#3498db', linestyle='--', alpha=0.7, 
									label=f'Left Avg: {avg_left:.3f}m')
				axes[0, 1].axhline(y=avg_right, color= '#e74c3c', linestyle='--', alpha=0.7,
									label=f'Right Avg: {avg_right:.3f}m')
				
			else:
				axes[0, 1].text(0.5, 0.5, 'No 3D distance data', 
							ha='center', va='center', transform=axes[0, 1].transAxes)
				axes[0, 1].set_title('3D Weighted Distance Over Time')
		else:
			axes[0, 1].text(0.5, 0.5, 'No 3D coordinate data available', 
						ha='center', va='center', transform=axes[0, 1].transAxes)
			axes[0, 1].set_title('3D Weighted Distance Over Time')

	
		# 3. Original gaze coordinates over time 
	
		min_length = min(len(timestamps), len(gaze_x), len(gaze_y))
		times_trimmed = (np.array(timestamps) - timestamps[0])[:min_length]
		gaze_x_trimmed = gaze_x[:min_length]
		gaze_y_trimmed = gaze_y[:min_length]
		

		axes[1, 0].plot(times_trimmed, gaze_x_trimmed, '#3498db', label='X Coordinate', alpha=0.7)
		axes[1, 0].plot(times_trimmed, gaze_y_trimmed, '#e74c3c', label='Y Coordinate', alpha=0.7)
		axes[1, 0].set_title('Original Gaze Coordinates Over Time')
		axes[1, 0].set_xlabel('Time (s)')
		axes[1, 0].set_ylabel('Screen Coordinates (pixels)')
		axes[1, 0].legend()
		axes[1, 0].grid(True)

		
		# to do : check
		# 4. Filtered gaze coordinates over time 

		times = np.array(timestamps) - timestamps[0]
		filtered_times = times[:]
		filtered_x_coords = []
		filtered_y_coords = []

		# 再次检查长度，确保数据没有被修改
		final_gaze_points_len = len(gaze_points)
		if final_gaze_points_len != initial_gaze_points_len:
			print(f"<WARNING> gaze_points length changed from {initial_gaze_points_len} to {final_gaze_points_len}!")

		for i in range(len(gaze_points)):
			if i in filtered_indices:
				filtered_x_coords.append(gaze_x[i])
				filtered_y_coords.append(gaze_y[i])
			else:
				# use the last valid point to replace the current invalid point
				filtered_x_coords.append(gaze_x[i-1])
				filtered_y_coords.append(gaze_y[i-1])
		
		min_length = min(len(filtered_times), len(filtered_x_coords), len(filtered_y_coords))
			
		axes[1, 1].plot(filtered_times, filtered_x_coords,'#3498db', label='X Coordinate', alpha=0.7)
		axes[1, 1].plot(filtered_times, filtered_y_coords,'#e74c3c', label='Y Coordinate', alpha=0.7)
		axes[1, 1].set_title('Filtered Gaze Coordinates Over Time')
		axes[1, 1].set_xlabel('Time (s)')
		axes[1, 1].set_ylabel('Screen Coordinates (pixels)')
		axes[1, 1].legend()
		axes[1, 1].grid(True)

		
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
			'filtered_indices': self.filtered_indices,
			'outlier_indices': self.outlier_indices,
			'filtered_gaze_points': self.filtered_gaze_points,
		}
		np.save(os.path.join(self.save_dir, f'gaze_filter_data.npy'), data)

	def get_recent_gaze_points(self, current_timestamp):
		"""
		获取时间窗口内的注视点坐标，并自动清理过期数据
		"""
		if not hasattr(self, 'gaze_points_3d_with_time') or not self.gaze_points_3d_with_time:
			return []
		
		# 使用传入的gaze时间戳计算时间窗口
		window_seconds = self.filter_params['window_seconds']
		window_start = current_timestamp - window_seconds
		
		# 清理过期数据（保留窗口内的数据）
		# self.gaze_points_3d_with_time = [
		# 	(ts, pos) for ts, pos in self.gaze_points_3d_with_time 
		# 	if ts >= window_start
		# ]
		
		# 获取当前窗口内的数据
		recent_points = []
		for timestamp, gaze_pos in self.gaze_points_3d_with_time:
			if timestamp >= window_start:
				recent_points.append(gaze_pos)
		
		# # 可选：调试信息
		# if hasattr(self, 'debug') and self.debug:
		# 	print(f"时间窗口: [{window_start:.3f}, {current_timestamp:.3f}]")
		# 	print(f"有效数据点: {len(recent_points)}/{len(self.gaze_points_with_3d_time)}")
		
		return recent_points