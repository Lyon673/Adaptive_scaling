import numpy as np

import os
import sys
import os
from scipy.ndimage import gaussian_filter1d
import params.config as config
from scipy import stats
import csv

import torch

def get_latest_data_dir(data_base_dir):
	"""获取data文件夹下最新的数据目录"""
	if not os.path.exists(data_base_dir):
		raise FileNotFoundError(f"Data directory not found: {data_base_dir}")
	
	# 获取所有子目录
	subdirs = [d for d in os.listdir(data_base_dir) 
			   if os.path.isdir(os.path.join(data_base_dir, d))]
	
	if not subdirs:
		raise FileNotFoundError(f"No subdirectories found in: {data_base_dir}")
	
	# 按名称排序（假设目录名格式为 N_data_MM-DD）
	subdirs.sort(key=lambda x: int(x.split('_')[0]) if x.split('_')[0].isdigit() else 0)
	
	latest_dir = os.path.join(data_base_dir, subdirs[-1])
	return latest_dir

def _safe_mean(arr):
	"""对空数组求均值会触发 RuntimeWarning；返回 nan 并避免警告。"""
	a = np.asarray(arr)
	if a.size == 0:
		return np.nan
	return float(np.mean(a))


def find_experiment_start(ipa_left, ipa_right, threshold=1, window=1):
	"""
	检测实验真正开始的索引
	参数:
		ipa_left, ipa_right: IPA数据
		threshold: IPA阈值，低于此值认为实验开始
		window: 连续多少个点低于阈值才确认实验开始
	"""
	min_length = min(len(ipa_left), len(ipa_right))
	
	for i in range(min_length - window):
		# 计算窗口内的平均IPA
		avg_left = np.mean(ipa_left[i:i+window])
		avg_right = np.mean(ipa_right[i:i+window])
		
		# 如果左右手的平均IPA都低于阈值，认为实验开始
		if avg_left < threshold and avg_right < threshold:
			return i
	
	return 0

# 【新增】将一维标签数组转换为连续段，用于绘制类似 vis_seq_comp 的序列图
def get_continuous_segments(labels):
	"""将一维标签数组转换为连续段: [(start, end, class_id), ...]"""
	segments = []
	if len(labels) == 0: return segments
	
	start = 0
	current_label = labels[0]
	for i in range(1, len(labels)):
		if labels[i] != current_label:
			segments.append((start, i, current_label))
			start = i
			current_label = labels[i]
			
	segments.append((start, len(labels), current_label))
	return segments


def visualize_data(data_dir=None, save_statistics=True, output_path=None):
	"""
	生成数据可视化图表并保存
	
	参数:
		data_dir: 数据目录路径，如果为None则自动使用最新数据
		save_statistics: 是否打印统计信息
	
	返回:
		str: 保存的图像路径，如果失败则返回None
	"""

	import matplotlib
	matplotlib.use('Agg')  # 设置非交互式后端，避免线程冲突
	import matplotlib.pyplot as plt
	try:
		# 如果未指定数据目录，自动获取最新数据
		if data_dir is None:
			current_file_path = os.path.abspath(__file__)
			current_dir = os.path.dirname(current_file_path)
			data_base_dir = os.path.join(current_dir, 'data')
			data_dir = get_latest_data_dir(data_base_dir)
		
		if save_statistics:
			print(f"Using data directory: {data_dir}")
		
		# 加载数据
		ipaL_data = np.load(os.path.join(data_dir, 'ipaL_data.npy'), allow_pickle=True)
		ipaR_data = np.load(os.path.join(data_dir, 'ipaR_data.npy'), allow_pickle=True)
		
		Lpsm_velocity = np.load(os.path.join(data_dir, 'Lpsm_velocity.npy'), allow_pickle=True)
		Rpsm_velocity = np.load(os.path.join(data_dir, 'Rpsm_velocity.npy'), allow_pickle=True)
		Lpsm_velocity_magnitude = np.sqrt(Lpsm_velocity[:, 0]**2 + Lpsm_velocity[:, 1]**2 + Lpsm_velocity[:, 2]**2)
		Rpsm_velocity_magnitude = np.sqrt(Rpsm_velocity[:, 0]**2 + Rpsm_velocity[:, 1]**2 + Rpsm_velocity[:, 2]**2)
		
		# 尝试加载滤波后的数据，如果不存在则使用原始数据
		try:
			Lpsm_velocity_filtered = np.load(os.path.join(data_dir, 'Lpsm_velocity_filtered.npy'), allow_pickle=True)
			Rpsm_velocity_filtered = np.load(os.path.join(data_dir, 'Rpsm_velocity_filtered.npy'), allow_pickle=True)
			ipaL_data_filtered = np.load(os.path.join(data_dir, 'ipaL_data_filtered.npy'), allow_pickle=True)
			ipaR_data_filtered = np.load(os.path.join(data_dir, 'ipaR_data_filtered.npy'), allow_pickle=True)
			use_filtered = True
			if save_statistics:
				print("  Using filtered data")
		except FileNotFoundError:
			Lpsm_velocity_filtered = None
			Rpsm_velocity_filtered = None
			ipaL_data_filtered = None
			ipaR_data_filtered = None
			use_filtered = False
			if save_statistics:
				print("  Filtered data not found, using raw data")
		
		GP_distance_data = np.load(os.path.join(data_dir, 'GP_distance_data.npy'), allow_pickle=True)
		GP_distance_array = np.array(GP_distance_data)
		left_distances = GP_distance_array[:, 0]
		right_distances = GP_distance_array[:, 1]
		
		scale_data = np.load(os.path.join(data_dir, 'scale_data.npy'), allow_pickle=True)
		scale_array = np.array(scale_data)
		left_scales = scale_array[:, 0]
		right_scales = scale_array[:, 1]
		
		psms_distance_data = np.load(os.path.join(data_dir, 'psms_distance_data.npy'), allow_pickle=True)
		
		# 加载 theta 数据
		theta_data = np.load(os.path.join(data_dir, 'theta.npy'), allow_pickle=True)
		theta_array = np.array(theta_data)
		thetaL = np.degrees(theta_array[:, 0]) if theta_array.ndim > 1 else np.degrees(theta_array)
		thetaR = np.degrees(theta_array[:, 1]) if theta_array.ndim > 1 else np.zeros_like(thetaL)
		
		# 加载 Codirectional/Antiparallel factor 数据
		try:
			Lforward_factor = np.load(os.path.join(data_dir, 'Lforward_factor.npy'), allow_pickle=True)
			Lbackward_factor = np.load(os.path.join(data_dir, 'Lbackward_factor.npy'), allow_pickle=True)
			Rforward_factor = np.load(os.path.join(data_dir, 'Rforward_factor.npy'), allow_pickle=True)
			Rbackward_factor = np.load(os.path.join(data_dir, 'Rbackward_factor.npy'), allow_pickle=True)
			has_factors = True
		except FileNotFoundError:
			Lforward_factor = None
			Lbackward_factor = None
			Rforward_factor = None
			Rbackward_factor = None
			has_factors = False
			if save_statistics:
				print("  Codirectional/Antiparallel factor data not found")
		
		# 加载阶段预测数据
		try:
			phase_labels = np.load(os.path.join(data_dir, 'phase_labels.npy'))
			has_phase = True
			if save_statistics:
				print("  Phase labels loaded")
		except FileNotFoundError:
			phase_labels = None
			has_phase = False
		phase_probs = None
		try:
			phase_probs = np.load(os.path.join(data_dir, 'phase_probs.npy'))
		except FileNotFoundError:
			pass
		
		# 找到实验开始的索引
		#start_idx = find_experiment_start(ipaL_data, ipaR_data)
		start_idx = 0
		end_idx = -1

		# 如果检测到有预实验阶段，截取数据
		if start_idx > 0:
			if save_statistics:
				print(f"  Removing {start_idx} pre-experiment data points")
			ipaL_data = ipaL_data[start_idx:end_idx]
			ipaR_data = ipaR_data[start_idx:end_idx]
			Lpsm_velocity_magnitude = Lpsm_velocity_magnitude[start_idx:end_idx]
			Rpsm_velocity_magnitude = Rpsm_velocity_magnitude[start_idx:end_idx]
			left_distances = left_distances[start_idx:end_idx]
			right_distances = right_distances[start_idx:end_idx]
			left_scales = left_scales[start_idx:end_idx]
			right_scales = right_scales[start_idx:end_idx]
			psms_distance_data = psms_distance_data[start_idx:end_idx]
			GP_distance_array = GP_distance_array[start_idx:end_idx]
			thetaL = thetaL[start_idx:end_idx]
			thetaR = thetaR[start_idx:end_idx]
			
			# 截取 Codirectional/Antiparallel factor 数据
			if has_factors:
				Lforward_factor = Lforward_factor[start_idx:end_idx]
				Lbackward_factor = Lbackward_factor[start_idx:end_idx]
				Rforward_factor = Rforward_factor[start_idx:end_idx]
				Rbackward_factor = Rbackward_factor[start_idx:end_idx]
			
			if has_phase:
				phase_labels = phase_labels[start_idx:end_idx]
			if phase_probs is not None:
				phase_probs = phase_probs[start_idx:end_idx]
			
			# 截取滤波后的数据
			if use_filtered:
				Lpsm_velocity_filtered = Lpsm_velocity_filtered[start_idx:end_idx]
				Rpsm_velocity_filtered = Rpsm_velocity_filtered[start_idx:end_idx]
				ipaL_data_filtered = ipaL_data_filtered[start_idx:end_idx]
				ipaR_data_filtered = ipaR_data_filtered[start_idx:end_idx]
		
		# 确保所有数据长度一致
		lengths = [len(ipaL_data), len(ipaR_data), len(Lpsm_velocity_magnitude), 
				   len(Rpsm_velocity_magnitude), len(left_distances), len(right_distances),
				   len(left_scales), len(right_scales), len(psms_distance_data),
				   len(thetaL), len(thetaR)]
		
		if use_filtered:
			lengths.extend([len(Lpsm_velocity_filtered), len(Rpsm_velocity_filtered),
						   len(ipaL_data_filtered), len(ipaR_data_filtered)])
		
		if has_factors:
			lengths.extend([len(Lforward_factor), len(Lbackward_factor),
						   len(Rforward_factor), len(Rbackward_factor)])
		
		if has_phase:
			lengths.append(len(phase_labels))
		
		min_length = min(lengths)
		if min_length < 1:
			print(
				f"[visualize_data] 无有效帧可绘制 (min_length={min_length})。"
				f"请检查各 .npy 是否非空且长度一致（含 phase_labels 时不能为 0 帧）。"
			)
			return None
		
		# 截取数据到相同长度
		timestamps = range(min_length)
		ipaL_data = ipaL_data[:min_length]
		ipaR_data = ipaR_data[:min_length]
		Lpsm_velocity_magnitude = Lpsm_velocity_magnitude[:min_length]
		Rpsm_velocity_magnitude = Rpsm_velocity_magnitude[:min_length]
		left_distances = left_distances[:min_length]
		right_distances = right_distances[:min_length]
		left_scales = left_scales[:min_length]
		right_scales = right_scales[:min_length]
		psms_distance_data = psms_distance_data[:min_length]
		thetaL = thetaL[:min_length]
		thetaR = thetaR[:min_length]
		
		# 截取滤波后的数据
		if use_filtered:
			Lpsm_velocity_filtered = Lpsm_velocity_filtered[:min_length]
			Rpsm_velocity_filtered = Rpsm_velocity_filtered[:min_length]
			ipaL_data_filtered = ipaL_data_filtered[:min_length]
			ipaR_data_filtered = ipaR_data_filtered[:min_length]
		
		# 截取 Codirectional/Antiparallel factor 数据
		if has_factors:
			Lforward_factor = Lforward_factor[:min_length]
			Lbackward_factor = Lbackward_factor[:min_length]
			Rforward_factor = Rforward_factor[:min_length]
			Rbackward_factor = Rbackward_factor[:min_length]
		
		if has_phase:
			phase_labels = phase_labels[:min_length]
		if phase_probs is not None:
			phase_probs = phase_probs[:min_length]
		
		# 决定使用filtered还是smoothed数据
		if use_filtered:
			# 使用滤波后的数据
			Lpsm_velocity_smoothed = Lpsm_velocity_filtered
			Rpsm_velocity_smoothed = Rpsm_velocity_filtered
			ipaL_smoothed = ipaL_data_filtered
			ipaR_smoothed = ipaR_data_filtered
		else:
			# 使用高斯平滑处理原始数据
			sigma = 5
			Lpsm_velocity_smoothed = gaussian_filter1d(Lpsm_velocity_magnitude, sigma=sigma)
			Rpsm_velocity_smoothed = gaussian_filter1d(Rpsm_velocity_magnitude, sigma=sigma)
			ipaL_smoothed = gaussian_filter1d(ipaL_data, sigma=sigma)
			ipaR_smoothed = gaussian_filter1d(ipaR_data, sigma=sigma)
		
		# 计算平均IPA
		ipa_average = (ipaL_data + ipaR_data) / 2
		ipa_average_smoothed = (ipaL_smoothed + ipaR_smoothed) / 2
		
		# 定义配色方案
		color_left = '#3498db'
		color_right = '#e74c3c'
		
		# 创建可视化 - 使用 GridSpec 支持底部全宽的阶段条
		from matplotlib.gridspec import GridSpec
		from matplotlib.colors import ListedColormap, BoundaryNorm
		
		n_data_rows = 4
		if has_phase:
			fig = plt.figure(figsize=(20, 24))
			gs = GridSpec(n_data_rows + 1, 2, figure=fig,
						 height_ratios=[1]*n_data_rows + [0.2], hspace=0.35)
		else:
			fig = plt.figure(figsize=(18, 22))
			gs = GridSpec(n_data_rows, 2, figure=fig, hspace=0.35)
		
		axs = np.empty((n_data_rows, 2), dtype=object)
		for r in range(n_data_rows):
			for c in range(2):
				axs[r, c] = fig.add_subplot(gs[r, c])
		
		fig.patch.set_facecolor('white')
		
		filter_method = 'Real-Time Filtered' if use_filtered else 'Gaussian Smoothed'
		
		axs[0, 0].plot(timestamps, psms_distance_data, color='#f39c12', alpha=0.8, 
					   linewidth=2.5)
		axs[0, 0].fill_between(timestamps, psms_distance_data, alpha=0.15, color='#f39c12')
		axs[0, 0].set_title('Distance Between Left and Right PSM', 
							fontsize=17, fontweight='bold', pad=15)
		axs[0, 0].set_xlabel('Frame Index', fontsize=15)
		axs[0, 0].set_ylabel('Distance', fontsize=15)
		axs[0, 0].grid(True, alpha=0.2, linestyle='--')
		axs[0, 0].set_facecolor('#f8f9fa')

		
		axs[0, 1].plot(timestamps, left_distances, color=color_left, alpha=0.8, 
					   linewidth=2.5, label='Left PSM')
		axs[0, 1].plot(timestamps, right_distances, color=color_right, alpha=0.8, 
					   linewidth=2.5, label='Right PSM')
		axs[0, 1].fill_between(timestamps, left_distances, alpha=0.15, color=color_left)
		axs[0, 1].fill_between(timestamps, right_distances, alpha=0.15, color=color_right)
		axs[0, 1].set_title('3D Distances Between Gaze Point and PSMs', 
							fontsize=17, fontweight='bold', pad=15)
		axs[0, 1].set_xlabel('Frame Index', fontsize=15)
		axs[0, 1].set_ylabel('Distance', fontsize=15)
		axs[0, 1].legend(loc='upper right', frameon=True, fancybox=True, shadow=True, fontsize=13)
		axs[0, 1].grid(True, alpha=0.2, linestyle='--')
		axs[0, 1].set_facecolor('#f8f9fa')
		

		theta_factorL = 2*np.fabs(0.5- 1 / (1 + np.exp(-4.45 * (np.deg2rad(thetaL) - np.pi/2)**3))) + 0.2
		theta_factorR = 2*np.fabs(0.5- 1 / (1 + np.exp(-4.45 * (np.deg2rad(thetaR) - np.pi/2)**3))) + 0.2

		axs[1, 0].plot(timestamps, theta_factorL, color=color_left, alpha=0.8, 
					   linewidth=2.5, label='Left PSM')
		axs[1, 0].plot(timestamps, theta_factorR, color=color_right, alpha=0.8, 
					   linewidth=2.5, label='Right PSM')
		axs[1, 0].set_title('Movement Direction Theta Factor', fontsize=17, fontweight='bold', pad=15)
		axs[1, 0].set_xlabel('Frame Index', fontsize=15)
		axs[1, 0].set_ylabel('Theta Factor', fontsize=15)
		axs[1, 0].legend(loc='upper right', frameon=True, fancybox=True, shadow=True, fontsize=13)
		axs[1, 0].grid(True, alpha=0.2, linestyle='--')
		axs[1, 0].set_facecolor('#f8f9fa')

		# 4. 速度数据 - 显示原始和滤波后的数据
		Lpsm_velocity_smoothed = np.where(Lpsm_velocity_smoothed < 0, 0, Lpsm_velocity_smoothed)
		Rpsm_velocity_smoothed = np.where(Rpsm_velocity_smoothed < 0, 0, Rpsm_velocity_smoothed)
		axs[1, 1].plot(timestamps, Lpsm_velocity_smoothed, color=color_left, 
					   alpha=0.9, linewidth=2.5, label='Left PSM')
		axs[1, 1].plot(timestamps, Rpsm_velocity_smoothed, color=color_right, 
					   alpha=0.9, linewidth=2.5, label='Right PSM')
		axs[1, 1].set_title(f'PSMs Linear Speed', 
							fontsize=17, fontweight='bold', pad=15)
		axs[1, 1].set_xlabel('Frame Index', fontsize=15)
		axs[1, 1].set_ylabel('Velocity', fontsize=15)
		axs[1, 1].legend(loc='upper right', frameon=True, fancybox=True, shadow=True, fontsize=13)
		axs[1, 1].grid(True, alpha=0.2, linestyle='--')
		axs[1, 1].set_facecolor('#f8f9fa')
		
		# 5. [2,0] 左手 Codirectional/Antiparallel Factors
		if has_factors:
			axs[2, 0].plot(timestamps, Lforward_factor, color='#2ecc71', alpha=0.9, 
						   linewidth=2.5, label= 'Gaze Carrier Factor')
			axs[2, 0].plot(timestamps, Lbackward_factor, color='#e67e22', alpha=0.9, 
						   linewidth=2.5, label='Speed Modulation Factor')
			axs[2, 0].fill_between(timestamps, Lforward_factor, alpha=0.1, color='#2ecc71')
			#axs[2, 0].fill_between(timestamps, Lbackward_factor, alpha=0.1, color='#e67e22')

		axs[2, 0].set_title('Left PSM component Factors', 
							fontsize=17, fontweight='bold', pad=15)
		axs[2, 0].set_xlabel('Frame Index', fontsize=15)
		axs[2, 0].set_ylabel('Factor Value', fontsize=15)
		axs[2, 0].legend(loc='upper right', frameon=True, fancybox=True, shadow=True, fontsize=13)
		axs[2, 0].grid(True, alpha=0.2, linestyle='--')
		axs[2, 0].set_facecolor('#f8f9fa')
		
		# 6. [2,1] 右手 Codirectional/Antiparallel Factors
		if has_factors:
			axs[2, 1].plot(timestamps, Rforward_factor, color='#2ecc71', alpha=0.9, 
						   linewidth=2.5, label='Gaze Carrier Factor')
			axs[2, 1].plot(timestamps, Rbackward_factor, color='#e67e22', alpha=0.9, 
						   linewidth=2.5, label='Speed Modulation Factor')
			axs[2, 1].fill_between(timestamps, Rforward_factor, alpha=0.1, color='#2ecc71')
			#axs[2, 1].fill_between(timestamps, Rbackward_factor, alpha=0.1, color='#e67e22')
			
		axs[2, 1].set_title('Right PSM component Factors', 
							fontsize=17, fontweight='bold', pad=15)
		axs[2, 1].set_xlabel('Frame Index', fontsize=15)
		axs[2, 1].set_ylabel('Factor Value', fontsize=15)
		axs[2, 1].legend(loc='upper right', frameon=True, fancybox=True, shadow=True, fontsize=13)
		axs[2, 1].grid(True, alpha=0.2, linestyle='--')
		axs[2, 1].set_facecolor('#f8f9fa')

		# 7. [3,0] Scale数据 Left
		axs[3, 0].plot(timestamps, left_scales*0.01, color=color_left, alpha=0.8, 
					   linewidth=2.5)
		axs[3, 0].fill_between(timestamps, left_scales*0.01, alpha=0.1, color=color_left)
		axs[3, 0].set_ylim(0.12, 0.25)
		axs[3, 0].set_title('Left Adaptive Scaling Factor', fontsize=17, fontweight='bold', pad=15)
		axs[3, 0].set_xlabel('Frame Index', fontsize=15)
		axs[3, 0].set_ylabel('Scaling Factor', fontsize=15)
		axs[3, 0].grid(True, alpha=0.2, linestyle='--')
		axs[3, 0].set_facecolor('#f8f9fa')
		
		# 8. [3,1] Scale数据 Right
		axs[3, 1].plot(timestamps, right_scales*0.01, color=color_right, alpha=0.8, 
					   linewidth=2.5)
		axs[3, 1].fill_between(timestamps, right_scales*0.01, alpha=0.1, color=color_right)
		axs[3, 1].set_ylim(0.12, 0.25)
		axs[3, 1].set_title('Right Adaptive Scaling Factor', fontsize=17, fontweight='bold', pad=15)
		axs[3, 1].set_xlabel('Frame Index', fontsize=15)
		axs[3, 1].set_ylabel('Scaling Factor', fontsize=15)
		axs[3, 1].grid(True, alpha=0.2, linestyle='--')
		axs[3, 1].set_facecolor('#f8f9fa')
		
		
		# ── 阶段预测条状图 & 概率曲线 & 阶段转换虚线 ─────────────────────
		if has_phase and phase_labels is not None:
			PHASE_NAMES = [ 'P0', 'P1', 'P2', 'P3', 'P4', 'P5', 'P6']
			phase_colors = ['#FF6B6B', '#EDC58C', '#45B7D1', '#96CEB4', '#FFEAA7', '#DDA15E', '#B16DCF']
			
			T = len(phase_labels)
			t_frames = np.arange(T)
			T_plot = max(T, 1)

			ax_phase = fig.add_subplot(gs[n_data_rows, :])
			
			if T > 0:
				# 1. 将连续标签转换为分段，绘制堆叠条形图 (Gantt style)
				segments = get_continuous_segments(phase_labels)
				
				for start, end, phase in segments:
					phase_idx = int(phase)
					# 兼容如果存在 -1 等情况
					color = phase_colors[phase_idx] if 0 <= phase_idx < len(phase_colors) else '#E0E0E0'
					
					# 使用 barh 画出色带
					ax_phase.barh(0, end - start, left=start, height=0.2, 
								  color=color, edgecolor='none', align='center', alpha=1.0)
					

				ax_phase.set_xlim(0, T_plot)
				ax_phase.set_yticks([]) 
				#ax_phase.set_xlabel('Frame Index', fontsize=15)
				ax_phase.set_title('Predicted Surgical Phase Sequence',
								   fontsize=17, fontweight='bold', pad=15)
				ax_phase.set_facecolor('#f8f9fa')
				
				# 去掉上、右、左的外边框，让序列图看起来更干净
				ax_phase.spines['top'].set_visible(False)
				ax_phase.spines['right'].set_visible(False)
				ax_phase.spines['left'].set_visible(False)


				# 绘制图例
				from matplotlib.patches import Patch
				legend_labels = ['Invalid Phase'] + [lbl.replace('\n', ' ') for lbl in PHASE_NAMES]
				legend_colors = ['#E0E0E0'] + phase_colors
				legend_handles = [Patch(facecolor=c, label=l) for c, l in zip(legend_colors, legend_labels)]
				
				# 统一图例放置在底部
				ax_phase.legend(handles=legend_handles, loc='upper center',
							   bbox_to_anchor=(0.5, -0.2), ncol=8, fontsize=15, frameon=False)
			else:
				ax_phase.text(0.5, 0.5, 'No phase labels', ha='center', va='center',
							  transform=ax_phase.transAxes, fontsize=12)
				ax_phase.set_axis_off()

			# 4. 在所有上半部分的子图上画出阶段转换的虚线
			transitions = []
			for i in range(1, len(phase_labels)):
				if phase_labels[i] >= 0 and phase_labels[i] != phase_labels[i-1]:
					transitions.append((i, int(phase_labels[i])))

			for r in range(n_data_rows):
				for c in range(2):
					for tf, new_phase in transitions:
						clr = phase_colors[new_phase] if 0 <= new_phase < len(PHASE_NAMES) else '#7f8c8d'
						axs[r, c].axvline(x=tf, color=clr, linestyle='--',
										  alpha=0.7, linewidth=3.0)
		
		# 保存图表
		if not has_phase:
			plt.tight_layout()
		else:
			# 为了底部的序列图例留出空间
			plt.subplots_adjust(bottom=0.05)
			
		output_path = os.path.join(output_path, 'Feature_visualization.png')
		plt.savefig(output_path, dpi=300, bbox_inches='tight')
		plt.close()
		
		return output_path
		
	except FileNotFoundError as e:
		print(f"Error loading data: {e}")
		return None
	except Exception as e:
		print(f"Error generating visualization: {e}")
		import traceback
		traceback.print_exc()
		return None


# 如果直接运行此脚本，使用最新数据生成可视化
if __name__ == '__main__':
	print("Generating visualization...")
	file_path = os.path.abspath(__file__)
	current_dir = os.path.dirname(file_path)
	output_path = os.path.join(current_dir, 'Essay_image_results')
	visualize_data(data_dir='data/183_data_04-09', output_path=output_path)
