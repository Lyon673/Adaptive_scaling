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

def visualize_data(data_dir=None, save_statistics=True):
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

		# for i in range(len(left_scales)):
		# 	if left_scales[i] > 2:
		# 		left_scales[i] = 2
		# for i in range(len(right_scales)):
		# 	if right_scales[i] > 2:
		# 		right_scales[i] = 2

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
		color_left_light = '#5dade2'
		color_right_light = '#ec7063'
		
		# 创建可视化 - 使用 GridSpec 支持底部全宽的阶段条
		from matplotlib.gridspec import GridSpec
		from matplotlib.colors import ListedColormap, BoundaryNorm
		
		n_data_rows = 4
		if has_phase:
			fig = plt.figure(figsize=(16, 28))
			gs = GridSpec(n_data_rows + 1, 2, figure=fig,
						 height_ratios=[1]*n_data_rows + [1.0], hspace=0.35)
		else:
			fig = plt.figure(figsize=(16, 22))
			gs = GridSpec(n_data_rows, 2, figure=fig, hspace=0.35)
		
		axs = np.empty((n_data_rows, 2), dtype=object)
		for r in range(n_data_rows):
			for c in range(2):
				axs[r, c] = fig.add_subplot(gs[r, c])
		
		fig.patch.set_facecolor('white')
		
		filter_method = 'Real-Time Filtered' if use_filtered else 'Gaussian Smoothed'
		
		# # 1. IPA数据（左右手）- 注释掉
		# axs[0, 0].scatter(timestamps, ipaL_data, c=color_left, alpha=0.3, s=10, 
		# 				  edgecolors='none', label='Left Raw')
		# axs[0, 0].scatter(timestamps, ipaR_data, c=color_right, alpha=0.3, s=10, 
		# 				  edgecolors='none', label='Right Raw')
		# axs[0, 0].plot(timestamps, ipaL_smoothed, color=color_left, alpha=0.9, 
		# 			   linewidth=2.5, label='Left Filtered')
		# axs[0, 0].plot(timestamps, ipaR_smoothed, color=color_right, alpha=0.9, 
		# 			   linewidth=2.5, label='Right Filtered')
		# axs[0, 0].set_title(f'IPA Data ({filter_method})', 
		# 					fontsize=13, fontweight='bold', pad=15)
		# axs[0, 0].set_xlabel('Frame Index', fontsize=11)
		# axs[0, 0].set_ylabel('IPA Values', fontsize=11)
		# axs[0, 0].legend(loc='best', frameon=True, fancybox=True, shadow=True, fontsize=9)
		# axs[0, 0].grid(True, alpha=0.2, linestyle='--')
		# axs[0, 0].set_facecolor('#f8f9fa')
		
		axs[0, 0].plot(timestamps, psms_distance_data, color='#f39c12', alpha=0.8, 
					   linewidth=2.5, label='Distance Between PSMs')
		axs[0, 0].fill_between(timestamps, psms_distance_data, alpha=0.15, color='#f39c12')
		axs[0, 0].set_title('Distance Between Left and Right PSM', 
							fontsize=13, fontweight='bold', pad=15)
		axs[0, 0].set_xlabel('Frame Index', fontsize=11)
		axs[0, 0].set_ylabel('Distance (meters)', fontsize=11)
		axs[0, 0].legend(loc='best', frameon=True, fancybox=True, shadow=True)
		axs[0, 0].grid(True, alpha=0.2, linestyle='--')
		axs[0, 0].set_facecolor('#f8f9fa')

		
		axs[0, 1].plot(timestamps, left_distances, color=color_left, alpha=0.8, 
					   linewidth=2.5, label='Left Hand Distance')
		axs[0, 1].plot(timestamps, right_distances, color=color_right, alpha=0.8, 
					   linewidth=2.5, label='Right Hand Distance')
		axs[0, 1].fill_between(timestamps, left_distances, alpha=0.15, color=color_left)
		axs[0, 1].fill_between(timestamps, right_distances, alpha=0.15, color=color_right)
		axs[0, 1].set_title('3D Distance Between Hands and Gaze Point', 
							fontsize=13, fontweight='bold', pad=15)
		axs[0, 1].set_xlabel('Frame Index', fontsize=11)
		axs[0, 1].set_ylabel('Distance (meters)', fontsize=11)
		axs[0, 1].legend(loc='best', frameon=True, fancybox=True, shadow=True)
		axs[0, 1].grid(True, alpha=0.2, linestyle='--')
		axs[0, 1].set_facecolor('#f8f9fa')
		
		avg_dist_left = _safe_mean(left_distances)
		avg_dist_right = _safe_mean(right_distances)
		if not np.isnan(avg_dist_left):
			axs[0, 1].axhline(y=avg_dist_left, color=color_left, linestyle=':', alpha=0.6, linewidth=2)
		if not np.isnan(avg_dist_right):
			axs[0, 1].axhline(y=avg_dist_right, color=color_right, linestyle=':', alpha=0.6, linewidth=2)

		security_factor = 1-np.exp(-1000**2 * psms_distance_data**config.init_params['B_safety'])
		axs[1, 0].plot(timestamps, security_factor, color='#9b59b6', alpha=0.9, 
					   linewidth=2.5, label=f'Security Factor')
		axs[1, 0].set_title(f'Security Factor', 
							fontsize=13, fontweight='bold', pad=15)
		axs[1, 0].set_xlabel('Frame Index', fontsize=11)
		axs[1, 0].set_ylabel('Security Factor', fontsize=11)
		axs[1, 0].legend(loc='best', frameon=True, fancybox=True, shadow=True)
		axs[1, 0].grid(True, alpha=0.2, linestyle='--')
		axs[1, 0].set_facecolor('#f8f9fa')
		
		# 4. 速度数据 - 显示原始和滤波后的数据
		axs[1, 1].plot(timestamps, Lpsm_velocity_magnitude, color=color_left_light, 
					   alpha=0.3, linewidth=1.5, label='Left Raw')
		axs[1, 1].plot(timestamps, Rpsm_velocity_magnitude, color=color_right_light,
					   alpha=0.3, linewidth=1.5, label='Right Raw')
		Lpsm_velocity_smoothed = np.where(Lpsm_velocity_smoothed < 0, 0, Lpsm_velocity_smoothed)
		Rpsm_velocity_smoothed = np.where(Rpsm_velocity_smoothed < 0, 0, Rpsm_velocity_smoothed)
		axs[1, 1].plot(timestamps, Lpsm_velocity_smoothed, color=color_left, 
					   alpha=0.9, linewidth=2.5, label='Left Filtered')
		axs[1, 1].plot(timestamps, Rpsm_velocity_smoothed, color=color_right, 
					   alpha=0.9, linewidth=2.5, label='Right Filtered')
		axs[1, 1].set_title(f'PSM Linear Velocity ({filter_method})', 
							fontsize=13, fontweight='bold', pad=15)
		# axs[1, 1].plot(timestamps, Lmtm_velocity_magnitude, color=color_left_light, 
		# 			   alpha=0.3, linewidth=1.5, label='Left Raw')
		# axs[1, 1].plot(timestamps, Rmtm_velocity_magnitude, color=color_right_light,
		# 			   alpha=0.3, linewidth=1.5, label='Right Raw')
		# Lmtm_velocity_smoothed = np.where(Lmtm_velocity_smoothed < 0, 0, Lmtm_velocity_smoothed)
		# Rmtm_velocity_smoothed = np.where(Rmtm_velocity_smoothed < 0, 0, Rmtm_velocity_smoothed)
		# axs[1, 1].plot(timestamps, Lmtm_velocity_smoothed, color=color_left, 
		# 			   alpha=0.9, linewidth=2.5, label='Left Filtered')
		# axs[1, 1].plot(timestamps, Rmtm_velocity_smoothed, color=color_right, 
		# 			   alpha=0.9, linewidth=2.5, label='Right Filtered')
		# axs[1, 1].set_title(f'MTM Linear Velocity ({filter_method})', 
		# 					fontsize=13, fontweight='bold', pad=15)
		axs[1, 1].set_xlabel('Frame Index', fontsize=11)
		axs[1, 1].set_ylabel('Velocity (m/s)', fontsize=11)
		axs[1, 1].legend(loc='best', frameon=True, fancybox=True, shadow=True, fontsize=9)
		axs[1, 1].grid(True, alpha=0.2, linestyle='--')
		axs[1, 1].set_facecolor('#f8f9fa')
		
		avg_vel_left = _safe_mean(Lpsm_velocity_smoothed)
		avg_vel_right = _safe_mean(Rpsm_velocity_smoothed)
		if not np.isnan(avg_vel_left):
			axs[1, 1].axhline(y=avg_vel_left, color=color_left, linestyle=':', alpha=0.6, linewidth=2)
		if not np.isnan(avg_vel_right):
			axs[1, 1].axhline(y=avg_vel_right, color=color_right, linestyle=':', alpha=0.6, linewidth=2)
		
		# 5. [2,0] 左手 Codirectional/Antiparallel Factors
		if has_factors:
			axs[2, 0].plot(timestamps, Lforward_factor, color='#2ecc71', alpha=0.9, 
						   linewidth=2.5, label= 'gaze Factor')
			axs[2, 0].plot(timestamps, Lbackward_factor, color='#e67e22', alpha=0.9, 
						   linewidth=2.5, label='directional v Factor')
			axs[2, 0].fill_between(timestamps, Lforward_factor, alpha=0.1, color='#2ecc71')
			axs[2, 0].fill_between(timestamps, Lbackward_factor, alpha=0.1, color='#e67e22')
			
			avg_lforward = _safe_mean(Lforward_factor)
			avg_lbackward = _safe_mean(Lbackward_factor)
			if not np.isnan(avg_lforward):
				axs[2, 0].axhline(y=avg_lforward, color='#2ecc71', linestyle=':', alpha=0.6, linewidth=2)
			if not np.isnan(avg_lbackward):
				axs[2, 0].axhline(y=avg_lbackward, color='#e67e22', linestyle=':', alpha=0.6, linewidth=2)
			
			axs[2, 0].text(0.02, 0.98, f'dgp Avg: {avg_lforward:.3f}\n theta-v Avg: {avg_lbackward:.3f}',
						   transform=axs[2, 0].transAxes, verticalalignment='top', fontsize=10,
						   bbox=dict(boxstyle='round', facecolor='#ecf0f1', alpha=0.8, edgecolor='#95a5a6'))
		else:
			axs[2, 0].text(0.5, 0.5, 'No factor data available', 
						   transform=axs[2, 0].transAxes, ha='center', va='center', fontsize=12)
		
		axs[2, 0].set_title('Left PSM component Factors', 
							fontsize=13, fontweight='bold', pad=15)
		axs[2, 0].set_xlabel('Frame Index', fontsize=11)
		axs[2, 0].set_ylabel('Factor Value', fontsize=11)
		axs[2, 0].legend(loc='best', frameon=True, fancybox=True, shadow=True)
		axs[2, 0].grid(True, alpha=0.2, linestyle='--')
		axs[2, 0].set_facecolor('#f8f9fa')
		
		# 6. [2,1] 右手 Codirectional/Antiparallel Factors
		if has_factors:
			axs[2, 1].plot(timestamps, Rforward_factor, color='#2ecc71', alpha=0.9, 
						   linewidth=2.5, label='gaze Factor')
			axs[2, 1].plot(timestamps, Rbackward_factor, color='#e67e22', alpha=0.9, 
						   linewidth=2.5, label='directional v Factor')
			axs[2, 1].fill_between(timestamps, Rforward_factor, alpha=0.1, color='#2ecc71')
			axs[2, 1].fill_between(timestamps, Rbackward_factor, alpha=0.1, color='#e67e22')
			
			avg_rforward = _safe_mean(Rforward_factor)
			avg_rbackward = _safe_mean(Rbackward_factor)
			if not np.isnan(avg_rforward):
				axs[2, 1].axhline(y=avg_rforward, color='#2ecc71', linestyle=':', alpha=0.6, linewidth=2)
			if not np.isnan(avg_rbackward):
				axs[2, 1].axhline(y=avg_rbackward, color='#e67e22', linestyle=':', alpha=0.6, linewidth=2)
			
			axs[2, 1].text(0.02, 0.98, f'dgp Avg: {avg_rforward:.3f}\n theta-v Avg: {avg_rbackward:.3f}',
						   transform=axs[2, 1].transAxes, verticalalignment='top', fontsize=10,
						   bbox=dict(boxstyle='round', facecolor='#ecf0f1', alpha=0.8, edgecolor='#95a5a6'))
		else:
			axs[2, 1].text(0.5, 0.5, 'No factor data available', 
						   transform=axs[2, 1].transAxes, ha='center', va='center', fontsize=12)
		
		axs[2, 1].set_title('Right PSM component Factors', 
							fontsize=13, fontweight='bold', pad=15)
		axs[2, 1].set_xlabel('Frame Index', fontsize=11)
		axs[2, 1].set_ylabel('Factor Value', fontsize=11)
		axs[2, 1].legend(loc='best', frameon=True, fancybox=True, shadow=True)
		axs[2, 1].grid(True, alpha=0.2, linestyle='--')
		axs[2, 1].set_facecolor('#f8f9fa')
		
		# 7. [3,0] Theta 数据 - 左右手
		# axs[3, 0].plot(timestamps, thetaL, color=color_left, alpha=0.8, 
		# 			   linewidth=2.5, label='Left PSM Theta')
		# axs[3, 0].plot(timestamps, thetaR, color=color_right, alpha=0.8, 
		# 			   linewidth=2.5, label='Right PSM Theta')
		# axs[3, 0].fill_between(timestamps, thetaL, alpha=0.1, color=color_left)
		# axs[3, 0].fill_between(timestamps, thetaR, alpha=0.1, color=color_right)
		# axs[3, 0].set_title('Angle Between Velocity and Gaze Direction (Both PSMs)', 
		# 					fontsize=13, fontweight='bold', pad=15)
		# axs[3, 0].set_xlabel('Frame Index', fontsize=11)
		# axs[3, 0].set_ylabel('Theta (degrees)', fontsize=11)
		# axs[3, 0].legend(loc='best', frameon=True, fancybox=True, shadow=True)
		# axs[3, 0].grid(True, alpha=0.2, linestyle='--')
		# axs[3, 0].set_facecolor('#f8f9fa')
		# avg_thetaL = np.mean(thetaL)
		# avg_thetaR = np.mean(thetaR)
		# axs[3, 0].axhline(y=avg_thetaL, color=color_left, linestyle=':', alpha=0.6, linewidth=2)
		# axs[3, 0].axhline(y=avg_thetaR, color=color_right, linestyle=':', alpha=0.6, linewidth=2)
		# axs[3, 0].text(0.02, 0.98, f'Left Avg: {avg_thetaL:.2f}°\nRight Avg: {avg_thetaR:.2f}°',
		# 			   transform=axs[3, 0].transAxes, verticalalignment='top', fontsize=10,
		# 			   bbox=dict(boxstyle='round', facecolor='#ecf0f1', alpha=0.8, edgecolor='#95a5a6'))

		theta_factorL = 1 - 1 / (1 + np.exp(-2 * (np.deg2rad(thetaL) - np.pi/2)**3))
		theta_factorR = 1 - 1 / (1 + np.exp(-2 * (np.deg2rad(thetaR) - np.pi/2)**3))
		axs[3, 0].plot(timestamps, theta_factorL, color=color_left, alpha=0.8, 
					   linewidth=2.5, label='Left PSM Theta Factor')
		axs[3, 0].plot(timestamps, theta_factorR, color=color_right, alpha=0.8, 
					   linewidth=2.5, label='Right PSM Theta Factor')
		axs[3, 0].fill_between(timestamps, theta_factorL, alpha=0.1, color=color_left)
		axs[3, 0].fill_between(timestamps, theta_factorR, alpha=0.1, color=color_right)
		axs[3, 0].set_title('Theta Factor', 
							fontsize=13, fontweight='bold', pad=15)
		axs[3, 0].set_xlabel('Frame Index', fontsize=11)
		axs[3, 0].set_ylabel('Theta Factor', fontsize=11)
		axs[3, 0].legend(loc='best', frameon=True, fancybox=True, shadow=True)
		axs[3, 0].grid(True, alpha=0.2, linestyle='--')
		axs[3, 0].set_facecolor('#f8f9fa')

		avg_thetaL = _safe_mean(theta_factorL)
		avg_thetaR = _safe_mean(theta_factorR)
		if not np.isnan(avg_thetaL):
			axs[3, 0].axhline(y=avg_thetaL, color=color_left, linestyle=':', alpha=0.6, linewidth=2)
		if not np.isnan(avg_thetaR):
			axs[3, 0].axhline(y=avg_thetaR, color=color_right, linestyle=':', alpha=0.6, linewidth=2)
		_theta_txt = (
			f"Left Avg: {avg_thetaL:.2f}\nRight Avg: {avg_thetaR:.2f}"
			if not (np.isnan(avg_thetaL) and np.isnan(avg_thetaR))
			else "Avg: N/A (empty)"
		)
		axs[3, 0].text(0.02, 0.98, _theta_txt,
					   transform=axs[3, 0].transAxes, verticalalignment='top', fontsize=10,
					   bbox=dict(boxstyle='round', facecolor='#ecf0f1', alpha=0.8, edgecolor='#95a5a6'))
		
		# 8. [3,1] Scale数据
		axs[3, 1].plot(timestamps, left_scales*0.1, color=color_left, alpha=0.8, 
					   linewidth=2.5, label='Left Hand Scale')
		axs[3, 1].plot(timestamps, right_scales*0.1, color=color_right, alpha=0.8, 
					   linewidth=2.5, label='Right Hand Scale')
		axs[3, 1].set_title('Adaptive Scaling Factors', fontsize=13, fontweight='bold', pad=15)
		axs[3, 1].set_xlabel('Frame Index', fontsize=11)
		axs[3, 1].set_ylabel('Scale Factor', fontsize=11)
		axs[3, 1].legend(loc='best', frameon=True, fancybox=True, shadow=True)
		axs[3, 1].grid(True, alpha=0.2, linestyle='--')
		axs[3, 1].set_facecolor('#f8f9fa')
		
		avg_scale_left = _safe_mean(left_scales)
		avg_scale_right = _safe_mean(right_scales)
		_scale_txt = (
			f'Left Avg: {avg_scale_left:.3f}\nRight Avg: {avg_scale_right:.3f}'
			if not (np.isnan(avg_scale_left) and np.isnan(avg_scale_right))
			else 'Avg: N/A (empty)'
		)
		axs[3, 1].text(0.02, 0.98, _scale_txt,
					   transform=axs[3, 1].transAxes, verticalalignment='top', fontsize=10,
					   bbox=dict(boxstyle='round', facecolor='#ecf0f1', alpha=0.8, edgecolor='#95a5a6'))
		
		# ── 阶段预测条状图 & 概率曲线 & 阶段转换虚线 ─────────────────────
		if has_phase and phase_labels is not None:
			PHASE_NAMES = [
				'P0 Right Move', 'P1 Pick Needle', 'P2 Right Move2',
				'P3 Pass Needle', 'P4 Left Move', 'P5 Left Pick', 'P6 Pull Thread',
			]
			phase_colors = [
				'#95a5a6',  # warmup (-1)
				'#1abc9c', '#3498db', '#9b59b6',
				'#e67e22', '#e74c3c', '#2ecc71', '#f1c40f',
			]
			cmap = ListedColormap(phase_colors)
			norm = BoundaryNorm(np.arange(-0.5, len(phase_colors)), len(phase_colors))

			shifted = phase_labels.astype(int) + 1  # -1→0(warmup), 0→1, …, 6→7
			T = len(shifted)
			t_frames = np.arange(T)
			# T==0 时 extent 左右同为 0 会触发 matplotlib 奇异变换警告
			T_plot = max(T, 1)

			ax_phase = fig.add_subplot(gs[n_data_rows, :])
			if T > 0:
				ax_phase.imshow(
					shifted[np.newaxis, :], aspect='auto', cmap=cmap, norm=norm,
					extent=[0, T_plot, 0, 1], interpolation='nearest',
				)

				if phase_probs is not None and len(phase_probs) == T:
					ax_prob = ax_phase.twinx()
					num_classes = phase_probs.shape[1]
					for ci in range(num_classes):
						ax_prob.plot(t_frames, phase_probs[:, ci],
									 color='black', linewidth=1.0,
									 alpha=0.6)
					ax_prob.set_ylim(-0.05, 1.05)
					ax_prob.set_ylabel('Probability', fontsize=11)
					ax_prob.tick_params(labelsize=9)

				ax_phase.set_yticks([])
				ax_phase.set_xlim(0, T_plot)
				ax_phase.set_xlabel('Frame Index', fontsize=11)
				ax_phase.set_title('Predicted Surgical Phase & Probability',
								   fontsize=13, fontweight='bold', pad=10)
				ax_phase.set_facecolor('#f8f9fa')

				from matplotlib.patches import Patch
				legend_labels = ['Warmup'] + PHASE_NAMES
				legend_handles = [Patch(facecolor=c, label=l)
								  for c, l in zip(phase_colors, legend_labels)]
				ax_phase.legend(handles=legend_handles, loc='upper center',
							   bbox_to_anchor=(0.5, -0.15), ncol=4, fontsize=9,
							   frameon=True, fancybox=True, shadow=True)
			else:
				ax_phase.text(0.5, 0.5, 'No phase labels', ha='center', va='center',
							  transform=ax_phase.transAxes, fontsize=12)
				ax_phase.set_axis_off()

			# 找到阶段转换帧（包含 warmup→P0）
			transitions = []
			for i in range(1, len(phase_labels)):
				if phase_labels[i] >= 0 and phase_labels[i] != phase_labels[i-1]:
					transitions.append((i, int(phase_labels[i])))

			# 在所有数据子图上画阶段转换虚线（颜色与转换后阶段一致）
			for r in range(n_data_rows):
				for c in range(2):
					for tf, new_phase in transitions:
						clr = phase_colors[new_phase + 1] if 0 <= new_phase < len(PHASE_NAMES) else '#7f8c8d'
						axs[r, c].axvline(x=tf, color=clr, linestyle='--',
										  alpha=0.7, linewidth=1.0)
		
		# 保存图表
		if not has_phase:
			plt.tight_layout()
		output_path = os.path.join(data_dir, 'visualization_results.png')
		plt.savefig(output_path, dpi=300, bbox_inches='tight')
		plt.close()
		
		# 打印统计信息
		# if save_statistics:
		# 	print(f"\nVisualization saved to: {output_path}")
		# 	print("\n" + "="*50)
		# 	print("Statistics Summary:")
		# 	print("="*50)
		# 	print(f"\nData Points: {min_length}")
		# 	print(f"\nIPA Data:")
		# 	print(f"  Left  - Mean: {np.mean(ipaL_data):.4f}, Std: {np.std(ipaL_data):.4f}")
		# 	print(f"  Right - Mean: {np.mean(ipaR_data):.4f}, Std: {np.std(ipaR_data):.4f}")
		# 	print(f"  Average - Mean: {np.mean(ipa_average):.4f}, Std: {np.std(ipa_average):.4f}")
		
		# 	print(f"\nVelocity Data (m/s):")
		# 	print(f"  Left  - Mean: {avg_vel_left:.4f}, Max: {np.max(Lpsm_velocity_magnitude):.4f}")
		# 	print(f"  Right - Mean: {avg_vel_right:.4f}, Max: {np.max(Rpsm_velocity_magnitude):.4f}")
		
		# 	print(f"\nDistance Data (meters):")
		# 	print(f"  Left  - Mean: {avg_dist_left:.4f}, Min: {np.min(left_distances):.4f}, Max: {np.max(left_distances):.4f}")
		# 	print(f"  Right - Mean: {avg_dist_right:.4f}, Min: {np.min(right_distances):.4f}, Max: {np.max(right_distances):.4f}")
		
		# 	print(f"\nScale Data:")
		# 	print(f"  Left  - Mean: {avg_scale_left:.3f}, Min: {np.min(left_scales):.3f}, Max: {np.max(left_scales):.3f}")
		# 	print(f"  Right - Mean: {avg_scale_right:.3f}, Min: {np.min(right_scales):.3f}, Max: {np.max(right_scales):.3f}")
		
		# 	print(f"\nPSMs Distance:")
		# 	print(f"  Mean: {avg_psms_dist:.4f}m, Min: {min_psms_dist:.4f}m, Max: {max_psms_dist:.4f}m")
		# 	print("="*50)
		
		return output_path
		
	except FileNotFoundError as e:
		print(f"Error loading data: {e}")
		return None
	except Exception as e:
		print(f"Error generating visualization: {e}")
		import traceback
		traceback.print_exc()
		return None

def visualize_output(data_dir=None, save_statistics=False):
	"""
	生成数据可视化图表并保存
	
	参数:
		data_dir: 数据目录路径，如果为None则自动使用最新数据
		save_statistics: 是否打印统计信息
	
	返回:
		str: 保存的图像路径，如果失败则返回None
	"""
	import matplotlib
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
		thetaL = theta_array[:, 0] if theta_array.ndim > 1 else theta_array
		thetaR = theta_array[:, 1] if theta_array.ndim > 1 else np.zeros_like(thetaL)
		
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
		
		# 找到实验开始的索引
		#start_idx = find_experiment_start(ipaL_data, ipaR_data)
		start_idx = 0
		end_idx = -1

		# for i in range(len(left_scales)):
		# 	if left_scales[i] > 2:
		# 		left_scales[i] = 2
		# for i in range(len(right_scales)):
		# 	if right_scales[i] > 2:
		# 		right_scales[i] = 2

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
		
		min_length = min(lengths)
		if min_length < 1:
			print(
				f"[visualize_output] 无有效帧可绘制 (min_length={min_length})。"
				f"请检查各 .npy 是否非空且长度一致。"
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
		color_left_light = '#5dade2'
		color_right_light = '#ec7063'
		
		# 创建可视化 - 4行2列布局
		fig, axs = plt.subplots(3, 2, figsize=(16, 18))
		fig.patch.set_facecolor('white')
		
		filter_method = 'Real-Time Filtered' if use_filtered else 'Gaussian Smoothed'

		
		axs[0, 0].plot(timestamps, psms_distance_data, color='#f39c12', alpha=0.8, 
					   linewidth=2.5, label='Distance Between PSMs')
		axs[0, 0].fill_between(timestamps, psms_distance_data, alpha=0.15, color='#f39c12')
		axs[0, 0].set_title('Distance Between Left and Right PSM', 
							fontsize=13, fontweight='bold', pad=15)
		axs[0, 0].set_xlabel('Frame Index', fontsize=11)
		axs[0, 0].set_ylabel('Distance (meters)', fontsize=11)
		axs[0, 0].legend(loc='best', frameon=True, fancybox=True, shadow=True)
		axs[0, 0].grid(True, alpha=0.2, linestyle='--')
		axs[0, 0].set_facecolor('#f8f9fa')

		
		axs[0, 1].plot(timestamps, left_distances, color=color_left, alpha=0.8, 
					   linewidth=2.5, label='Left Hand Distance')
		axs[0, 1].plot(timestamps, right_distances, color=color_right, alpha=0.8, 
					   linewidth=2.5, label='Right Hand Distance')
		axs[0, 1].fill_between(timestamps, left_distances, alpha=0.15, color=color_left)
		axs[0, 1].fill_between(timestamps, right_distances, alpha=0.15, color=color_right)
		axs[0, 1].set_title('3D Distance Between Hands and Gaze Point', 
							fontsize=13, fontweight='bold', pad=15)
		axs[0, 1].set_xlabel('Frame Index', fontsize=11)
		axs[0, 1].set_ylabel('Distance (meters)', fontsize=11)
		axs[0, 1].legend(loc='best', frameon=True, fancybox=True, shadow=True)
		axs[0, 1].grid(True, alpha=0.2, linestyle='--')
		axs[0, 1].set_facecolor('#f8f9fa')
		
		avg_dist_left = _safe_mean(left_distances)
		avg_dist_right = _safe_mean(right_distances)
		if not np.isnan(avg_dist_left):
			axs[0, 1].axhline(y=avg_dist_left, color=color_left, linestyle=':', alpha=0.6, linewidth=2)
		if not np.isnan(avg_dist_right):
			axs[0, 1].axhline(y=avg_dist_right, color=color_right, linestyle=':', alpha=0.6, linewidth=2)

		axs[1, 0].plot(timestamps, thetaL, color=color_left, alpha=0.8, 
					   linewidth=2.5, label='Left PSM Theta Factor')
		axs[1, 0].plot(timestamps, thetaR, color=color_right, alpha=0.8, 
					   linewidth=2.5, label='Right PSM Theta Factor')
		axs[1, 0].set_title('Theta Factor', 
							fontsize=13, fontweight='bold', pad=15)
		axs[1, 0].set_xlabel('Frame Index', fontsize=11)
		axs[1, 0].set_ylabel('Theta Factor', fontsize=11)
		axs[1, 0].legend(loc='best', frameon=True, fancybox=True, shadow=True)
		axs[1, 0].grid(True, alpha=0.2, linestyle='--')
		axs[1, 0].set_facecolor('#f8f9fa')

		avg_thetaL = _safe_mean(thetaL)
		avg_thetaR = _safe_mean(thetaR)
		if not np.isnan(avg_thetaL):
			axs[1, 0].axhline(y=avg_thetaL, color=color_left, linestyle=':', alpha=0.6, linewidth=2)
		if not np.isnan(avg_thetaR):
			axs[1, 0].axhline(y=avg_thetaR, color=color_right, linestyle=':', alpha=0.6, linewidth=2)
		_theta_txt_out = (
			f'Left Avg: {avg_thetaL:.2f}\nRight Avg: {avg_thetaR:.2f}'
			if not (np.isnan(avg_thetaL) and np.isnan(avg_thetaR))
			else 'Avg: N/A'
		)
		axs[1, 0].text(0.02, 0.98, _theta_txt_out,
					   transform=axs[1, 0].transAxes, verticalalignment='top', fontsize=10,
					   bbox=dict(boxstyle='round', facecolor='#ecf0f1', alpha=0.8, edgecolor='#95a5a6'))

		
		
		# 4. 速度数据 - 显示原始和滤波后的数据
		axs[1, 1].plot(timestamps, Lpsm_velocity_magnitude, color=color_left_light, 
					   alpha=0.3, linewidth=1.5, label='Left Raw')
		axs[1, 1].plot(timestamps, Rpsm_velocity_magnitude, color=color_right_light,
					   alpha=0.3, linewidth=1.5, label='Right Raw')
		Lpsm_velocity_smoothed = np.where(Lpsm_velocity_smoothed < 0, 0, Lpsm_velocity_smoothed)
		Rpsm_velocity_smoothed = np.where(Rpsm_velocity_smoothed < 0, 0, Rpsm_velocity_smoothed)
		axs[1, 1].plot(timestamps, Lpsm_velocity_smoothed, color=color_left, 
					   alpha=0.9, linewidth=2.5, label='Left Filtered')
		axs[1, 1].plot(timestamps, Rpsm_velocity_smoothed, color=color_right, 
					   alpha=0.9, linewidth=2.5, label='Right Filtered')
		axs[1, 1].set_title(f'PSM Linear Velocity ({filter_method})', 
							fontsize=13, fontweight='bold', pad=15)
		axs[1, 1].set_xlabel('Frame Index', fontsize=11)
		axs[1, 1].set_ylabel('Velocity (m/s)', fontsize=11)
		axs[1, 1].legend(loc='best', frameon=True, fancybox=True, shadow=True, fontsize=9)
		axs[1, 1].grid(True, alpha=0.2, linestyle='--')
		axs[1, 1].set_facecolor('#f8f9fa')
		
		avg_vel_left = _safe_mean(Lpsm_velocity_smoothed)
		avg_vel_right = _safe_mean(Rpsm_velocity_smoothed)
		if not np.isnan(avg_vel_left):
			axs[1, 1].axhline(y=avg_vel_left, color=color_left, linestyle=':', alpha=0.6, linewidth=2)
		if not np.isnan(avg_vel_right):
			axs[1, 1].axhline(y=avg_vel_right, color=color_right, linestyle=':', alpha=0.6, linewidth=2)
		
	
		security_factor = 1-np.exp(-1000**2 * psms_distance_data**config.init_params['B_safety'])
		axs[2, 0].plot(timestamps, security_factor, color='#9b59b6', alpha=0.9, 
					   linewidth=2.5, label=f'Security Factor')
		axs[2, 0].set_title(f'Security Factor', 
							fontsize=13, fontweight='bold', pad=15)
		axs[2, 0].set_xlabel('Frame Index', fontsize=11)
		axs[2, 0].set_ylabel('Security Factor', fontsize=11)
		axs[2, 0].legend(loc='best', frameon=True, fancybox=True, shadow=True)
		axs[2, 0].grid(True, alpha=0.2, linestyle='--')
		axs[2, 0].set_facecolor('#f8f9fa')
		
		
		# 8. [2,1] Scale数据
		axs[2, 1].plot(timestamps, left_scales*0.1, color=color_left, alpha=0.8, 
					   linewidth=2.5, label='Left Hand Scale')
		axs[2, 1].plot(timestamps, right_scales*0.1, color=color_right, alpha=0.8, 
					   linewidth=2.5, label='Right Hand Scale')
		axs[2, 1].set_title('Adaptive Scaling Factors', fontsize=13, fontweight='bold', pad=15)
		axs[2, 1].set_xlabel('Frame Index', fontsize=11)
		axs[2, 1].set_ylabel('Scale Factor', fontsize=11)
		axs[2, 1].legend(loc='best', frameon=True, fancybox=True, shadow=True)
		axs[2, 1].grid(True, alpha=0.2, linestyle='--')
		axs[2, 1].set_facecolor('#f8f9fa')
		
		avg_scale_left = _safe_mean(left_scales)
		avg_scale_right = _safe_mean(right_scales)
		_scale_txt_out = (
			f'Left Avg: {avg_scale_left:.3f}\nRight Avg: {avg_scale_right:.3f}'
			if not (np.isnan(avg_scale_left) and np.isnan(avg_scale_right))
			else 'Avg: N/A'
		)
		axs[2, 1].text(0.02, 0.98, _scale_txt_out,
					   transform=axs[2, 1].transAxes, verticalalignment='top', fontsize=10,
					   bbox=dict(boxstyle='round', facecolor='#ecf0f1', alpha=0.8, edgecolor='#95a5a6'))
		
		# 保存图表
		plt.tight_layout()
		

		# plt.show()
		output_path = os.path.join(data_dir, 'visualization_output.png')
		plt.savefig(output_path, dpi=300, bbox_inches='tight')
		
		return 
	except FileNotFoundError as e:
		print(f"Error loading data: {e}")
		return None
	except Exception as e:
		print(f"Error generating visualization: {e}")
		import traceback
		traceback.print_exc()
		return None

def visualize_demo_state(data_dir=None):
	"""
	可视化最新一条演示的运动学状态（位置、速度、夹爪），
	并用虚线标出预测阶段转换帧和底部阶段色带。

	布局（5 行 × 2 列，底行全宽）：
	  Row 0: Position  (left / right)
	  Row 1: Velocity 3-axis  (left / right)
	  Row 2: Velocity magnitude  (left / right)
	  Row 3: Gripper state  (left / right)
	  Row 4: Phase color bar  (full width, only if phase_labels exists)
	"""
	import matplotlib
	matplotlib.use('Agg')
	import matplotlib.pyplot as plt
	from matplotlib.gridspec import GridSpec
	from matplotlib.colors import ListedColormap, BoundaryNorm
	from matplotlib.patches import Patch

	try:
		if data_dir is None:
			current_dir = os.path.dirname(os.path.abspath(__file__))
			data_base_dir = os.path.join(current_dir, 'data')
			data_dir = get_latest_data_dir(data_base_dir)

		print(f"[visualize_demo_state] data_dir = {data_dir}")

		# ── 加载 scaler（与 realtime_phase_predictor 同源）──────────────
		trajectory_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'Trajectory')
		if trajectory_dir not in sys.path:
			sys.path.insert(0, trajectory_dir)

		model_path = os.path.join(
			os.path.dirname(os.path.abspath(__file__)),
			'Trajectory', 'LSTM_model', 'lstm_sequence_model.pth'
		)
		scalers = None
		if os.path.exists(model_path):
			ckpt = torch.load(model_path, map_location='cpu')
			if 'scalers' in ckpt:
				scalers = ckpt['scalers']
				print("[visualize_demo_state] Scalers loaded from checkpoint.")
			else:
				print("[visualize_demo_state] Checkpoint has no scalers, "
					  "fitting from training data …")
				from Trajectory.load_data import (
					load_demonstrations_state, _scale_demos, ratio,
				)
				excluded = [80, 81, 92, 109, 112, 117, 122, 144, 145]
				demo_ids = np.delete(np.arange(148), excluded)
				raw_all = load_demonstrations_state(
					shuffle=True, without_quat=False,
					resample=False, demo_id_list=demo_ids,
				)
				bound = round(ratio * len(raw_all))
				_, scalers = _scale_demos(raw_all[:bound])
		else:
			print(f"[visualize_demo_state] Model not found: {model_path}, "
				  "skipping normalization.")

		join = os.path.join
		L_pos = np.load(join(data_dir, 'Lpsm_position.npy'), allow_pickle=True)
		R_pos = np.load(join(data_dir, 'Rpsm_position.npy'), allow_pickle=True)
		L_vel = np.load(join(data_dir, 'Lpsm_velocity.npy'), allow_pickle=True)
		R_vel = np.load(join(data_dir, 'Rpsm_velocity.npy'), allow_pickle=True)
		L_grip = np.load(join(data_dir, 'Lgripper_state.npy'), allow_pickle=True)
		R_grip = np.load(join(data_dir, 'Rgripper_state.npy'), allow_pickle=True)

		try:
			phase_labels = np.load(join(data_dir, 'phase_labels.npy'))
			has_phase = True
		except FileNotFoundError:
			phase_labels = None
			has_phase = False
		phase_probs = None
		try:
			phase_probs = np.load(join(data_dir, 'phase_probs.npy'))
		except FileNotFoundError:
			pass

		# 对齐长度
		min_len = min(len(L_pos), len(R_pos), len(L_vel), len(R_vel),
					  len(L_grip), len(R_grip))
		if has_phase:
			min_len = min(min_len, len(phase_labels))
			phase_labels = phase_labels[:min_len]
		if phase_probs is not None:
			phase_probs = phase_probs[:min_len]
		L_pos  = np.array(L_pos[:min_len]);  R_pos  = np.array(R_pos[:min_len])
		L_vel  = np.array(L_vel[:min_len]);  R_vel  = np.array(R_vel[:min_len])
		L_grip = np.array(L_grip[:min_len]); R_grip = np.array(R_grip[:min_len])

		if min_len < 1:
			print("[visualize_demo_state] 无有效帧 (min_len=0)，跳过绘图。")
			return None

		# ── 归一化 pos / vel 前 3 列（与训练一致）─────────────────────────
		# 原始数据：pos=(x,y,z,timestamp)  vel=(vx,vy,vz,wx,wy,wz)
		# scaler 只针对前 3 列（3D 位置 / 3D 线速度）
		if scalers is not None:
			L_pos[:, :3] = scalers['pos'][0].transform(L_pos[:, :3])
			R_pos[:, :3] = scalers['pos'][1].transform(R_pos[:, :3])
			L_vel[:, :3] = scalers['vel3'][0].transform(L_vel[:, :3])
			R_vel[:, :3] = scalers['vel3'][1].transform(R_vel[:, :3])

		L_speed = np.linalg.norm(L_vel, axis=1)
		R_speed = np.linalg.norm(R_vel, axis=1)
		t = np.arange(min_len)

		# ── 阶段颜色定义（供转换虚线和底部色带共用）─────────────────────
		PHASE_NAMES = [
			'P0 Right Move', 'P1 Pick Needle', 'P2 Right Move2',
			'P3 Pass Needle', 'P4 Left Move', 'P5 Left Pick', 'P6 Pull Thread',
		]
		phase_colors = [
			'#95a5a6',
			'#1abc9c', '#3498db', '#9b59b6',
			'#e67e22', '#e74c3c', '#2ecc71', '#f1c40f',
		]

		# ── 阶段转换帧 (frame, new_phase)（包含 warmup→P0）─────────────
		transitions = []
		if has_phase:
			for i in range(1, len(phase_labels)):
				if phase_labels[i] >= 0 and phase_labels[i] != phase_labels[i-1]:
					transitions.append((i, int(phase_labels[i])))

		# ── 创建画布 ───────────────────────────────────────────────────────
		n_rows = 4
		if has_phase:
			fig = plt.figure(figsize=(14, 22))
			gs = GridSpec(n_rows + 1, 2, figure=fig,
						 height_ratios=[1]*n_rows + [1.0], hspace=0.38)
		else:
			fig = plt.figure(figsize=(14, 17))
			gs = GridSpec(n_rows, 2, figure=fig, hspace=0.38)

		axs = np.empty((n_rows, 2), dtype=object)
		for r in range(n_rows):
			for c in range(2):
				axs[r, c] = fig.add_subplot(gs[r, c])
		fig.patch.set_facecolor('white')

		def _style(ax, title, ylabel='Value'):
			ax.set_title(title, fontsize=11, fontweight='bold')
			ax.set_ylabel(ylabel, fontsize=9)
			ax.axhline(0, color='k', linewidth=0.4, linestyle='--')
			ax.grid(True, alpha=0.3)

		def _draw_transitions(ax):
			for tf, new_phase in transitions:
				clr = phase_colors[new_phase + 1] if 0 <= new_phase < len(PHASE_NAMES) else '#7f8c8d'
				ax.axvline(tf, color=clr, linestyle='--', alpha=0.7, linewidth=1.0)

		# ── Row 0: Position ─────────────────────────────────────────────
		for ci, (label, clr) in enumerate(zip(['x','y','z'],
											  ['tab:red','tab:green','tab:blue'])):
			axs[0,0].plot(t, L_pos[:,ci], label=label, color=clr, linewidth=0.8)
			axs[0,1].plot(t, R_pos[:,ci], label=label, color=clr, linewidth=0.8)
		pos_ylabel = 'Position (scaled)' if scalers else 'Position (m)'
		for ax, side in zip(axs[0], ['Left','Right']):
			_style(ax, f'{side} Hand — Position', pos_ylabel)
			ax.legend(loc='upper right', fontsize=8)
			_draw_transitions(ax)

		# ── Row 1: Velocity 3-axis ──────────────────────────────────────
		for ci, (label, clr) in enumerate(zip(['vx','vy','vz'],
											  ['tab:red','tab:green','tab:blue'])):
			axs[1,0].plot(t, L_vel[:,ci], label=label, color=clr, linewidth=0.8)
			axs[1,1].plot(t, R_vel[:,ci], label=label, color=clr, linewidth=0.8)
		vel_ylabel = 'Velocity (scaled)' if scalers else 'Velocity (m/s)'
		for ax, side in zip(axs[1], ['Left','Right']):
			_style(ax, f'{side} Hand — Velocity 3-axis', vel_ylabel)
			ax.legend(loc='upper right', fontsize=8)
			_draw_transitions(ax)

		# ── Row 2: Velocity scalar ──────────────────────────────────────
		axs[2,0].plot(t, L_speed, color='tab:purple', linewidth=0.8)
		axs[2,1].plot(t, R_speed, color='tab:orange', linewidth=0.8)
		spd_ylabel = 'Speed (scaled)' if scalers else 'Speed (m/s)'
		for ax, side in zip(axs[2], ['Left','Right']):
			_style(ax, f'{side} Hand — Speed (magnitude)', spd_ylabel)
			_draw_transitions(ax)

		# ── Row 3: Gripper ──────────────────────────────────────────────
		axs[3,0].plot(t, L_grip, color='tab:brown', linewidth=0.8)
		axs[3,0].fill_between(t, L_grip, alpha=0.25, color='tab:brown')
		axs[3,1].plot(t, R_grip, color='tab:cyan',  linewidth=0.8)
		axs[3,1].fill_between(t, R_grip, alpha=0.25, color='tab:cyan')

		for ax, grip, side in zip(axs[3], [L_grip, R_grip], ['Left','Right']):
			_style(ax, f'{side} Hand — Gripper', 'State (0/1)')
			ax.set_xlabel('Frame')
			_draw_transitions(ax)
			diff = np.diff(grip.astype(float), prepend=grip[0])
			rise = t[diff > 0.5]
			fall = t[diff < -0.5]
			# for f in rise:
			# 	ax.axvline(f, color='green', linewidth=0.8, linestyle='--', alpha=0.7)
			# for f in fall:
			# 	ax.axvline(f, color='red', linewidth=0.8, linestyle='--', alpha=0.7)
			lines = []
			if len(rise): lines.append('0→1: ' + ', '.join(str(int(f)) for f in rise))
			if len(fall): lines.append('1→0: ' + ', '.join(str(int(f)) for f in fall))
			if lines:
				ax.text(0.01, 0.97, '\n'.join(lines), transform=ax.transAxes,
						fontsize=7, va='top',
						bbox=dict(boxstyle='round,pad=0.3', facecolor='lightyellow',
								  edgecolor='gray', alpha=0.8))

		# ── Row 4: Phase color bar + probability curves ───────────────
		if has_phase:
			cmap = ListedColormap(phase_colors)
			norm = BoundaryNorm(np.arange(-0.5, len(phase_colors)), len(phase_colors))
			shifted = phase_labels.astype(int) + 1
			T = len(shifted)
			t_frames = np.arange(T)
			T_plot = max(T, 1)

			ax_ph = fig.add_subplot(gs[n_rows, :])
			if T > 0:
				ax_ph.imshow(shifted[np.newaxis, :], aspect='auto', cmap=cmap, norm=norm,
							 extent=[0, T_plot, 0, 1], interpolation='nearest')

				if phase_probs is not None and len(phase_probs) == T:
					ax_prob = ax_ph.twinx()
					num_classes = phase_probs.shape[1]
					for ci in range(num_classes):
						ax_prob.plot(t_frames, phase_probs[:, ci],
									 color='black', linewidth=1.0,
									 alpha=0.6)
					ax_prob.set_ylim(-0.05, 1.05)
					ax_prob.set_ylabel('Probability', fontsize=10)
					ax_prob.tick_params(labelsize=8)

				ax_ph.set_yticks([])
				ax_ph.set_xlim(0, T_plot)
				ax_ph.set_xlabel('Frame Index', fontsize=11)
				ax_ph.set_title('Predicted Surgical Phase & Probability',
								fontsize=11, fontweight='bold', pad=8)
				legend_labels = ['Warmup'] + PHASE_NAMES
				handles = [Patch(facecolor=c, label=l) for c, l in zip(phase_colors, legend_labels)]
				ax_ph.legend(handles=handles, loc='upper center',
							 bbox_to_anchor=(0.5, -0.15), ncol=4, fontsize=8,
							 frameon=True, fancybox=True, shadow=True)
			else:
				ax_ph.text(0.5, 0.5, 'No phase labels', ha='center', va='center',
						   transform=ax_ph.transAxes, fontsize=12)
				ax_ph.set_axis_off()

		fig.suptitle(f'Demo State Visualization — {os.path.basename(data_dir)}',
					 fontsize=14, fontweight='bold')

		output_path = os.path.join(data_dir, 'demo_state_visualization.png')
		fig.savefig(output_path, dpi=200, bbox_inches='tight')
		plt.close(fig)
		print(f"[visualize_demo_state] saved to: {output_path}")
		return output_path

	except Exception as e:
		print(f"[visualize_demo_state] Error: {e}")
		import traceback
		traceback.print_exc()
		return None


# 如果直接运行此脚本，使用最新数据生成可视化
if __name__ == '__main__':
	print("Generating visualization...")
	# result = visualize_data(data_dir='data/0_data_01-18')
	# if result:
	# 	print(f"\nSuccess! Visualization saved to: {result}")
	# else:
	# 	print("\nFailed to generate visualization")
	# 	sys.exit(1)
	#visualize_output(data_dir='data/33_data_03-20')
	visualize_demo_state()  # 可视化最新一条演示的运动学状态
	visualize_data()
