import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import os
import sys
from scipy.ndimage import gaussian_filter1d

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

def visualize_data(data_dir=None, save_statistics=True, show_plot=False):
	"""
	生成数据可视化图表并保存
	
	参数:
		data_dir: 数据目录路径，如果为None则自动使用最新数据
		save_statistics: 是否打印统计信息
	
	返回:
		str: 保存的图像路径，如果失败则返回None
	"""
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
		
		GP_distance_data = np.load(os.path.join(data_dir, 'GP_distance_data.npy'), allow_pickle=True)
		GP_distance_array = np.array(GP_distance_data)
		left_distances = GP_distance_array[:, 0]
		right_distances = GP_distance_array[:, 1]
		
		scale_data = np.load(os.path.join(data_dir, 'scale_data.npy'), allow_pickle=True)
		scale_array = np.array(scale_data)
		left_scales = scale_array[:, 0]
		right_scales = scale_array[:, 1]
		
		psms_distance_data = np.load(os.path.join(data_dir, 'psms_distance_data.npy'), allow_pickle=True)
		
		# 找到实验开始的索引
		start_idx = find_experiment_start(ipaL_data, ipaR_data) + 2
		
		# 如果检测到有预实验阶段，截取数据
		if start_idx > 0:
			if save_statistics:
				print(f"  Removing {start_idx} pre-experiment data points")
			ipaL_data = ipaL_data[start_idx:]
			ipaR_data = ipaR_data[start_idx:]
			Lpsm_velocity_magnitude = Lpsm_velocity_magnitude[start_idx:]
			Rpsm_velocity_magnitude = Rpsm_velocity_magnitude[start_idx:]
			left_distances = left_distances[start_idx:]
			right_distances = right_distances[start_idx:]
			left_scales = left_scales[start_idx:]
			right_scales = right_scales[start_idx:]
			psms_distance_data = psms_distance_data[start_idx:]
			GP_distance_array = GP_distance_array[start_idx:]
		
		# 确保所有数据长度一致
		min_length = min(len(ipaL_data), len(ipaR_data), len(Lpsm_velocity_magnitude), 
						 len(Rpsm_velocity_magnitude), len(left_distances), len(right_distances),
						 len(left_scales), len(right_scales), len(psms_distance_data))
		
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
		
		# 计算平均IPA
		ipa_average = (ipaL_data + ipaR_data) / 2
		
		# 对速度数据进行平滑处理
		sigma = 5
		Lpsm_velocity_smoothed = gaussian_filter1d(Lpsm_velocity_magnitude, sigma=sigma)
		Rpsm_velocity_smoothed = gaussian_filter1d(Rpsm_velocity_magnitude, sigma=sigma)
		
		# 定义配色方案
		color_left = '#3498db'
		color_right = '#e74c3c'
		color_left_light = '#5dade2'
		color_right_light = '#ec7063'
		
		# 创建可视化 - 3行2列布局
		fig, axs = plt.subplots(3, 2, figsize=(16, 18))
		fig.patch.set_facecolor('white')
		
		# 1. IPA数据（左右手）
		axs[0, 0].scatter(timestamps, ipaL_data, c=color_left, alpha=0.6, s=15, 
						  edgecolors='none', label='Left Hand IPA')
		axs[0, 0].scatter(timestamps, ipaR_data, c=color_right, alpha=0.6, s=15, 
						  edgecolors='none', label='Right Hand IPA')
		axs[0, 0].set_title('IPA Data (Index of Pupillary Activity)', 
							fontsize=13, fontweight='bold', pad=15)
		axs[0, 0].set_xlabel('Frame Index', fontsize=11)
		axs[0, 0].set_ylabel('IPA Values', fontsize=11)
		axs[0, 0].legend(loc='best', frameon=True, fancybox=True, shadow=True)
		axs[0, 0].grid(True, alpha=0.2, linestyle='--')
		axs[0, 0].set_facecolor('#f8f9fa')
		
		# 2. 平均IPA数据
		axs[0, 1].scatter(timestamps, ipa_average, c='#9b59b6', alpha=0.6, s=15, 
						  edgecolors='none', label='Average IPA')
		axs[0, 1].set_title('Average IPA (Both Hands)', 
							fontsize=13, fontweight='bold', pad=15)
		axs[0, 1].set_xlabel('Frame Index', fontsize=11)
		axs[0, 1].set_ylabel('Average IPA Value', fontsize=11)
		axs[0, 1].legend(loc='best', frameon=True, fancybox=True, shadow=True)
		axs[0, 1].grid(True, alpha=0.2, linestyle='--')
		axs[0, 1].set_facecolor('#f8f9fa')
		
		# 3. GP距离数据
		axs[1, 0].plot(timestamps, left_distances, color=color_left, alpha=0.8, 
					   linewidth=2.5, label='Left Hand Distance')
		axs[1, 0].plot(timestamps, right_distances, color=color_right, alpha=0.8, 
					   linewidth=2.5, label='Right Hand Distance')
		axs[1, 0].fill_between(timestamps, left_distances, alpha=0.15, color=color_left)
		axs[1, 0].fill_between(timestamps, right_distances, alpha=0.15, color=color_right)
		axs[1, 0].set_title('3D Distance Between Hands and Gaze Point', 
							fontsize=13, fontweight='bold', pad=15)
		axs[1, 0].set_xlabel('Frame Index', fontsize=11)
		axs[1, 0].set_ylabel('Distance (meters)', fontsize=11)
		axs[1, 0].legend(loc='best', frameon=True, fancybox=True, shadow=True)
		axs[1, 0].grid(True, alpha=0.2, linestyle='--')
		axs[1, 0].set_facecolor('#f8f9fa')
		
		avg_dist_left = np.mean(left_distances)
		avg_dist_right = np.mean(right_distances)
		axs[1, 0].axhline(y=avg_dist_left, color=color_left, linestyle=':', alpha=0.6, linewidth=2)
		axs[1, 0].axhline(y=avg_dist_right, color=color_right, linestyle=':', alpha=0.6, linewidth=2)
		
		# 4. 速度数据
		axs[1, 1].plot(timestamps, Lpsm_velocity_magnitude, color=color_left_light, 
					   alpha=0.3, linewidth=0.8, label='Left Raw')
		axs[1, 1].plot(timestamps, Rpsm_velocity_magnitude, color=color_right_light, 
					   alpha=0.3, linewidth=0.8, label='Right Raw')
		axs[1, 1].plot(timestamps, Lpsm_velocity_smoothed, color=color_left, 
					   alpha=0.9, linewidth=2.5, label='Left Hand Velocity (Smoothed)')
		axs[1, 1].plot(timestamps, Rpsm_velocity_smoothed, color=color_right, 
					   alpha=0.9, linewidth=2.5, label='Right Hand Velocity (Smoothed)')
		axs[1, 1].set_title('PSM Linear Velocity (Gaussian Smoothed)', 
							fontsize=13, fontweight='bold', pad=15)
		axs[1, 1].set_xlabel('Frame Index', fontsize=11)
		axs[1, 1].set_ylabel('Velocity (m/s)', fontsize=11)
		axs[1, 1].legend(loc='best', frameon=True, fancybox=True, shadow=True, fontsize=9)
		axs[1, 1].grid(True, alpha=0.2, linestyle='--')
		axs[1, 1].set_facecolor('#f8f9fa')
		
		avg_vel_left = np.mean(Lpsm_velocity_smoothed)
		avg_vel_right = np.mean(Rpsm_velocity_smoothed)
		axs[1, 1].axhline(y=avg_vel_left, color=color_left, linestyle=':', alpha=0.6, linewidth=2)
		axs[1, 1].axhline(y=avg_vel_right, color=color_right, linestyle=':', alpha=0.6, linewidth=2)
		
		# 5. PSMs距离数据
		axs[2, 0].plot(timestamps, psms_distance_data, color='#f39c12', alpha=0.8, 
					   linewidth=2.5, label='Distance Between PSMs')
		axs[2, 0].fill_between(timestamps, psms_distance_data, alpha=0.15, color='#f39c12')
		axs[2, 0].set_title('Distance Between Left and Right PSM', 
							fontsize=13, fontweight='bold', pad=15)
		axs[2, 0].set_xlabel('Frame Index', fontsize=11)
		axs[2, 0].set_ylabel('Distance (meters)', fontsize=11)
		axs[2, 0].legend(loc='best', frameon=True, fancybox=True, shadow=True)
		axs[2, 0].grid(True, alpha=0.2, linestyle='--')
		axs[2, 0].set_facecolor('#f8f9fa')
		
		avg_psms_dist = np.mean(psms_distance_data)
		min_psms_dist = np.min(psms_distance_data)
		max_psms_dist = np.max(psms_distance_data)
		axs[2, 0].axhline(y=avg_psms_dist, color='#f39c12', linestyle=':', alpha=0.6, linewidth=2)
		axs[2, 0].text(0.02, 0.98, f'Avg: {avg_psms_dist:.4f}m\nMin: {min_psms_dist:.4f}m\nMax: {max_psms_dist:.4f}m',
					   transform=axs[2, 0].transAxes, verticalalignment='top', fontsize=10,
					   bbox=dict(boxstyle='round', facecolor='#ecf0f1', alpha=0.8, edgecolor='#95a5a6'))
		
		# 6. Scale数据
		axs[2, 1].plot(timestamps, left_scales, color=color_left, alpha=0.8, 
					   linewidth=2.5, label='Left Hand Scale')
		axs[2, 1].plot(timestamps, right_scales, color=color_right, alpha=0.8, 
					   linewidth=2.5, label='Right Hand Scale')
		axs[2, 1].set_title('Adaptive Scaling Factors', fontsize=13, fontweight='bold', pad=15)
		axs[2, 1].set_xlabel('Frame Index', fontsize=11)
		axs[2, 1].set_ylabel('Scale Factor', fontsize=11)
		axs[2, 1].legend(loc='best', frameon=True, fancybox=True, shadow=True)
		axs[2, 1].grid(True, alpha=0.2, linestyle='--')
		
		avg_scale_left = np.mean(left_scales)
		avg_scale_right = np.mean(right_scales)
		axs[2, 1].text(0.02, 0.98, f'Left Avg: {avg_scale_left:.3f}\nRight Avg: {avg_scale_right:.3f}',
					   transform=axs[2, 1].transAxes, verticalalignment='top', fontsize=10,
					   bbox=dict(boxstyle='round', facecolor='#ecf0f1', alpha=0.8, edgecolor='#95a5a6'))
		
		# 保存图表
		plt.tight_layout()
		output_path = os.path.join(data_dir, 'visualization_results.png')
		plt.savefig(output_path, dpi=300, bbox_inches='tight')
		if show_plot:
			plt.show()
		else:
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


# 如果直接运行此脚本，使用最新数据生成可视化
if __name__ == '__main__':
	print("Generating visualization...")
	result = visualize_data(save_statistics=True, show_plot=True)
	if result:
		print(f"\nSuccess! Visualization saved to: {result}")
	else:
		print("\nFailed to generate visualization")
		sys.exit(1)
