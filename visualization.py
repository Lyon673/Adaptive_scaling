import numpy as np
import matplotlib.pyplot as plt
import os
import sys
from scipy.ndimage import gaussian_filter1d

# 添加获取最新数据目录的函数
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
	print(f"Using latest data directory: {latest_dir}")
	return latest_dir

# 获取当前脚本所在目录
current_file_path = os.path.abspath(__file__)
current_dir = os.path.dirname(current_file_path)
data_base_dir = os.path.join(current_dir, 'data')

# 获取最新数据目录
try:
	latest_dir = get_latest_data_dir(data_base_dir)
except FileNotFoundError as e:
	print(f"Error: {e}")
	sys.exit(1)

# 加载数据
print("Loading data files...")
try:
	# 加载IPA数据（左右手）
	ipaL_data = np.load(os.path.join(latest_dir, 'ipaL_data.npy'), allow_pickle=True)
	ipaR_data = np.load(os.path.join(latest_dir, 'ipaR_data.npy'), allow_pickle=True)
	print(f"  IPA Left: {len(ipaL_data)} points")
	print(f"  IPA Right: {len(ipaR_data)} points")
	
	# 加载速度数据（左右手）
	Lpsm_velocity = np.load(os.path.join(latest_dir, 'Lpsm_velocity.npy'), allow_pickle=True)
	Rpsm_velocity = np.load(os.path.join(latest_dir, 'Rpsm_velocity.npy'), allow_pickle=True)
	# 计算线性速度大小
	Lpsm_velocity_magnitude = np.sqrt(Lpsm_velocity[:, 0]**2 + Lpsm_velocity[:, 1]**2 + Lpsm_velocity[:, 2]**2)
	Rpsm_velocity_magnitude = np.sqrt(Rpsm_velocity[:, 0]**2 + Rpsm_velocity[:, 1]**2 + Rpsm_velocity[:, 2]**2)
	print(f"  Velocity Left: {len(Lpsm_velocity_magnitude)} points")
	print(f"  Velocity Right: {len(Rpsm_velocity_magnitude)} points")
	
	# 加载距离数据（左右手到注视点的距离）
	GP_distance_data = np.load(os.path.join(latest_dir, 'GP_distance_data.npy'), allow_pickle=True)
	GP_distance_array = np.array(GP_distance_data)
	left_distances = GP_distance_array[:, 0]
	right_distances = GP_distance_array[:, 1]
	print(f"  Distance data: {len(GP_distance_array)} points")
	
	# 加载Scale数据
	scale_data = np.load(os.path.join(latest_dir, 'scale_data.npy'), allow_pickle=True)
	scale_array = np.array(scale_data)
	left_scales = scale_array[:, 0]
	right_scales = scale_array[:, 1]
	print(f"  Scale data: {len(scale_array)} points")
	
	# 加载PSMs距离数据（两个PSM之间的距离）
	psms_distance_data = np.load(os.path.join(latest_dir, 'psms_distance_data.npy'), allow_pickle=True)
	print(f"  PSMs distance data: {len(psms_distance_data)} points")
	
except FileNotFoundError as e:
	print(f"Error loading data file: {e}")
	sys.exit(1)

# # 数据预处理 - 处理IPA异常值
# for i in range(1, len(ipaL_data)):
# 	if ipaL_data[i] < 0.5:
# 		ipaL_data[i] = ipaL_data[i-1]

# for i in range(1, len(ipaR_data)):
# 	if ipaR_data[i] < 0.5:
# 		ipaR_data[i] = ipaR_data[i-1]

# 检测实验开始点 - 找到IPA不再恒定为1的位置
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
			print(f"\n  Detected experiment start at index {i}")
			print(f"  IPA Left: {ipa_left[i]:.4f}, IPA Right: {ipa_right[i]:.4f}")
			return i
	
	print("\n  No clear experiment start detected, using all data")
	return 0

# 找到实验开始的索引
start_idx = find_experiment_start(ipaL_data, ipaR_data)+2
#start_idx = 0

# 如果检测到有预实验阶段，截取数据
if start_idx > 0:
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
	
	print(f"  Data length after trimming: {len(ipaL_data)} points")

# 确保所有数据长度一致（取最小长度）
min_length = min(len(ipaL_data), len(ipaR_data), len(Lpsm_velocity_magnitude), 
				 len(Rpsm_velocity_magnitude), len(left_distances), len(right_distances),
				 len(left_scales), len(right_scales), len(psms_distance_data))

print(f"\nUsing minimum data length: {min_length} points")

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

# 对速度数据进行平滑处理（高斯滤波）
sigma = 5  # 平滑程度，值越大越平滑
Lpsm_velocity_smoothed = gaussian_filter1d(Lpsm_velocity_magnitude, sigma=sigma)
Rpsm_velocity_smoothed = gaussian_filter1d(Rpsm_velocity_magnitude, sigma=sigma)

# 定义现代化的配色方案
color_left = '#3498db'   # 现代蓝色
color_right = '#e74c3c'  # 现代红色
color_left_light = '#5dade2'  # 浅蓝色
color_right_light = '#ec7063'  # 浅红色

# 创建可视化 - 3行2列布局
print("\nGenerating visualization...")
fig, axs = plt.subplots(3, 2, figsize=(16, 18))
fig.patch.set_facecolor('white')

# 1. IPA数据（左右手） - 使用散点图
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
# 添加移动平均线
window_size = 1
ipa_ma = np.convolve(ipa_average, np.ones(window_size)/window_size, mode='same')
#axs[0, 1].plot(timestamps, ipa_ma, color='#8e44ad', linewidth=2.5, alpha=0.8, 
#               label=f'Moving Average (window={window_size})')
axs[0, 1].set_title('Average IPA (Both Hands)', 
                    fontsize=13, fontweight='bold', pad=15)
axs[0, 1].set_xlabel('Frame Index', fontsize=11)
axs[0, 1].set_ylabel('Average IPA Value', fontsize=11)
axs[0, 1].legend(loc='best', frameon=True, fancybox=True, shadow=True)
axs[0, 1].grid(True, alpha=0.2, linestyle='--')
axs[0, 1].set_facecolor('#f8f9fa')

# # 添加平均值线
avg_ipa = np.mean(ipa_average)
# axs[0, 1].axhline(y=avg_ipa, color='#8e44ad', linestyle=':', alpha=0.6, linewidth=2)
# axs[0, 1].text(0.02, 0.98, f'Overall Avg: {avg_ipa:.4f}',
#                transform=axs[0, 1].transAxes, verticalalignment='top', fontsize=10,
#                bbox=dict(boxstyle='round', facecolor='#ecf0f1', alpha=0.8, edgecolor='#95a5a6'))

# 3. GP距离数据（左右手到注视点）

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

# 添加平均值线
avg_dist_left = np.mean(left_distances)
avg_dist_right = np.mean(right_distances)
axs[1, 0].axhline(y=avg_dist_left, color=color_left, linestyle=':', alpha=0.6, linewidth=2)
axs[1, 0].axhline(y=avg_dist_right, color=color_right, linestyle=':', alpha=0.6, linewidth=2)

# 4. 速度数据（左右手）- 平滑后的数据
# 先绘制原始数据（浅色背景）
axs[1, 1].plot(timestamps, Lpsm_velocity_magnitude, color=color_left_light, 
               alpha=0.3, linewidth=0.8, label='Left Raw')
axs[1, 1].plot(timestamps, Rpsm_velocity_magnitude, color=color_right_light, 
               alpha=0.3, linewidth=0.8, label='Right Raw')
# 再绘制平滑后的数据（粗线）
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

# 添加平均值线
avg_vel_left = np.mean(Lpsm_velocity_smoothed)
avg_vel_right = np.mean(Rpsm_velocity_smoothed)
axs[1, 1].axhline(y=avg_vel_left, color=color_left, linestyle=':', alpha=0.6, linewidth=2)
axs[1, 1].axhline(y=avg_vel_right, color=color_right, linestyle=':', alpha=0.6, linewidth=2)

# 5. PSMs距离数据（两个PSM之间的距离）
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

# 添加平均值和统计信息
avg_psms_dist = np.mean(psms_distance_data)
min_psms_dist = np.min(psms_distance_data)
max_psms_dist = np.max(psms_distance_data)
axs[2, 0].axhline(y=avg_psms_dist, color='#f39c12', linestyle=':', alpha=0.6, linewidth=2)
axs[2, 0].text(0.02, 0.98, f'Avg: {avg_psms_dist:.4f}m\nMin: {min_psms_dist:.4f}m\nMax: {max_psms_dist:.4f}m',
               transform=axs[2, 0].transAxes, verticalalignment='top', fontsize=10,
               bbox=dict(boxstyle='round', facecolor='#ecf0f1', alpha=0.8, edgecolor='#95a5a6'))

# 6. Scale数据（左右手）
axs[2, 1].plot(timestamps, left_scales, color=color_left, alpha=0.8, 
               linewidth=2.5, label='Left Hand Scale')
axs[2, 1].plot(timestamps, right_scales, color=color_right, alpha=0.8, 
               linewidth=2.5, label='Right Hand Scale')
#axs[2, 1].fill_between(timestamps, left_scales, alpha=0.15, color=color_left)
#axs[2, 1].fill_between(timestamps, right_scales, alpha=0.15, color=color_right)
axs[2, 1].set_title('Adaptive Scaling Factors', fontsize=13, fontweight='bold', pad=15)
axs[2, 1].set_xlabel('Frame Index', fontsize=11)
axs[2, 1].set_ylabel('Scale Factor', fontsize=11)
axs[2, 1].legend(loc='best', frameon=True, fancybox=True, shadow=True)
axs[2, 1].grid(True, alpha=0.2, linestyle='--')
#axs[2, 1].set_facecolor('#f8f9fa')

# # 添加参考线（scale的合理范围）
# axs[2, 1].axhline(y=1.0, color='#27ae60', linestyle=':', alpha=0.6, linewidth=2, label='Neutral (1.0)')
# axs[2, 1].axhline(y=0.1, color='#95a5a6', linestyle='--', alpha=0.4, linewidth=1.5, label='Min (0.1)')
# axs[2, 1].axhline(y=4.0, color='#95a5a6', linestyle='--', alpha=0.4, linewidth=1.5, label='Max (4.0)')

# 添加平均值文本框
avg_scale_left = np.mean(left_scales)
avg_scale_right = np.mean(right_scales)
axs[2, 1].text(0.02, 0.98, f'Left Avg: {avg_scale_left:.3f}\nRight Avg: {avg_scale_right:.3f}',
			   transform=axs[2, 1].transAxes, verticalalignment='top', fontsize=10,
			   bbox=dict(boxstyle='round', facecolor='#ecf0f1', alpha=0.8, edgecolor='#95a5a6'))

# 调整子图布局
plt.tight_layout()

# 保存图表
output_path = os.path.join(latest_dir, 'visualization_results.png')
plt.savefig(output_path, dpi=300, bbox_inches='tight')
print(f"\nVisualization saved to: {output_path}")

# 打印统计信息
print("\n" + "="*50)
print("Statistics Summary:")
print("="*50)
print(f"\nIPA Data:")
print(f"  Left  - Mean: {np.mean(ipaL_data):.4f}, Std: {np.std(ipaL_data):.4f}")
print(f"  Right - Mean: {np.mean(ipaR_data):.4f}, Std: {np.std(ipaR_data):.4f}")

print(f"\nVelocity Data (m/s):")
print(f"  Left  - Mean: {avg_vel_left:.4f}, Max: {np.max(Lpsm_velocity_magnitude):.4f}")
print(f"  Right - Mean: {avg_vel_right:.4f}, Max: {np.max(Rpsm_velocity_magnitude):.4f}")

print(f"\nDistance Data (meters):")
print(f"  Left  - Mean: {avg_dist_left:.4f}, Min: {np.min(left_distances):.4f}, Max: {np.max(left_distances):.4f}")
print(f"  Right - Mean: {avg_dist_right:.4f}, Min: {np.min(right_distances):.4f}, Max: {np.max(right_distances):.4f}")

print(f"\nScale Data:")
print(f"  Left  - Mean: {avg_scale_left:.3f}, Min: {np.min(left_scales):.3f}, Max: {np.max(left_scales):.3f}")
print(f"  Right - Mean: {avg_scale_right:.3f}, Min: {np.min(right_scales):.3f}, Max: {np.max(right_scales):.3f}")

print(f"\nAverage IPA:")
print(f"  Overall - Mean: {avg_ipa:.4f}, Std: {np.std(ipa_average):.4f}")

print(f"\nPSMs Distance (Between Left and Right Hand):")
print(f"  Mean: {avg_psms_dist:.4f}m, Min: {min_psms_dist:.4f}m, Max: {max_psms_dist:.4f}m")
print("="*50)

# 显示图表
try:
    plt.show()
except:
	print("\nNote: GUI display not available, but image has been saved.")
