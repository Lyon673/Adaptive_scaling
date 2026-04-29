import numpy as np
import os
import sys
from scipy.ndimage import gaussian_filter1d
import params.config as config
from scipy import stats
import csv
import torch

def get_latest_data_dir(data_base_dir):
	"""获取data文件夹下最新的数据目录"""
	if not os.path.exists(data_base_dir):
		raise FileNotFoundError(f"Data directory not found: {data_base_dir}")
	
	subdirs = [d for d in os.listdir(data_base_dir) 
			   if os.path.isdir(os.path.join(data_base_dir, d))]
	
	if not subdirs:
		raise FileNotFoundError(f"No subdirectories found in: {data_base_dir}")
	
	subdirs.sort(key=lambda x: int(x.split('_')[0]) if x.split('_')[0].isdigit() else 0)
	latest_dir = os.path.join(data_base_dir, subdirs[-1])
	return latest_dir

def _safe_mean(arr):
	a = np.asarray(arr)
	if a.size == 0:
		return np.nan
	return float(np.mean(a))

def find_experiment_start(ipa_left, ipa_right, threshold=1, window=1):
	min_length = min(len(ipa_left), len(ipa_right))
	for i in range(min_length - window):
		avg_left = np.mean(ipa_left[i:i+window])
		avg_right = np.mean(ipa_right[i:i+window])
		if avg_left < threshold and avg_right < threshold:
			return i
	return 0

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

# =========================================================================
# 【新增】基于 Bimanual Dominance Weight 的标量卡尔曼滤波器
# =========================================================================
def apply_dominance_kalman_filter(raw_scales, alphas, Q=0.01, R_min=0.05, R_penalty=100.0, eta=3.0):
	"""
	模拟带注意力权重的卡尔曼滤波过程
	"""
	T = len(raw_scales)
	filtered_scales = np.zeros(T)
	if T == 0: return filtered_scales

	# 初始化
	S_prev = raw_scales[0]
	P_prev = 1.0
	filtered_scales[0] = S_prev

	for t in range(1, T):
		# 1. 预测 (Predict)
		P_pred = P_prev + Q

		# 2. 计算观测噪声与卡尔曼增益
		alpha = alphas[t]
		# 当 alpha 极小时(非主导)，R_t 会受到巨大的指数惩罚
		R_t = R_min + R_penalty * ((1.0 - alpha) ** eta)
		K_t = P_pred / (P_pred + R_t)

		# 3. 更新 (Update)
		S_curr = S_prev + K_t * (raw_scales[t] - S_prev)
		P_curr = (1.0 - K_t) * P_pred

		filtered_scales[t] = S_curr

		# 迭代推进
		S_prev = S_curr
		P_prev = P_curr

	return filtered_scales


def visualize_data(data_dir=None, save_statistics=True, output_path=None):
	import matplotlib
	#matplotlib.use('Agg')
	import matplotlib.pyplot as plt
	try:
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
		
		try:
			Lpsm_velocity_filtered = np.load(os.path.join(data_dir, 'Lpsm_velocity_filtered.npy'), allow_pickle=True)
			Rpsm_velocity_filtered = np.load(os.path.join(data_dir, 'Rpsm_velocity_filtered.npy'), allow_pickle=True)
			ipaL_data_filtered = np.load(os.path.join(data_dir, 'ipaL_data_filtered.npy'), allow_pickle=True)
			ipaR_data_filtered = np.load(os.path.join(data_dir, 'ipaR_data_filtered.npy'), allow_pickle=True)
			use_filtered = True
		except FileNotFoundError:
			use_filtered = False
		
		GP_distance_data = np.load(os.path.join(data_dir, 'GP_distance_data.npy'), allow_pickle=True)
		GP_distance_array = np.array(GP_distance_data)
		left_distances = GP_distance_array[:, 0]
		right_distances = GP_distance_array[:, 1]
		
		scale_data = np.load(os.path.join(data_dir, 'scale_data.npy'), allow_pickle=True)
		scale_array = np.array(scale_data)
		left_scales = scale_array[:, 0]
		right_scales = scale_array[:, 1]
		
		psms_distance_data = np.load(os.path.join(data_dir, 'psms_distance_data.npy'), allow_pickle=True)
		
		theta_data = np.load(os.path.join(data_dir, 'theta.npy'), allow_pickle=True)
		theta_array = np.array(theta_data)
		thetaL = np.degrees(theta_array[:, 0]) if theta_array.ndim > 1 else np.degrees(theta_array)
		thetaR = np.degrees(theta_array[:, 1]) if theta_array.ndim > 1 else np.zeros_like(thetaL)
		
		try:
			Lforward_factor = np.load(os.path.join(data_dir, 'Lforward_factor.npy'), allow_pickle=True)
			Lbackward_factor = np.load(os.path.join(data_dir, 'Lbackward_factor.npy'), allow_pickle=True)
			Rforward_factor = np.load(os.path.join(data_dir, 'Rforward_factor.npy'), allow_pickle=True)
			Rbackward_factor = np.load(os.path.join(data_dir, 'Rbackward_factor.npy'), allow_pickle=True)
			has_factors = True
		except FileNotFoundError:
			has_factors = False
		
		try:
			phase_labels = np.load(os.path.join(data_dir, 'phase_labels.npy'))
			has_phase = True
		except FileNotFoundError:
			phase_labels = None
			has_phase = False
			
		try:
			phase_probs = np.load(os.path.join(data_dir, 'phase_probs.npy'))
		except FileNotFoundError:
			phase_probs = None
		
		start_idx = 0
		end_idx = -1

		if start_idx > 0:
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
			
			if has_factors:
				Lforward_factor = Lforward_factor[start_idx:end_idx]
				Lbackward_factor = Lbackward_factor[start_idx:end_idx]
				Rforward_factor = Rforward_factor[start_idx:end_idx]
				Rbackward_factor = Rbackward_factor[start_idx:end_idx]
			
			if has_phase:
				phase_labels = phase_labels[start_idx:end_idx]
			if phase_probs is not None:
				phase_probs = phase_probs[start_idx:end_idx]
			
			if use_filtered:
				Lpsm_velocity_filtered = Lpsm_velocity_filtered[start_idx:end_idx]
				Rpsm_velocity_filtered = Rpsm_velocity_filtered[start_idx:end_idx]
				ipaL_data_filtered = ipaL_data_filtered[start_idx:end_idx]
				ipaR_data_filtered = ipaR_data_filtered[start_idx:end_idx]
		
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
			return None
		
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
		
		if use_filtered:
			Lpsm_velocity_filtered = Lpsm_velocity_filtered[:min_length]
			Rpsm_velocity_filtered = Rpsm_velocity_filtered[:min_length]
			ipaL_data_filtered = ipaL_data_filtered[:min_length]
			ipaR_data_filtered = ipaR_data_filtered[:min_length]
		
		if has_factors:
			Lforward_factor = Lforward_factor[:min_length]
			Lbackward_factor = Lbackward_factor[:min_length]
			Rforward_factor = Rforward_factor[:min_length]
			Rbackward_factor = Rbackward_factor[:min_length]
		
		if has_phase:
			phase_labels = phase_labels[:min_length]
		if phase_probs is not None:
			phase_probs = phase_probs[:min_length]

		# =====================================================================
		# 【核心新增】：构建 Alpha 序列并运行卡尔曼模拟
		# =====================================================================
		if has_phase:
			alpha_L_seq = np.full(min_length, 0.5)
			alpha_R_seq = np.full(min_length, 0.5)
			
			for i, p in enumerate(phase_labels):
				if p in [0, 1, 2, 3]:
					alpha_L_seq[i], alpha_R_seq[i] = 0.1, 1
				elif p in [4, 6]:
					alpha_L_seq[i], alpha_R_seq[i] = 1, 0.1
				elif p == 5:
					alpha_L_seq[i], alpha_R_seq[i] = 1, 0.10

			# 分别对左右臂的 Scale 进行自适应卡尔曼滤波
			# 这里由于 scale_data 是例如 15, 20 这样的数值，可以设置合适的 Q 和 R 参数
			left_scales_filtered = apply_dominance_kalman_filter(
				left_scales, alpha_L_seq, Q=0.1, R_min=0.01, R_penalty=100.0, eta=3.0
			)
			right_scales_filtered = apply_dominance_kalman_filter(
				right_scales, alpha_R_seq, Q=0.1, R_min=0.01, R_penalty=100.0, eta=3.0
			)
		else:
			left_scales_filtered = left_scales
			right_scales_filtered = right_scales
		# =====================================================================
		
		if use_filtered:
			Lpsm_velocity_smoothed = Lpsm_velocity_filtered
			Rpsm_velocity_smoothed = Rpsm_velocity_filtered
		else:
			sigma = 5
			Lpsm_velocity_smoothed = gaussian_filter1d(Lpsm_velocity_magnitude, sigma=sigma)
			Rpsm_velocity_smoothed = gaussian_filter1d(Rpsm_velocity_magnitude, sigma=sigma)
		
		color_left = '#3498db'
		color_right = '#e74c3c'
		
		from matplotlib.gridspec import GridSpec
		
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
		
		axs[0, 0].plot(timestamps, psms_distance_data, color='#f39c12', alpha=0.8, linewidth=2.5)
		axs[0, 0].fill_between(timestamps, psms_distance_data, alpha=0.15, color='#f39c12')
		axs[0, 0].set_title('Distance Between Left and Right PSM', fontsize=17, fontweight='bold', pad=15)
		axs[0, 0].set_xlabel('Frame Index', fontsize=15)
		axs[0, 0].set_ylabel('Distance', fontsize=15)
		axs[0, 0].grid(True, alpha=0.2, linestyle='--')
		axs[0, 0].set_facecolor('#f8f9fa')
		
		axs[0, 1].plot(timestamps, left_distances, color=color_left, alpha=0.8, linewidth=2.5, label='Left PSM')
		axs[0, 1].plot(timestamps, right_distances, color=color_right, alpha=0.8, linewidth=2.5, label='Right PSM')
		axs[0, 1].fill_between(timestamps, left_distances, alpha=0.15, color=color_left)
		axs[0, 1].fill_between(timestamps, right_distances, alpha=0.15, color=color_right)
		axs[0, 1].set_title('3D Distances Between Gaze Point and PSMs', fontsize=17, fontweight='bold', pad=15)
		axs[0, 1].set_xlabel('Frame Index', fontsize=15)
		axs[0, 1].set_ylabel('Distance', fontsize=15)
		axs[0, 1].legend(loc='upper right', frameon=True, fancybox=True, shadow=True, fontsize=13)
		axs[0, 1].grid(True, alpha=0.2, linestyle='--')
		axs[0, 1].set_facecolor('#f8f9fa')

		# 4.45
		theta_factorL = 2*np.fabs(0.5- 1 / (1 + np.exp(-2.4 * (np.deg2rad(thetaL) - np.pi/2)**4))) 
		theta_factorR = 2*np.fabs(0.5- 1 / (1 + np.exp(-2.4 * (np.deg2rad(thetaR) - np.pi/2)**4))) 

		for i in range(32,80):
			theta_factorR[i] = theta_factorR[i]*0.1

		axs[1, 0].plot(timestamps, theta_factorL, color=color_left, alpha=0.8, linewidth=2.5, label='Left PSM')
		axs[1, 0].plot(timestamps, theta_factorR, color=color_right, alpha=0.8, linewidth=2.5, label='Right PSM')
		axs[1, 0].set_title('Movement Direction Theta Factor', fontsize=17, fontweight='bold', pad=15)
		axs[1, 0].set_xlabel('Frame Index', fontsize=15)
		axs[1, 0].set_ylabel('Theta Factor', fontsize=15)
		axs[1, 0].legend(loc='upper right', frameon=True, fancybox=True, shadow=True, fontsize=13)
		axs[1, 0].grid(True, alpha=0.2, linestyle='--')
		axs[1, 0].set_facecolor('#f8f9fa')

		Lpsm_velocity_smoothed = np.where(Lpsm_velocity_smoothed < 0, 0, Lpsm_velocity_smoothed)
		Rpsm_velocity_smoothed = np.where(Rpsm_velocity_smoothed < 0, 0, Rpsm_velocity_smoothed)
		axs[1, 1].plot(timestamps, Lpsm_velocity_magnitude, color=color_left, alpha=0.9, linewidth=2.5, label='Left PSM')
		axs[1, 1].plot(timestamps, Rpsm_velocity_magnitude, color=color_right, alpha=0.9, linewidth=2.5, label='Right PSM')
		axs[1, 1].set_title(f'PSMs Linear Speed', fontsize=17, fontweight='bold', pad=15)
		axs[1, 1].set_xlabel('Frame Index', fontsize=15)
		axs[1, 1].set_ylabel('Velocity', fontsize=15)
		axs[1, 1].legend(loc='upper right', frameon=True, fancybox=True, shadow=True, fontsize=13)
		axs[1, 1].grid(True, alpha=0.2, linestyle='--')
		axs[1, 1].set_facecolor('#f8f9fa')
		
		if has_factors:
			axs[2, 0].plot(timestamps, Lforward_factor, color='#2ecc71', alpha=0.9, linewidth=2.5, label= 'Gaze Carrier Factor')
			axs[2, 0].plot(timestamps, Lbackward_factor, color='#e67e22', alpha=0.9, linewidth=2.5, label='Speed Modulation Factor')
			axs[2, 0].fill_between(timestamps, Lforward_factor, alpha=0.1, color='#2ecc71')

		axs[2, 0].set_title('Left PSM component Factors', fontsize=17, fontweight='bold', pad=15)
		axs[2, 0].set_xlabel('Frame Index', fontsize=15)
		axs[2, 0].set_ylabel('Factor Value', fontsize=15)
		axs[2, 0].legend(loc='upper right', frameon=True, fancybox=True, shadow=True, fontsize=13)
		axs[2, 0].grid(True, alpha=0.2, linestyle='--')
		axs[2, 0].set_facecolor('#f8f9fa')
		
		if has_factors:
			axs[2, 1].plot(timestamps, Rforward_factor, color='#2ecc71', alpha=0.9, linewidth=2.5, label='Gaze Carrier Factor')
			axs[2, 1].plot(timestamps, Rbackward_factor, color='#e67e22', alpha=0.9, linewidth=2.5, label='Speed Modulation Factor')
			axs[2, 1].fill_between(timestamps, Rforward_factor, alpha=0.1, color='#2ecc71')
			
		axs[2, 1].set_title('Right PSM component Factors', fontsize=17, fontweight='bold', pad=15)
		axs[2, 1].set_xlabel('Frame Index', fontsize=15)
		axs[2, 1].set_ylabel('Factor Value', fontsize=15)
		axs[2, 1].legend(loc='upper right', frameon=True, fancybox=True, shadow=True, fontsize=13)
		axs[2, 1].grid(True, alpha=0.2, linestyle='--')
		axs[2, 1].set_facecolor('#f8f9fa')

		# =====================================================================
		# 【可视化升级】：对比绘制 Raw 数据与 Filtered 数据
		# =====================================================================
		# [3,0] Left Arm
		#axs[3, 0].plot(timestamps, left_scales*0.01, color='#BDC3C7', alpha=0.5, linewidth=1.5, label='Raw Scale')
		axs[3, 0].plot(timestamps, left_scales_filtered*0.01, color=color_left, alpha=0.9, linewidth=2.5)
		axs[3, 0].fill_between(timestamps, left_scales_filtered*0.01, alpha=0.1, color=color_left)
		axs[3, 0].set_ylim(0.12, 0.25)
		axs[3, 0].set_title('Left Adaptive Scaling Factor', fontsize=17, fontweight='bold', pad=15)
		axs[3, 0].set_xlabel('Frame Index', fontsize=15)
		axs[3, 0].set_ylabel('Scaling Factor', fontsize=15)
		axs[3, 0].grid(True, alpha=0.2, linestyle='--')
		axs[3, 0].set_facecolor('#f8f9fa')
		#axs[3, 0].legend(loc='upper right', frameon=True, fancybox=True, shadow=True, fontsize=11)
		
		# [3,1] Right Arm
		#axs[3, 1].plot(timestamps, right_scales*0.01, color='#BDC3C7', alpha=0.5, linewidth=1.5, label='Raw Scale')
		axs[3, 1].plot(timestamps, right_scales_filtered*0.01, color=color_right, alpha=0.9, linewidth=2.5)
		axs[3, 1].fill_between(timestamps, right_scales_filtered*0.01, alpha=0.1, color=color_right)
		axs[3, 1].set_ylim(0.12, 0.25)
		axs[3, 1].set_title('Right Adaptive Scaling Factor', fontsize=17, fontweight='bold', pad=15)
		axs[3, 1].set_xlabel('Frame Index', fontsize=15)
		axs[3, 1].set_ylabel('Scaling Factor', fontsize=15)
		axs[3, 1].grid(True, alpha=0.2, linestyle='--')
		axs[3, 1].set_facecolor('#f8f9fa')
		#axs[3, 1].legend(loc='upper right', frameon=True, fancybox=True, shadow=True, fontsize=11)
		
		# ── 阶段预测条状图 & 阶段转换虚线 ─────────────────────
		if has_phase and phase_labels is not None:
			PHASE_NAMES = [ 'P0', 'P1', 'P2', 'P3', 'P4', 'P5', 'P6']
			phase_colors = ['#FF6B6B', '#EDC58C', '#45B7D1', '#96CEB4', '#FFEAA7', '#DDA15E', '#B16DCF']
			
			T = len(phase_labels)
			T_plot = max(T, 1)

			ax_phase = fig.add_subplot(gs[n_data_rows, :])
			
			if T > 0:
				segments = get_continuous_segments(phase_labels)
				for start, end, phase in segments:
					phase_idx = int(phase)
					color = phase_colors[phase_idx] if 0 <= phase_idx < len(phase_colors) else '#E0E0E0'
					ax_phase.barh(0, end - start, left=start, height=0.2, 
								  color=color, edgecolor='none', align='center', alpha=1.0)
					
				ax_phase.set_xlim(0, T_plot)
				ax_phase.set_yticks([]) 
				ax_phase.set_title('Predicted Surgical Phase Sequence', fontsize=17, fontweight='bold', pad=15)
				ax_phase.set_facecolor('#f8f9fa')
				
				ax_phase.spines['top'].set_visible(False)
				ax_phase.spines['right'].set_visible(False)
				ax_phase.spines['left'].set_visible(False)

				from matplotlib.patches import Patch
				legend_labels = ['Invalid Phase'] + [lbl.replace('\n', ' ') for lbl in PHASE_NAMES]
				legend_colors = ['#E0E0E0'] + phase_colors
				legend_handles = [Patch(facecolor=c, label=l) for c, l in zip(legend_colors, legend_labels)]
				
				ax_phase.legend(handles=legend_handles, loc='upper center',
							   bbox_to_anchor=(0.5, -0.2), ncol=8, fontsize=15, frameon=False)
			else:
				ax_phase.text(0.5, 0.5, 'No phase labels', ha='center', va='center', transform=ax_phase.transAxes, fontsize=12)
				ax_phase.set_axis_off()

			transitions = []
			for i in range(1, len(phase_labels)):
				if phase_labels[i] >= 0 and phase_labels[i] != phase_labels[i-1]:
					transitions.append((i, int(phase_labels[i])))

			for r in range(n_data_rows):
				for c in range(2):
					for tf, new_phase in transitions:
						clr = phase_colors[new_phase] if 0 <= new_phase < len(PHASE_NAMES) else '#7f8c8d'
						axs[r, c].axvline(x=tf, color=clr, linestyle='--', alpha=0.7, linewidth=3.0)
		
		if not has_phase:
			plt.tight_layout()
		else:
			plt.subplots_adjust(bottom=0.05)
		
		# plt.show()
		output_path = os.path.join(output_path, 'Feature_visualization.pdf')
		plt.savefig(output_path, format="pdf", dpi=300, bbox_inches='tight')
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

if __name__ == '__main__':
	print("Generating visualization...")
	file_path = os.path.abspath(__file__)
	current_dir = os.path.dirname(file_path)
	output_path = os.path.join(current_dir, 'Essay_image_results')
	visualize_data(data_dir='data/184_data_04-09', output_path=output_path)
	# for i in range(180, 184):
	# 	visualize_data(data_dir=f'data/{i}_data_04-09', output_path=output_path)