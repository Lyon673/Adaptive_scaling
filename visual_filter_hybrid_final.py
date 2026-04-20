import os
import numpy as np
import matplotlib.pyplot as plt
import math
import cv2
from scipy.ndimage import gaussian_filter, label, center_of_mass

# =========================================================================
# 学术级全局字体与样式规范
# =========================================================================
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "axes.labelsize": 14,
    "axes.titlesize": 16,
    "font.size": 12,
    "legend.fontsize": 13,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
})

class Config:
    resolution_x = 960
    resolution_y = 540
    
    # 1080p 背景图片的路径 (请确保该图片位于同级目录)
    bg_image_path = 'background.png' 
    
    # 密度与平滑参数 
    gaze_filter_params = {
        'buffer_size': 30,
        'cluster_radius': 55.0,
        'min_density_ratio': 0.85, 
        'one_euro': {
            'min_cutoff': 1.0,
            'beta': 0.02,       
            'd_cutoff': 1.0
        }
    }
    
    # 聚类与走廊参数 
    cluster_params = {
        'grid_size': (540, 960),
        'quantile_threshold': 0.1,
        'sigma': 5,
        'corridor_thickness': 80,  
        'min_cluster_size': 500
    }
    
    max_jump_distance = 300 # 帧间最大允许跳变

config = Config()

# =========================================================================
# 核心算法类保持不变 (OneEuroFilter, GazeSpatialModel, HybridGazeFilter)
# =========================================================================
class OneEuroFilter:
    def __init__(self, t0, x0, min_cutoff=1.0, beta=0.0, d_cutoff=1.0):
        self.min_cutoff, self.beta, self.d_cutoff = float(min_cutoff), float(beta), float(d_cutoff)
        self.x_prev, self.dx_prev, self.t_prev = float(x0), 0.0, float(t0)

    def alpha(self, cutoff, dt):
        tau = 1.0 / (2 * math.pi * cutoff)
        return 1.0 / (1.0 + tau / dt)

    def apply(self, t, x):
        dt = t - self.t_prev
        if dt <= 0: return self.x_prev
        alpha_d = self.alpha(self.d_cutoff, dt)
        dx = (x - self.x_prev) / dt
        dx_hat = alpha_d * dx + (1.0 - alpha_d) * self.dx_prev
        cutoff = self.min_cutoff + self.beta * abs(dx_hat)
        alpha = self.alpha(cutoff, dt)
        x_hat = alpha * x + (1.0 - alpha) * self.x_prev
        self.x_prev, self.dx_prev, self.t_prev = x_hat, dx_hat, t
        return x_hat

class GazeSpatialModel:
    def __init__(self):
        self.mask = None
        self.grid_h, self.grid_w = config.cluster_params['grid_size']

    def train(self, training_paths):
        print("正在训练空间遮罩模型（识别操作区与转移走廊）...")
        combined_pts = []
        for path in training_paths:
            if not os.path.exists(path): continue
            data = np.load(path)
            if data.ndim < 2 or data.shape[0] == 0 or data.shape[1] < 2: continue
            valid = (data[:, 0] > 0) & (data[:, 0] < config.resolution_x) & \
                    (data[:, 1] > 0) & (data[:, 1] < config.resolution_y)
            if np.any(valid): combined_pts.append(data[valid, :2])
        
        all_pts = np.vstack(combined_pts)
        heatmap, _, _ = np.histogram2d(all_pts[:, 1], all_pts[:, 0], 
                                      bins=[self.grid_h, self.grid_w],
                                      range=[[0, config.resolution_y], [0, config.resolution_x]])
        heatmap = gaussian_filter(heatmap, sigma=config.cluster_params['sigma'])
        
        thresh = np.quantile(heatmap[heatmap > 0], 1 - config.cluster_params['quantile_threshold'])
        island_mask = (heatmap >= thresh).astype(np.uint8)
        
        labeled, num_features = label(island_mask)
        centroids = []
        final_mask = island_mask.copy()

        for i in range(1, num_features + 1):
            if np.sum(labeled == i) < config.cluster_params['min_cluster_size']:
                final_mask[labeled == i] = 0
                continue
            com = center_of_mass(labeled == i)
            centroids.append((int(com[1]), int(com[0])))

        for i in range(len(centroids)):
            for j in range(i + 1, len(centroids)):
                cv2.line(final_mask, centroids[i], centroids[j], 1, thickness=config.cluster_params['corridor_thickness'])
        
        self.mask = final_mask.astype(bool)

    def is_in_safe_zone(self, x, y):
        if self.mask is None: return True
        col, row = int(np.clip(x, 0, config.resolution_x-1)), int(np.clip(y, 0, config.resolution_y-1))
        return self.mask[row, col]

class HybridGazeFilter:
    def __init__(self, spatial_model):
        self.spatial_model = spatial_model
        self.point_buffer = []
        self.filter_x = None
        self.filter_y = None
        self.last_valid_out = None
        
        self.params = config.gaze_filter_params
        self.one_euro_cfg = config.gaze_filter_params['one_euro']

    def process(self, ts, gx, gy):
        current_pt = np.array([gx, gy])
        
        if not (0 < gx < config.resolution_x and 0 < gy < config.resolution_y):
            return self.last_valid_out, False

        in_safe_zone = self.spatial_model.is_in_safe_zone(gx, gy)
        
        self.point_buffer.append(current_pt)
        if len(self.point_buffer) > self.params['buffer_size']:
            self.point_buffer.pop(0)
            
        pts_array = np.array(self.point_buffer)
        dists = np.sqrt(np.sum((pts_array - current_pt)**2, axis=1))
        neighbor_count = np.sum(dists < self.params['cluster_radius'])
        density_ratio = neighbor_count / len(self.point_buffer)
        density_ok = density_ratio >= self.params['min_density_ratio']

        is_jump = False
        if self.last_valid_out is not None:
            dist_from_last = np.sqrt((gx - self.last_valid_out[0])**2 + (gy - self.last_valid_out[1])**2)
            if dist_from_last > config.max_jump_distance:
                is_jump = True

        if (density_ok) or (in_safe_zone and not is_jump):
            if self.filter_x is None:
                self.filter_x = OneEuroFilter(ts, gx, **self.one_euro_cfg)
                self.filter_y = OneEuroFilter(ts, gy, **self.one_euro_cfg)
                self.last_valid_out = (gx, gy)
            else:
                fx = self.filter_x.apply(ts, gx)
                fy = self.filter_y.apply(ts, gy)
                self.last_valid_out = (fx, fy)
            return self.last_valid_out, True
        else:
            return self.last_valid_out, False

# =========================================================================
# 高级视觉评估与渲染引擎
# =========================================================================
def generate_evaluation_plot(data_path, save_path, subdir_name, spatial_model):
    raw_data = np.load(os.path.join(data_path, 'gazepoint_position_data.npy'))
    raw_x, raw_y, raw_ts_raw = raw_data[:, 0], raw_data[:, 1], raw_data[:, 2]

    # 时间标准化
    t_diff = raw_ts_raw[-1] - raw_ts_raw[0]
    scale = 1000000.0 if t_diff > 100000 else (1000.0 if t_diff > 100 else 1.0)
    times = (raw_ts_raw - raw_ts_raw[0]) / scale

    h_filter = HybridGazeFilter(spatial_model)
    out_x, out_y, valid_mask = [], [], []

    for i in range(len(raw_data)):
        pos, is_v = h_filter.process(times[i], raw_x[i], raw_y[i])
        if pos is None: pos = (raw_x[0], raw_y[0])
        out_x.append(pos[0])
        out_y.append(pos[1])
        valid_mask.append(is_v)

    out_x, out_y, valid_mask = np.array(out_x), np.array(out_y), np.array(valid_mask)

    # ---------------------------------------------------------
    # 画布与子图布局配置 (增加总体宽度，优化比例)
    # ---------------------------------------------------------
    fig = plt.figure(figsize=(20, 6))
    # 赋予右侧图像略微更多的宽度权重，确保其拥有足够空间展示 16:9
    gs = fig.add_gridspec(2, 2, width_ratios=[1, 1.2], wspace=0.15, hspace=0.25)
    
    ax_x = fig.add_subplot(gs[0, 0])
    ax_y = fig.add_subplot(gs[1, 0])
    ax_2d = fig.add_subplot(gs[:, 1])

    # 柔和色彩定义
    COLOR_RAW = '#BDC3C7'      # 浅银灰色
    COLOR_FILTERED = '#2C3E50' # 深石板蓝
    COLOR_OUTLIER = '#E18283'  # 柔红色 (剔除点)
    COLOR_VALID = '#5681B9'    # 柔蓝色 (有效点)

    # 1. 绘制 X 轴时间序列
    ax_x.plot(times, raw_x, color=COLOR_RAW, alpha=0.6, linewidth=1.5, label='Raw Gaze Signal')
    ax_x.plot(times, out_x, color=COLOR_FILTERED, linewidth=2.0, label='Hybrid Filtered Signal')
    ax_x.set_title(f"X-Axis Tracking - {subdir_name}", fontweight='bold', pad=10)
    ax_x.set_ylabel("X Coordinate (Pixels)")
    ax_x.grid(axis='y', linestyle=':', alpha=0.6)
    
    # 图例修饰
    legend_x = ax_x.legend(loc='upper right', framealpha=0.9, edgecolor='#CCCCCC')
    
    # 2. 绘制 Y 轴时间序列
    ax_y.plot(times, raw_y, color=COLOR_RAW, alpha=0.6, linewidth=1.5)
    ax_y.plot(times, out_y, color=COLOR_FILTERED, linewidth=2.0)
    ax_y.invert_yaxis() # 图像坐标系 Y 轴向下
    ax_y.set_title("Y-Axis Tracking", fontweight='bold', pad=10)
    ax_y.set_xlabel("Time (Seconds)")
    ax_y.set_ylabel("Y Coordinate (Pixels)")
    ax_y.grid(axis='y', linestyle=':', alpha=0.6)
    
    for ax in [ax_x, ax_y]:
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    # ---------------------------------------------------------
    # 3. 右侧空间注视点叠加图 (强制锁定 16:9 比例)
    # ---------------------------------------------------------
    if os.path.exists(config.bg_image_path):
        bg_img = plt.imread(config.bg_image_path)
        # aspect='equal' 配合 960x540 的 extent，将在物理画布上强制锁定完美的 16:9 比例
        ax_2d.imshow(bg_img, extent=[0, config.resolution_x, config.resolution_y, 0], 
                     aspect='equal', alpha=0.85)
    else:
        print(f"\n⚠️ 提示: 未找到背景图片 '{config.bg_image_path}'。已回退至空间遮罩显示。")
        ax_2d.imshow(spatial_model.mask, extent=[0, config.resolution_x, config.resolution_y, 0], 
                     cmap='Blues', aspect='equal', alpha=0.1)

    # 散点渲染优化
    ax_2d.scatter(raw_x[~valid_mask][::2], 540-raw_y[~valid_mask][::2], c=COLOR_OUTLIER, s=15, alpha=0.5, 
                 linewidths=0.2, label='Filtered Outliers')
    ax_2d.scatter(out_x[::2], 540-out_y[::2], c=COLOR_VALID, s=15, alpha=0.5, 
                  linewidths=0.2, label='Valid Gaze Path')
    
    # 确保视窗边界完全贴合设定分辨率
    ax_2d.set_xlim(0, config.resolution_x)
    ax_2d.set_ylim(config.resolution_y, 0)
    ax_2d.set_title("Gaze Path Overlay on Operational Field", fontweight='bold', pad=15)
    ax_2d.set_xlabel("Screen Width (Pixels)")
    ax_2d.set_ylabel("Screen Height (Pixels)")
    
    # 防止边框干扰背景图的美观
    ax_2d.spines['top'].set_color('#DDDDDD')
    ax_2d.spines['right'].set_color('#DDDDDD')
    ax_2d.spines['left'].set_color('#DDDDDD')
    ax_2d.spines['bottom'].set_color('#DDDDDD')
    
    # 右侧图例加强底色，避免被背景图掩盖
    legend_2d = ax_2d.legend(loc='upper right', framealpha=0.95, facecolor='white', edgecolor='#CCCCCC')

    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches='tight', facecolor='white')
    plt.close()

def process_data(start_idx, end_idx):
    current_dir = os.path.dirname(os.path.abspath(__file__))
    data_base_dir = os.path.join(current_dir, 'dataPre')
    output_dir = os.path.join(current_dir, 'Essay_image_results')
    if not os.path.exists(output_dir): os.makedirs(output_dir)

    subdirs = sorted([d for d in os.listdir(data_base_dir) if os.path.isdir(os.path.join(data_base_dir, d))])
    
    spatial_model = GazeSpatialModel()
    train_paths = [os.path.join(data_base_dir, s, 'gazepoint_position_data.npy') for s in (subdirs[0:27] + subdirs[29:147])]
    spatial_model.train(train_paths)

    for subdir in subdirs[start_idx : end_idx + 1]:
        path = os.path.join(data_base_dir, subdir)
        if os.path.exists(os.path.join(path, 'gazepoint_position_data.npy')):
            print(f"Processing -> {subdir}...")
            save_file = os.path.join(output_dir, f"{subdir}_hybrid.png")
            generate_evaluation_plot(path, save_file, subdir, spatial_model)

if __name__ == "__main__":
    process_data(28, 28)