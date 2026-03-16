import os
import numpy as np
import matplotlib.pyplot as plt

class Config:
    resolution_x = 1920
    resolution_y = 1080
    gaze_filter_params = {
        'filter_params': {
            'window_seconds': 2.0,       # 热图时间窗口
            'jump_threshold': 0.15,      # 跳变门限 (屏幕占比)
            'velocity_threshold': 5.0,   # 速度门限
            'attention_threshold': 0.05, # 热图判定门限
            
            'max_outlier_confirm': 5,    # 连续出现5个异常点则判定为扫视转移
            'saccade_stability_limit': 0.05 # 异常点间距阈值；异常点在此范围内聚集，说明视线在新位置稳住了
        },
        'gaussian_kernel_sigma': 2.5
    }

config = Config()

class AttentionHeatmapGenerator:
    def __init__(self, heatmap_size=(108, 192)):
        self.heatmap_size = heatmap_size
        self.realtime_heatmap = np.zeros(heatmap_size)
        self.all_timestamps = []
        self.all_gaze_points = []  
        self.filtered_gaze_points = []
        self.filtered_indices = []
        self.params = config.gaze_filter_params['filter_params']
        
        # 异常点确认缓冲区
        self.outlier_buffer = []

    def update_realtime_heatmap(self, current_ts):
        """高斯核热图更新逻辑"""
        self.realtime_heatmap = np.zeros(self.heatmap_size)
        window_start = current_ts - (self.params['window_seconds'] * 1000000)
        
        pts = []
        for ts, gx, gy in self.filtered_gaze_points:
            if ts >= window_start:
                xh = int(gx * (self.heatmap_size[1] - 1))
                yh = int(gy * (self.heatmap_size[0] - 1))
                pts.append((max(0, min(xh, self.heatmap_size[1]-1)), 
                            max(0, min(yh, self.heatmap_size[0]-1))))
        
        if not pts: return
        
        sigma = config.gaze_filter_params['gaussian_kernel_sigma']
        for x, y in pts:
            size = int(3 * sigma) + 1
            x_grid, y_grid = np.meshgrid(np.arange(-size, size+1), np.arange(-size, size+1))
            gaussian = np.exp(-(x_grid**2 + y_grid**2) / (2 * sigma**2))
            
            x_s, x_e = max(0, x-size), min(self.heatmap_size[1], x+size+1)
            y_s, y_e = max(0, y-size), min(self.heatmap_size[0], y+size+1)
            gx_s, gy_s = max(0, size-(x-x_s)), max(0, size-(y-y_s))
            gx_e, gy_e = gx_s + (x_e-x_s), gy_s + (y_e-y_s)
            self.realtime_heatmap[y_s:y_e, x_s:x_e] += gaussian[gy_s:gy_e, gx_s:gx_e]
        
        if np.max(self.realtime_heatmap) > 0:
            self.realtime_heatmap /= np.max(self.realtime_heatmap)

    def filter_outliers_focused(self, ts, gx, gy):
        """区分噪点与视线转移"""
        # 0. 初始阶段直接放行
        if len(self.all_gaze_points) < 10:
            return True

        # 1. 基础异常判定
        prev_valid = self.all_gaze_points[-1]
        dist = np.sqrt((gx - prev_valid[1])**2 + (gy - prev_valid[2])**2)
        jump_detected = dist > self.params['jump_threshold']
        
        xh = max(0, min(int(gx*(self.heatmap_size[1]-1)), self.heatmap_size[1]-1))
        yh = max(0, min(int(gy*(self.heatmap_size[0]-1)), self.heatmap_size[0]-1))
        heat_valid = self.realtime_heatmap[yh, xh] >= self.params['attention_threshold']
        
        is_initially_outlier = jump_detected or (not heat_valid)

        # 2. 扫视确认逻辑
        if is_initially_outlier:
            self.outlier_buffer.append((ts, gx, gy))
            
            # 如果异常点连续出现多帧
            if len(self.outlier_buffer) >= self.params['max_outlier_confirm']:
                # 检查这些异常点是否彼此接近（说明视线在新位置停稳了）
                buf_pts = self.outlier_buffer[-3:]
                move_range = np.sqrt((buf_pts[-1][1] - buf_pts[0][1])**2 + (buf_pts[-1][2] - buf_pts[0][2])**2)
                
                if move_range < self.params['saccade_stability_limit']:
                    # 【判定为扫视/视线转移】清空缓冲，强制通过
                    self.outlier_buffer = []
                    return True 
                else:
                    return False # 仍在剧烈运动中
            else:
                return False # 异常帧数不够，维持原位
        else:
            self.outlier_buffer = [] # 正常的点，重置缓冲
            return True


# 绘图
def generate_evaluation_plot(data_path, save_path, subdir_name):
    raw_data = np.load(os.path.join(data_path, 'gazepoint_position_data.npy'))
    # 列: [x_pixel, y_pixel, timestamp]
    raw_x = raw_data[:, 0] / config.resolution_x * 2
    raw_y = 1-raw_data[:, 1] / config.resolution_y * 2
    raw_ts = raw_data[:, 2]

    gen = AttentionHeatmapGenerator()
    out_x, out_y = [], []
    last_vx, last_vy = raw_x[0], raw_y[0]

    for i in range(len(raw_data)):
        ts = raw_ts[i]
        gen.update_realtime_heatmap(ts)
        if gen.filter_outliers_focused(ts, raw_x[i], raw_y[i]):
            last_vx, last_vy = raw_x[i], raw_y[i]
            gen.filtered_gaze_points.append((ts, raw_x[i], raw_y[i]))
            gen.filtered_indices.append(i)
        out_x.append(last_vx); out_y.append(last_vy)
        gen.all_timestamps.append(ts)
        gen.all_gaze_points.append((ts, raw_x[i], raw_y[i]))

    times = (raw_ts - raw_ts[0]) / 1000000.0
    out_x, out_y = np.array(out_x), np.array(out_y)

    fig = plt.figure(figsize=(20, 12))
    gs = fig.add_gridspec(2, 2)
    ax_x, ax_y, ax_2d = fig.add_subplot(gs[0,0]), fig.add_subplot(gs[1,0]), fig.add_subplot(gs[:,1])

    c_raw, c_filt, alpha = '#4682B4', 'black', 0.7

    # 子图1: X Temporal
    ax_x.plot(times, raw_x * config.resolution_x, color=c_raw, alpha=alpha, label='Raw', linewidth=0.7)
    ax_x.plot(times, out_x * config.resolution_x, color=c_filt, linestyle='-', label='Filtered', linewidth=1.5)
    ax_x.set_title(f"X-Coordinate Saccade-Aware Filter: {subdir_name}")
    ax_x.set_ylabel("Pixel X")
    ax_x.legend(loc='upper right', fontsize='small')
    ax_x.grid(True, alpha=0.2)

    # 子图2: Y Temporal
    ax_y.plot(times, raw_y * config.resolution_y, color=c_raw, alpha=alpha, label='Raw', linewidth=0.7)
    ax_y.plot(times, out_y * config.resolution_y, color=c_filt, linestyle='-', label='Filtered', linewidth=1.5)
    ax_y.set_xlabel("Time (s)"); ax_y.set_ylabel("Pixel Y")
    ax_y.legend(loc='upper right', fontsize='small')
    ax_y.grid(True, alpha=0.2)

    # 子图3: 2D Spatial
    ax_2d.scatter(raw_x * config.resolution_x, raw_y * config.resolution_y,
                  color=c_raw, s=2, alpha=0.2, label='Raw')
    ax_2d.scatter(out_x[gen.filtered_indices] * config.resolution_x,
                  out_y[gen.filtered_indices] * config.resolution_y,
                  color='#008080', s=3, label='Valid (Filtered)')
    ax_2d.set_xlim(0, config.resolution_x); ax_2d.set_ylim(config.resolution_y, 0)
    ax_2d.set_title("2D Spatial Audit")
    leg = ax_2d.legend(loc='upper right', markerscale=5)
    handles = leg.legend_handles if hasattr(leg, 'legend_handles') else leg.legendHandles
    for lh in handles: lh.set_alpha(1.0)

    plt.suptitle(f"Filter Audit | Session Duration: {times[-1]:.1f}s", fontsize=16)
    plt.tight_layout()
    plt.savefig(save_path, dpi=120); plt.close()


def process_all_data(num_to_process=None):
    current_dir = os.path.dirname(os.path.abspath(__file__))
    data_base_dir = os.path.join(current_dir, 'data')
    output_dir = os.path.join(current_dir, 'visualization_filter')
    if not os.path.exists(output_dir): os.makedirs(output_dir)

    subdirs = sorted([d for d in os.listdir(data_base_dir) if os.path.isdir(os.path.join(data_base_dir, d))], 
                     key=lambda x: int(x.split('_')[0]) if x.split('_')[0].isdigit() else 999)

    dirs_to_run = subdirs[:num_to_process+1] if num_to_process else subdirs
    print(f"模式: 处理前 {len(dirs_to_run)} 个")

    for subdir in dirs_to_run:
        path = os.path.join(data_base_dir, subdir)
        save_file = os.path.join(output_dir, f"{subdir}_filter.png")
        if os.path.exists(os.path.join(path, 'gazepoint_position_data.npy')):
            print(f"--> Processing {subdir}...")
            generate_evaluation_plot(path, save_file, subdir)

if __name__ == "__main__":
    process_all_data(num_to_process=147)