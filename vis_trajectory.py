import numpy as np
import matplotlib.pyplot as plt
import os

# ── 学术级样式设置 ──────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "axes.labelsize": 12,
    "axes.titlesize": 15,
    "font.size": 11,
    "legend.fontsize": 11,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
})

# 定义左右臂的专属学术配色 (延续之前的莫兰迪色系)
COLOR_LEFT = '#5681B9'   # 柔蓝色 (Left Arm)
COLOR_RIGHT = '#E18283'  # 柔红色 (Right Arm)

def load_trajectory_from_npy(file_path):
    """
    读取 .npy 格式的轨迹数据。
    针对 shape 为 (N, 4) 的数据，提取其中代表 X, Y, Z 的三列。
    """
    if not os.path.exists(file_path):
        print(f"⚠️ 警告: 找不到文件 {file_path}")
        return None
        
    data = np.load(file_path)
    
    # 提取前三列 [X, Y, Z]
    coords_3d = data[:, :3] 
    return coords_3d

def plot_3d_trajectories(traj_data_dict, save_path="3D_Trajectories_Comparison.png"):
    """
    绘制 1x3 的三维轨迹子图，强制统一坐标轴刻度，并优化 Z 轴和整体布局。
    """
    modes = ['Fixed Mode', 'Adaptive Mode', 'Phased Adaptive Mode']
    
    # =========================================================
    # 计算所有轨迹在 X, Y, Z 轴上的全局最小和最大值
    # =========================================================
    all_x, all_y, all_z = [], [], []
    for mode_name in modes:
        if mode_name in traj_data_dict:
            left_traj, right_traj = traj_data_dict[mode_name]
            if left_traj is not None and len(left_traj) > 0:
                all_x.extend(left_traj[:, 0])
                all_y.extend(left_traj[:, 1])
                all_z.extend(left_traj[:, 2])
            if right_traj is not None and len(right_traj) > 0:
                all_x.extend(right_traj[:, 0])
                all_y.extend(right_traj[:, 1])
                all_z.extend(right_traj[:, 2])
                
    if all_x and all_y and all_z:
        x_min, x_max = np.min(all_x), np.max(all_x)
        y_min, y_max = np.min(all_y), np.max(all_y)
        z_min, z_max = np.min(all_z), np.max(all_z)
        
        # 给坐标边界增加 5% 的 Padding 留白
        x_pad = (x_max - x_min) * 0.05
        y_pad = (y_max - y_min) * 0.05
        z_pad = (z_max - z_min) * 0.05
        
        global_xlim = (x_min - x_pad, x_max + x_pad)
        global_ylim = (y_min - y_pad, y_max + y_pad)
        global_zlim = (z_min - z_pad, z_max + z_pad)
    else:
        global_xlim = global_ylim = global_zlim = None

    # 【布局优化 1】：稍微缩短画幅宽度，让 1x3 布局更加紧凑
    fig = plt.figure(figsize=(16, 6))
    
    for i, mode_name in enumerate(modes):
        if mode_name not in traj_data_dict:
            continue
            
        left_traj, right_traj = traj_data_dict[mode_name]
        ax = fig.add_subplot(1, 3, i + 1, projection='3d')
        
        # 【布局优化 2】：调整 3D 框的比例，微微压缩 Z 轴高度，给标签腾出空间
        ax.set_box_aspect([1, 1, 0.8])
        
        # 1. 绘制左臂轨迹
        if left_traj is not None and len(left_traj) > 0:
            ax.plot(left_traj[:, 0], left_traj[:, 1], left_traj[:, 2], 
                    color=COLOR_LEFT, linewidth=3, alpha=0.85, label='Left Arm')
            ax.scatter(left_traj[0, 0], left_traj[0, 1], left_traj[0, 2], 
                       color=COLOR_LEFT, s=40, marker='o', edgecolor='black', linewidth=0.6, zorder=5)
            ax.scatter(left_traj[-1, 0], left_traj[-1, 1], left_traj[-1, 2], 
                       color=COLOR_LEFT, s=40, marker='s', edgecolor='black', linewidth=0.6, zorder=5)

        # 2. 绘制右臂轨迹
        if right_traj is not None and len(right_traj) > 0:
            ax.plot(right_traj[:, 0], right_traj[:, 1], right_traj[:, 2], 
                    color=COLOR_RIGHT, linewidth=3, alpha=0.85, label='Right Arm')
            ax.scatter(right_traj[0, 0], right_traj[0, 1], right_traj[0, 2], 
                       color=COLOR_RIGHT, s=40, marker='o', edgecolor='black', linewidth=0.6, zorder=5)
            ax.scatter(right_traj[-1, 0], right_traj[-1, 1], right_traj[-1, 2], 
                       color=COLOR_RIGHT, s=40, marker='s', edgecolor='black', linewidth=0.6, zorder=5)

        # 应用统一坐标轴限制
        if global_xlim and global_ylim and global_zlim:
            ax.set_xlim(global_xlim)
            ax.set_ylim(global_ylim)
            ax.set_zlim(global_zlim)

        # 3. 装饰与美化
        ax.set_title(mode_name, fontweight='bold', pad=15)
        
        # # 【布局优化 3】：坐标轴标签间距与旋转控制
        # ax.set_xlabel('X axis', labelpad=10)
        # ax.set_ylabel('Y axis', labelpad=10)
        
        # 强制 Z 轴标签垂直显示，并拉开与数字的距离
        # ax.zaxis.set_rotate_label(False) 
        # ax.set_zlabel('Z axis', labelpad=15, rotation=90)
        ax.tick_params(axis='z', pad=5)
        
        # 透明化背景面板
        ax.xaxis.pane.fill = False
        ax.yaxis.pane.fill = False
        ax.zaxis.pane.fill = False
        ax.xaxis._axinfo["grid"].update({'linestyle': '--', 'color': '#CCCCCC'})
        ax.yaxis._axinfo["grid"].update({'linestyle': '--', 'color': '#CCCCCC'})
        ax.zaxis._axinfo["grid"].update({'linestyle': '--', 'color': '#CCCCCC'})

        # 视角设置：azim=135 时，Z 轴通常显示在右侧。若想显示在左侧，可改为 azim=-45 或 45
        ax.view_init(elev=25, azim=135)
        
        # 仅在第一个子图显示图例
        if i == 0:
            ax.legend(loc='upper left', frameon=False, bbox_to_anchor=(0.0, 1.05))

    # 【布局优化 4】：使用 precise 的间距调整代替 tight_layout，拉近子图距离
    plt.subplots_adjust(left=0.05, right=0.95, bottom=0.1, top=0.9, wspace=0.08)

    # 保存文件
    os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"🎉 3D 轨迹对比图（排版优化版）已成功生成并保存至: {save_path}")

if __name__ == "__main__":
    base_dir = os.path.join(os.path.dirname(__file__), "data")

    fixed_demo_path = os.path.join(base_dir, "106_data_04-03")
    adaptive_demo_path = os.path.join(base_dir, "184_data_04-09") 
    phased_adaptive_demo_path = os.path.join(base_dir, "165_data_04-09") 
    
    file_paths = {
        'Fixed Mode': {
            'left':  os.path.join(fixed_demo_path, "Lpsm_position.npy"),
            'right': os.path.join(fixed_demo_path, "Rpsm_position.npy")
        },
        'Adaptive Mode': {
            'left':  os.path.join(adaptive_demo_path, "Lpsm_position.npy"),
            'right': os.path.join(adaptive_demo_path, "Rpsm_position.npy")
        },
        'Phased Adaptive Mode': {
            'left':  os.path.join(phased_adaptive_demo_path, "Lpsm_position.npy"),
            'right': os.path.join(phased_adaptive_demo_path, "Rpsm_position.npy")
        }
    }

    trajectory_dataset = {}
    for mode, paths in file_paths.items():
        left_data = load_trajectory_from_npy(paths['left'])
        right_data = load_trajectory_from_npy(paths['right'])
        trajectory_dataset[mode] = (left_data, right_data)
    
    file_path = os.path.join(os.path.dirname(__file__))
    output_filepath = os.path.join(file_path, "Essay_image_results", "3D_Trajectories_Comparison.png")
    plot_3d_trajectories(trajectory_dataset, save_path=output_filepath)





