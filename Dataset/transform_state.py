import numpy as np
import os

from collections import deque
from scipy.signal import savgol_filter

import matplotlib.pyplot as plt

class RealTimeVelocityFilter:
    """
    专门用于手术机器人轨迹速度平滑的实时 Savitzky-Golay 滤波器。
    提炼自 featureFilter.py 与 main.py。
    """
    def __init__(self, window_length=15, polyorder=3):
        # 严格遵守提供的参数逻辑：窗口长度必须为奇数，且大于多项式阶数
        if window_length % 2 == 0:
            raise ValueError('window_length must be odd')
        if polyorder >= window_length:
            raise ValueError('polyorder must be less than window_length')
            
        self.window_length = window_length
        self.polyorder = polyorder
        # 使用 deque 存储实时流入的样本数据
        self.buffer = deque(maxlen=window_length)

    def update(self, sample):
        """
        输入当前时刻的原始速度，输出平滑后的速度。
        """
        self.buffer.append(sample)

        # 初始阶段：如果数据点不足以支撑多项式拟合，直接返回原始值
        if len(self.buffer) <= self.polyorder + 1:
            return sample

        # 动态窗口处理：在缓冲区填满前，使用当前可用的最大奇数窗口
        if len(self.buffer) < self.window_length:
            curr_window = len(self.buffer) if len(self.buffer) % 2 == 1 else len(self.buffer) - 1
        else:
            curr_window = self.window_length

        # 执行 SG 滤波
        # mode='interp' 允许对窗口末端点进行多项式插值估计
        filtered_window = savgol_filter(
            np.array(self.buffer),
            window_length=curr_window,
            polyorder=self.polyorder,
            mode='interp',
        )
        
        # 核心：返回拟合序列的最后一个值，实现实时无延迟输出
        return filtered_window[-1]


    


def transform_state():
    current_dir = os.path.dirname(__file__) 
    state_path = os.path.join(current_dir, os.pardir, 'data')
    for demo_id in os.listdir(state_path):
        demo_dir = os.path.join(state_path, demo_id)
        left_pose = np.load(os.path.join(demo_dir, 'Lpsm_pose.npy'))
        right_pose = np.load(os.path.join(demo_dir, 'Rpsm_pose.npy'))
        left_gripper_state = np.load(os.path.join(demo_dir, 'Lgripper_state.npy'))
        right_gripper_state = np.load(os.path.join(demo_dir, 'Rgripper_state.npy'))
        
        min_len = np.min([len(left_pose), len(right_pose), len(left_gripper_state), len(right_gripper_state)])
        left_pose = left_pose[:min_len]
        right_pose = right_pose[:min_len]
        left_gripper_state = left_gripper_state[:min_len]
        right_gripper_state = right_gripper_state[:min_len]
        
        # Feature Engineering
        left_position = left_pose[:, :3]
        right_position = right_pose[:, :3]

        v_filter = RealTimeVelocityFilter(window_length=9, polyorder=5)
        left_velocity3_raw = np.diff(left_position, axis=0, prepend=left_position[:1, :])
        right_velocity3_raw = np.diff(right_position, axis=0, prepend=right_position[:1, :])
        left_velocity_raw = np.linalg.norm(left_velocity3_raw, axis=1)
        right_velocity_raw = np.linalg.norm(right_velocity3_raw, axis=1)

        l_filters = [RealTimeVelocityFilter(window_length=9, polyorder=3) for _ in range(3)]
        r_filters = [RealTimeVelocityFilter(window_length=9, polyorder=3) for _ in range(3)]
        left_velocity3  = np.array([[f.update(v) for f, v in zip(l_filters, row)] for row in left_velocity3_raw])
        right_velocity3 = np.array([[f.update(v) for f, v in zip(r_filters, row)] for row in right_velocity3_raw])


        left_velocity = []
        right_velocity = []
        for v_raw in left_velocity_raw:
            v_smooth = v_filter.update(v_raw)
            if v_smooth < 0:
                v_smooth = 0
            left_velocity.append(v_smooth)
        for v_raw in right_velocity_raw:
            v_smooth = v_filter.update(v_raw)
            if v_smooth < 0:
                v_smooth = 0
            right_velocity.append(v_smooth)
        left_velocity = np.array(left_velocity)
        right_velocity = np.array(right_velocity)
        
        
        demo_state = []

        for i in range(min_len):
            demo_state.append(np.hstack((left_position[i], left_velocity3[i], left_velocity[i], left_gripper_state[i], right_position[i], right_velocity3[i], right_velocity[i], right_gripper_state[i])))
            
        demo_state = np.array(demo_state)
        # create and save to txt

        demo_id_num = demo_id.split('_')[0]
        with open(os.path.join(current_dir, 'state', f'{demo_id_num}.txt'), 'w') as f:
            for state in demo_state:
                f.write(' '.join(map(str, state)) + '\n')

    return   

def visualize_state(demo_id):
    current_dir = os.path.dirname(__file__) 
    state_path = os.path.join(current_dir, os.pardir, 'data')
    
    demo_dir = os.path.join(state_path, demo_id)
    left_pose = np.load(os.path.join(demo_dir, 'Lpsm_pose.npy'))
    right_pose = np.load(os.path.join(demo_dir, 'Rpsm_pose.npy'))
    left_gripper_state = np.load(os.path.join(demo_dir, 'Lgripper_state.npy'))
    right_gripper_state = np.load(os.path.join(demo_dir, 'Rgripper_state.npy'))
    print(f"left_pose length: {len(left_pose)}")
    print(f"right_pose length: {len(right_pose)}")
    print(f"left_gripper_state length: {len(left_gripper_state)}")
    print(f"right_gripper_state length: {len(right_gripper_state)}")
    
    min_len = np.min([len(left_pose), len(right_pose), len(left_gripper_state), len(right_gripper_state)])
    print(f"min_len: {min_len}")
    left_pose = left_pose[:min_len]
    right_pose = right_pose[:min_len]
    left_gripper_state = left_gripper_state[:min_len]
    right_gripper_state = right_gripper_state[:min_len]
    
    # Feature Engineering
    left_position = left_pose[:, :3]
    right_position = right_pose[:, :3]

    v_filter = RealTimeVelocityFilter(window_length=9, polyorder=5)
    left_velocity3_raw = np.diff(left_position, axis=0, prepend=left_position[:1, :])
    right_velocity3_raw = np.diff(right_position, axis=0, prepend=right_position[:1, :])
    left_velocity_raw = np.linalg.norm(left_velocity3_raw, axis=1)
    right_velocity_raw = np.linalg.norm(right_velocity3_raw, axis=1)

    l_filters = [RealTimeVelocityFilter(window_length=9, polyorder=3) for _ in range(3)]
    r_filters = [RealTimeVelocityFilter(window_length=9, polyorder=3) for _ in range(3)]
    left_velocity3  = np.array([[f.update(v) for f, v in zip(l_filters, row)] for row in left_velocity3_raw])
    right_velocity3 = np.array([[f.update(v) for f, v in zip(r_filters, row)] for row in right_velocity3_raw])

    left_velocity = []
    right_velocity = []
    for v_raw in left_velocity_raw:
        v_smooth = v_filter.update(v_raw)
        if v_smooth < 0:
            v_smooth = 0
        left_velocity.append(v_smooth)
    for v_raw in right_velocity_raw:
        v_smooth = v_filter.update(v_raw)
        if v_smooth < 0:
            v_smooth = 0
        right_velocity.append(v_smooth)
    left_velocity = np.array(left_velocity)
    right_velocity = np.array(right_velocity)

    t_full = np.arange(min_len)

    fig, axes = plt.subplots(4, 2, figsize=(14, 13))
    fig.suptitle(f'State Visualization — {demo_id}', fontsize=14, fontweight='bold')

    # ── 左手 Position ──
    ax = axes[0, 0]
    ax.plot(t_full, left_position[:, 0], label='x', color='tab:red')
    ax.plot(t_full, left_position[:, 1], label='y', color='tab:green')
    ax.plot(t_full, left_position[:, 2], label='z', color='tab:blue')
    ax.set_title('Left Hand — Position')
    ax.set_ylabel('Position (m)')
    ax.legend(loc='upper right', fontsize=8)
    ax.grid(True, alpha=0.3)

    # ── 右手 Position ──
    ax = axes[0, 1]
    ax.plot(t_full, right_position[:, 0], label='x', color='tab:red')
    ax.plot(t_full, right_position[:, 1], label='y', color='tab:green')
    ax.plot(t_full, right_position[:, 2], label='z', color='tab:blue')
    ax.set_title('Right Hand — Position')
    ax.set_ylabel('Position (m)')
    ax.legend(loc='upper right', fontsize=8)
    ax.grid(True, alpha=0.3)

    # ── 左手 Velocity3 (vx/vy/vz) ──
    ax = axes[1, 0]
    ax.plot(t_full, left_velocity3_raw[:, 0], color='tab:red',   linewidth=0.6, alpha=0.3)
    ax.plot(t_full, left_velocity3_raw[:, 1], color='tab:green', linewidth=0.6, alpha=0.3)
    ax.plot(t_full, left_velocity3_raw[:, 2], color='tab:blue',  linewidth=0.6, alpha=0.3)
    ax.plot(t_full, left_velocity3[:, 0], label='vx', color='tab:red',   linewidth=0.8)
    ax.plot(t_full, left_velocity3[:, 1], label='vy', color='tab:green', linewidth=0.8)
    ax.plot(t_full, left_velocity3[:, 2], label='vz', color='tab:blue',  linewidth=0.8)
    ax.axhline(0, color='k', linewidth=0.5, linestyle='--')
    ax.set_title('Left Hand — Velocity3 (vx/vy/vz)')
    ax.set_ylabel('Velocity (m/frame)')
    ax.legend(loc='upper right', fontsize=8)
    ax.grid(True, alpha=0.3)

    # ── 右手 Velocity3 (vx/vy/vz) ──
    ax = axes[1, 1]
    ax.plot(t_full, right_velocity3_raw[:, 0], color='tab:red',   linewidth=0.6, alpha=0.3)
    ax.plot(t_full, right_velocity3_raw[:, 1], color='tab:green', linewidth=0.6, alpha=0.3)
    ax.plot(t_full, right_velocity3_raw[:, 2], color='tab:blue',  linewidth=0.6, alpha=0.3)
    ax.plot(t_full, right_velocity3[:, 0], label='vx', color='tab:red',   linewidth=0.8)
    ax.plot(t_full, right_velocity3[:, 1], label='vy', color='tab:green', linewidth=0.8)
    ax.plot(t_full, right_velocity3[:, 2], label='vz', color='tab:blue',  linewidth=0.8)
    ax.axhline(0, color='k', linewidth=0.5, linestyle='--')
    ax.set_title('Right Hand — Velocity3 (vx/vy/vz)')
    ax.set_ylabel('Velocity (m/frame)')
    ax.legend(loc='upper right', fontsize=8)
    ax.grid(True, alpha=0.3)

    # ── 左手 Velocity (scalar, smoothed) ──
    ax = axes[2, 0]
    ax.plot(t_full, left_velocity, color='tab:purple', linewidth=0.8)
    ax.set_title('Left Hand — Velocity (smoothed scalar)')
    ax.set_ylabel('Speed (m/frame)')
    ax.grid(True, alpha=0.3)

    # ── 右手 Velocity (scalar, smoothed) ──
    ax = axes[2, 1]
    ax.plot(t_full, right_velocity, color='tab:orange', linewidth=0.8)
    ax.set_title('Right Hand — Velocity (smoothed scalar)')
    ax.set_ylabel('Speed (m/frame)')
    ax.grid(True, alpha=0.3)

    # ── 左手 Gripper State ──
    ax = axes[3, 0]
    ax.plot(t_full, left_gripper_state, color='tab:brown', linewidth=0.8)
    ax.fill_between(t_full, left_gripper_state, alpha=0.3, color='tab:brown')
    ax.set_title('Left Hand — Gripper State')
    ax.set_ylabel('Gripper')
    ax.set_xlabel('Frame')
    ax.grid(True, alpha=0.3)

    # ── 右手 Gripper State ──
    ax = axes[3, 1]
    ax.plot(t_full, right_gripper_state, color='tab:cyan', linewidth=0.8)
    ax.fill_between(t_full, right_gripper_state, alpha=0.3, color='tab:cyan')
    ax.set_title('Right Hand — Gripper State')
    ax.set_ylabel('Gripper')
    ax.set_xlabel('Frame')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    save_dir = os.path.join(os.path.dirname(__file__), 'vis')
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f'{demo_id}_state_vis.png')
    plt.savefig(save_path, dpi=150)
    print(f'图像已保存至: {save_path}')
    plt.show()


if __name__ == '__main__':
    #transformed_state = transform_state()
    visualize_state('7_data_01-18')