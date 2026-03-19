import json
from random import shuffle
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from scipy.interpolate import interp1d

from scipy.spatial.transform import Rotation as R
from scipy.spatial.transform import Slerp

import config

# ==================== 数据集划分配置 ====================
# 设置随机种子以确保可重复性
RANDOM_SEED = 32
np.random.seed(RANDOM_SEED)

demo_num = 148

# 生成并shuffle demo ID列表
demo_id_list = np.arange(demo_num)
demo_id_list = np.random.permutation(demo_id_list)

ratio = 1.0

def load_annotations(json_path):
    with open(json_path, 'r') as f:
        data = json.load(f)
    return data

def get_frame_label(annotation_data, frame_num):
    """
    return the label of the frame, if the frame is not in the annotation, return -1
    """
    for ann in annotation_data['annotations']:
        start = ann['kine_start']
        end = ann['kine_end'] 
        if start <= frame_num <= end:
            return int(ann['label'])
    return -1

def generate_frame_label_map(dir_path, demo_id):
    """
    return a list of labels for each frame , whose length is the same as the total number of frames
    """
    label_path = os.path.join(dir_path, 'label', f'{demo_id}_output_annotations.json')

    annotation_data = load_annotations(label_path)
    kine_frames = int(annotation_data['kine_frames'])
    frame_map = []
    for i in range(kine_frames):
        # frame_num = i * annotation_data['total_frames'] / kine_frames
        frame_map.append(get_frame_label(annotation_data, i))
    return frame_map


current_dir = os.path.dirname(__file__) 
needle_dir_path = os.path.join(current_dir, os.pardir, 'Dataset')


def read_demo_kinematics_state(dir_path, demo_id):
    kine_path = os.path.join(dir_path, 'state', f'{demo_id}.txt')
    content = []
    with open(kine_path, 'r', encoding='utf-8') as file:
        for line in file:
            info = line.strip().split()
            content.append(list(map(float, info)))
        
    content = np.array(content, dtype=np.float32)

    # deal with the 7th state (grasp state)

    # for i in range(content.shape[0]):
    #     if content[i, 6] < 0.5:
    #         content[i, 6] = 0.0
    #     else:
    #         content[i, 6] = 1.0
    #     if content[i, 13] < 0.5:
    #         content[i, 13] = 0.0
    #     else:
    #         content[i, 13] = 1.0

    # left: 3p 3v 1v 1g; right: 3p 3v 1v 1g
    # left_state = content[:, :8]
    # right_state = content[:, 8:16]

    #return np.hstack((left_state, right_state))
    return content

def resample_bimanual_trajectory(data, step_size=config.resample_step, target_length=None, without_quat=False):
    
    if without_quat:
        l_pos = data[:, :3]
        l_grip = data[:, 3]
        r_pos = data[:, 4:7]
        r_grip = data[:, 7]
    else:
        # 左手
        l_pos = data[:, :3]
        l_quat = data[:, 3:7] 
        l_grip = data[:, 7]
        
        # 右手
        r_pos = data[:, 8:11]
        r_quat = data[:, 11:15]
        r_grip = data[:, 15]

  

    combined_traj = np.hstack((l_pos, r_pos))
    
    # 2. 计算在 6D 空间中的每一步位移
    deltas = np.diff(combined_traj, axis=0)
    
    # axis=1 求范数，相当于 sqrt(dx_L^2 + ... + dz_R^2)
    # 这就是比简单相加更科学的"联合距离"
    dists = np.linalg.norm(deltas, axis=1)
    
    # 3. 计算累计进度 (Progress Variable)
    s_cumulative = np.concatenate(([0], np.cumsum(dists)))
    total_dist = s_cumulative[-1]
    
    # 3.5 处理重复值问题：Slerp要求严格递增
    # 找出唯一值的索引（保持顺序）
    _, unique_indices = np.unique(s_cumulative, return_index=True)
    unique_indices = np.sort(unique_indices)  # 确保顺序
    
    # 如果有重复值，使用唯一值创建插值
    s_unique = s_cumulative[unique_indices]
    l_pos_unique = l_pos[unique_indices]
    r_pos_unique = r_pos[unique_indices]
    if not without_quat:
        l_quat_unique = l_quat[unique_indices]
        r_quat_unique = r_quat[unique_indices]
    l_grip_unique = l_grip[unique_indices]
    r_grip_unique = r_grip[unique_indices]
    
    # 4. 生成新的均匀进度网格
    if target_length is not None:
        step_size = total_dist/target_length
    s_new = np.arange(0, total_dist, step_size)
    
    
    # 如果数据点太少，直接返回原始数据
    if len(s_unique) < 2:
        print("Warning: Not enough unique points for interpolation")
        return data
    
    # 左手插值函数
    f_left = interp1d(s_unique, l_pos_unique, axis=0, bounds_error=False, fill_value='extrapolate')
    new_l_pos = f_left(s_new)
    
    # 右手插值函数
    f_right = interp1d(s_unique, r_pos_unique, axis=0, bounds_error=False, fill_value='extrapolate')
    new_r_pos = f_right(s_new)

    if not without_quat:
        # 左手旋转插值
        l_rot_obj = R.from_quat(l_quat_unique)
        l_slerp = Slerp(s_unique, l_rot_obj)
        new_l_quat = l_slerp(s_new).as_quat() # 返回插值后的四元数
        
        # 右手旋转插值
        r_rot_obj = R.from_quat(r_quat_unique)
        r_slerp = Slerp(s_unique, r_rot_obj)
        new_r_quat = r_slerp(s_new).as_quat()

    # Gripper插值（最近邻，适合离散的0/1值）
    f_l_grip = interp1d(s_unique, l_grip_unique, kind='nearest', fill_value='extrapolate')
    new_l_grip = f_l_grip(s_new)
    
    f_r_grip = interp1d(s_unique, r_grip_unique, kind='nearest', fill_value='extrapolate')
    new_r_grip = f_r_grip(s_new)



    # --- 5. 拼接结果 ---
    # 结果顺序: [Time, L_Pos(3), L_Quat(4), L_Grip(1), R_Pos(3), R_Quat(4), R_Grip(1), Delta_T(1)]
    if not without_quat:
        resampled = np.column_stack((
            new_l_pos, new_l_quat, new_l_grip,
            new_r_pos, new_r_quat, new_r_grip,
        ))
    else:
        resampled = np.column_stack((
            new_l_pos, new_l_grip,
            new_r_pos, new_r_grip,
        ))
    
    return resampled


def resample_label(demo_id, step_size=config.resample_step, target_length=None):

    state = read_demo_kinematics_state(needle_dir_path, demo_id)
    #state = resample_bimanual_trajectory(state, without_quat=False)

    l_pos = state[:, :3]

    r_pos = state[:, 8:11]

    combined_traj = np.hstack((l_pos, r_pos))
    
    # 2. 计算在 6D 空间中的每一步位移
    deltas = np.diff(combined_traj, axis=0)
    
    # axis=1 求范数，相当于 sqrt(dx_L^2 + ... + dz_R^2)
    # 这就是比简单相加更科学的"联合距离"
    dists = np.linalg.norm(deltas, axis=1)
    
    # 3. 计算累计进度 (Progress Variable)
    s_cumulative = np.concatenate(([0], np.cumsum(dists)))
    total_dist = s_cumulative[-1]
    
    # 3.5 处理重复值问题：Slerp要求严格递增
    # 找出唯一值的索引（保持顺序）
    _, unique_indices = np.unique(s_cumulative, return_index=True)
    unique_indices = np.sort(unique_indices)  # 确保顺序

    label = np.array(generate_frame_label_map(needle_dir_path, demo_id))
    # print(f"<LYON> length label: {len(label)}")
    # print(f"<LYON> unique indices: {unique_indices}")

    label_unique = label[unique_indices]
    
    # 如果有重复值，使用唯一值创建插值
    s_unique = s_cumulative[unique_indices]
    
    
    # 4. 生成新的均匀进度网格
    if target_length is not None:
        step_size = total_dist/target_length
    s_new = np.arange(0, total_dist, step_size)

    f_label = interp1d(s_unique, label_unique, kind='nearest', fill_value='extrapolate')
    new_label = f_label(s_new)
    return new_label


def get_shuffled_demo_ids(shuffle=True, demo_id_list=None):
    """获取统一的demo ID顺序，确保state和label使用相同顺序"""
    np.random.seed(RANDOM_SEED)  # 每次重置种子确保一致性
    if demo_id_list is None:
        demo_id_list = np.arange(demo_num)
    if shuffle:
        demo_id_list = np.random.permutation(demo_id_list)

    # len_demo_id_list = len(demo_id_list)
    # bound = int(ratio*len_demo_id_list)

    # print("*****************************************")
    # print(f"test demo id :{demo_id_list[bound:]}")
    # print("*****************************************")
    return demo_id_list


def load_demonstrations_state(shuffle=True, without_quat=False, resample=False, demo_id_list=None):
    if resample:
        print("Resampling demonstrations!!!")
    demo_id_list = get_shuffled_demo_ids(shuffle, demo_id_list)
    demo_states = []
    for demo_id in demo_id_list:
        state = read_demo_kinematics_state(needle_dir_path, demo_id)
        if without_quat:
            state = state[:, [0,1,2,7,8,9,10,15]]
        if resample:
            state = resample_bimanual_trajectory(state, without_quat=without_quat)
            demo_states.append(state)
        else:
            # if state.shape[0] <= 350:
            #     demo_states.append(state)
            # else:
            #     print(f"too long demo idx:{demo_id}")
            demo_states.append(state)
            continue
        
        
    print(f"last demo id: {demo_id_list[-1]}")
    #print(f"demo_states shape: {np.array(demo_states).shape}")


    return demo_states


GRIPPER_COLS    = [7, 15]              # 不参与任何归一化，原样保留
VELSCALAR_COLS  = [6, 14]              # 速度模长 —— MinMaxScaler → [0, 1]
VEL3_COL_GROUPS = [[3,4,5], [11,12,13]]  # 左/右手 velocity3 —— SharedStdScaler
POS_COL_GROUPS  = [[0,1,2], [8,9,10]]    # 左/右手 position  —— SharedStdScaler


class SharedStdScaler:
    """
    对一组列使用相同的标准差进行缩放，均值仍逐列去中心化。

    这样 vx/vy/vz 三轴用同一把尺子缩放，轴间相对幅度比例完全保留，
    避免独立 StandardScaler 将各轴强制拉到相同方差从而抹消方向差异。
    """
    def __init__(self):
        self.means_ = None   # shape: (n_cols,)
        self.shared_std_ = None   # scalar

    def fit(self, X):
        # X: (N, n_cols)
        self.means_ = X.mean(axis=0)
        # 用所有列的全部数值联合计算一个共享标准差
        self.shared_std_ = (X - self.means_).std()
        if self.shared_std_ == 0:
            self.shared_std_ = 1.0
        return self

    def transform(self, X):
        return (X - self.means_) / self.shared_std_

    def fit_transform(self, X):
        return self.fit(X).transform(X)


def _scale_demos(demo_states):
    """
    四类列分别处理：
      - GRIPPER_COLS    : 不变
      - VELSCALAR_COLS  : MinMaxScaler  → [0, 1]
      - POS_COL_GROUPS  : SharedStdScaler（逐组，均值逐列，共享标准差）
      - VEL3_COL_GROUPS : SharedStdScaler（逐组，均值逐列，共享标准差）
    返回 (scaled_demos, scalers_dict)
    """
    stacked = np.vstack(demo_states)

    mm_scaler = MinMaxScaler()
    mm_scaler.fit(stacked[:, VELSCALAR_COLS])

    # position：左/右手各一个 SharedStdScaler
    pos_scalers = []
    for group in POS_COL_GROUPS:
        sc = SharedStdScaler()
        sc.fit(stacked[:, group])
        pos_scalers.append(sc)

    # velocity3：左/右手各一个 SharedStdScaler
    vel3_scalers = []
    for group in VEL3_COL_GROUPS:
        sc = SharedStdScaler()
        sc.fit(stacked[:, group])
        vel3_scalers.append(sc)

    scaled = []
    for arr in demo_states:
        out = arr.copy()
        out[:, VELSCALAR_COLS] = mm_scaler.transform(arr[:, VELSCALAR_COLS])
        for group, sc in zip(POS_COL_GROUPS, pos_scalers):
            out[:, group] = sc.transform(arr[:, group])
        for group, sc in zip(VEL3_COL_GROUPS, vel3_scalers):
            out[:, group] = sc.transform(arr[:, group])
        scaled.append(out)

    scalers = {
        'pos': pos_scalers,
        'vel_scalar': mm_scaler,
        'vel3': vel3_scalers,
    }
    return scaled, scalers

def load_train_state(shuffle=True, without_quat=False, resample=False, demo_id_list=None):
    demo_states_all = load_demonstrations_state(shuffle=shuffle, without_quat=without_quat, resample=resample, demo_id_list=demo_id_list)
    bound = round(ratio * len(demo_states_all))
    demo_states = demo_states_all[:bound]
    scaled_demos, _ = _scale_demos(demo_states)
    return scaled_demos

def load_test_state(shuffle=True, without_quat=False, resample=False, demo_id_list=None):
    demo_states_all = load_demonstrations_state(shuffle=shuffle, without_quat=without_quat, resample=resample, demo_id_list=demo_id_list)
    bound = round(ratio * len(demo_states_all))
    train_states = demo_states_all[:bound]
    test_states  = demo_states_all[bound:]

    # 用训练集 fit 的 scalers 变换测试集，保证分布一致
    _, scalers = _scale_demos(train_states)
    pos_scalers  = scalers['pos']
    mm_scaler    = scalers['vel_scalar']
    vel3_scalers = scalers['vel3']

    scaled = []
    for arr in test_states:
        out = arr.copy()
        out[:, VELSCALAR_COLS] = mm_scaler.transform(arr[:, VELSCALAR_COLS])
        for group, sc in zip(POS_COL_GROUPS, pos_scalers):
            out[:, group] = sc.transform(arr[:, group])
        for group, sc in zip(VEL3_COL_GROUPS, vel3_scalers):
            out[:, group] = sc.transform(arr[:, group])
        scaled.append(out)
    return scaled


def load_demonstrations_label(shuffle=True, resample=False, demo_id_list=None):
    demo_id_list = get_shuffled_demo_ids(shuffle, demo_id_list) 
    demo_labels = []
    for demo_id in demo_id_list:
        if resample:
            label = resample_label(demo_id=demo_id)
            demo_labels.append(label)
     
        else:
            label = generate_frame_label_map(needle_dir_path, demo_id)
            # if len(label) <= 350 and len(label) > 0:
            #     demo_labels.append(label)
            # else:
            #     print(f"error demo label idx:{demo_id}")
            #     continue
            demo_labels.append(label)
        
    return demo_labels


def load_train_label(resample=False, demo_id_list=None):
    """
    load the label of the train demos
    params: resample - True/False
    """
    # 159 / 138
    demo_labels_all = load_demonstrations_label(resample=resample, demo_id_list=demo_id_list)
    bound = round(ratio*len(demo_labels_all))
    demo_labels = demo_labels_all[:bound]
    return demo_labels

def load_test_label(resample=False, demo_id_list=None):
    demo_labels_all = load_demonstrations_label(resample=resample, demo_id_list=demo_id_list)
    bound = round(ratio*len(demo_labels_all))
    demo_labels = demo_labels_all[bound:]
    return demo_labels

def load_specific_test_state(shuffle=True, without_quat=False, resample=False, demo_id_list=None):
    demo_states_all = load_demonstrations_state(shuffle=shuffle, without_quat=without_quat, resample=resample)
    bound = round(ratio * len(demo_states_all))
    train_states = demo_states_all[:bound]

    # 用训练集 fit 的 scalers 变换测试集，保证分布一致
    _, scalers = _scale_demos(train_states)
    pos_scalers  = scalers['pos']
    mm_scaler    = scalers['vel_scalar']
    vel3_scalers = scalers['vel3']

    # 直接从文件读取指定 demo 的状态
    test_states = []
    if demo_id_list is not None:
        for did in demo_id_list:
            state = read_demo_kinematics_state(needle_dir_path, did)
            if without_quat:
                state = state[:, [0, 1, 2, 7, 8, 9, 10, 15]]
            if resample:
                state = resample_bimanual_trajectory(state, without_quat=without_quat)
            test_states.append(state)
    else:
        for state in demo_states_all:
            test_states.append(state)

    
    scaled = []
    for arr in test_states:
        out = arr.copy()
        out[:, VELSCALAR_COLS] = mm_scaler.transform(arr[:, VELSCALAR_COLS])
        for group, sc in zip(POS_COL_GROUPS, pos_scalers):
            out[:, group] = sc.transform(arr[:, group])
        for group, sc in zip(VEL3_COL_GROUPS, vel3_scalers):
            out[:, group] = sc.transform(arr[:, group])
        scaled.append(out)
    return scaled

def load_specific_test_label(demo_id_list):
    """
    按原始文件 ID 直接读取标注标签，不依赖文件内的其他函数。

    参数:
        demo_id_list: 可迭代的 demo 原始 ID，例如 [76, 77, 78, 79, 80]

    返回:
        list[list[int]]: 每个元素是对应 demo 的帧级标签列表，
                         未被标注的帧返回 -1。
    """
    _base       = os.path.join(os.path.dirname(__file__), os.pardir, 'Dataset')
    _label_dir  = os.path.join(_base, 'label')

    all_labels = []
    for did in demo_id_list:
        label_path = os.path.join(_label_dir, f'{did}_output_annotations.json')
        with open(label_path, 'r') as f:
            data = json.load(f)

        kine_frames   = int(data['kine_frames'])
        annotations   = data['annotations']

        frame_map = []
        for frame_idx in range(kine_frames):
            label = -1
            for ann in annotations:
                if ann['kine_start'] <= frame_idx <= ann['kine_end']:
                    label = int(ann['label'])
                    break
            frame_map.append(label)

        all_labels.append(frame_map)

    return all_labels


def visualize_demo_lengths():
    """可视化所有demo的时间长度（索引）"""
    # 加载数据
    demos = load_demonstrations_state()
    
    # 计算每个demo的长度
    demo_lengths = [len(demo) for demo in demos]
    
    # 创建图形
    plt.figure(figsize=(12, 8))
    
    # 子图1：条形图显示每个demo的长度
    plt.subplot(2, 2, 1)
    demo_indices = range(len(demo_lengths))
    bars = plt.bar(demo_indices, demo_lengths, alpha=0.7, color='skyblue', edgecolor='navy')
    plt.xlabel('Demo Index')
    plt.ylabel('Length (frames)')
    plt.title('Length of Each Demo')
    plt.grid(True, alpha=0.3)
    
    # 在条形图上添加数值标签
    for i, (bar, length) in enumerate(zip(bars, demo_lengths)):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5, 
                str(length), ha='center', va='bottom', fontsize=8)
    
    # 子图2：长度分布直方图
    plt.subplot(2, 2, 2)
    plt.hist(demo_lengths, bins=20, alpha=0.7, color='lightgreen', edgecolor='darkgreen')
    plt.xlabel('Length (frames)')
    plt.ylabel('Frequency')
    plt.title('Distribution of Demo Lengths')
    plt.grid(True, alpha=0.3)
    
    # 子图3：累积分布
    plt.subplot(2, 2, 3)
    sorted_lengths = sorted(demo_lengths)
    cumulative = np.cumsum(sorted_lengths)
    plt.plot(sorted_lengths, cumulative, 'o-', color='red', alpha=0.7)
    plt.xlabel('Length (frames)')
    plt.ylabel('Cumulative Sum')
    plt.title('Cumulative Sum of Demo Lengths')
    plt.grid(True, alpha=0.3)
    
    # 子图4：统计信息
    plt.subplot(2, 2, 4)
    plt.axis('off')
    
    # 计算统计信息
    stats_text = f"""
    Demo Length Statistics:
    
    Total Demos: {len(demo_lengths)}
    Min Length: {min(demo_lengths)}
    Max Length: {max(demo_lengths)}
    Mean Length: {np.mean(demo_lengths):.1f}
    Median Length: {np.median(demo_lengths):.1f}
    Std Dev: {np.std(demo_lengths):.1f}
    
    Total Frames: {sum(demo_lengths)}
    Average per Demo: {np.mean(demo_lengths):.1f}
    """
    
    plt.text(0.1, 0.9, stats_text, transform=plt.gca().transAxes, 
             fontsize=10, verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
    
    plt.tight_layout()
    plt.show()
    
    # 打印详细信息
    print(f"\n=== Demo Length Analysis ===")
    print(f"Total number of demos: {len(demo_lengths)}")
    print(f"Length range: {min(demo_lengths)} - {max(demo_lengths)} frames")
    print(f"Mean length: {np.mean(demo_lengths):.1f} frames")
    print(f"Median length: {np.median(demo_lengths):.1f} frames")
    print(f"Standard deviation: {np.std(demo_lengths):.1f} frames")
    print(f"Total frames: {sum(demo_lengths)}")
    
    return demo_lengths

def cal_transition_time():
    label_data = load_demonstrations_label()
    train_label = label_data[:122]
    transition_time = []
    for i in range(len(train_label)):
        demo_transition_time = []
        for j in range(len(train_label[i])-1):
            if train_label[i][j] != train_label[i][j+1]:
                demo_transition_time.append(j)
        transition_time.append(demo_transition_time)

    average_transition_time = []
    std_transition_time = []
    #calculate the average transition time
    for i in range(len(transition_time[0])):
        average_transition_time.append(np.mean([transition_time[j][i] for j in range(len(transition_time))]))
        std_transition_time.append(np.std([transition_time[j][i] for j in range(len(transition_time))]))
    return average_transition_time, std_transition_time

def load_test_demonstration():
    test_demo = load_demonstrations_state()
    test_label = load_demonstrations_label()
    return test_demo[122:], test_label[122:]

def visualize_scaled_state(demo_idx=0, close_event=None):
    """
    可视化经过 StandardScaler 标准化后某条 demo 的各特征曲线，左右手分列展示。

    列布局（16 列）：
      左手: pos(0:3), vel3(3:6), vel_scalar(6), gripper(7)
      右手: pos(8:11), vel3(11:14), vel_scalar(14), gripper(15)

    close_event: threading.Event，外部置位后自动关闭图窗（线程模式用）。
                 子进程模式下直接 terminate() 即可，无需传入。
    """
    import matplotlib
    matplotlib.use('TkAgg')
    scaled_demos = load_train_state(shuffle=False)

    if demo_idx >= len(scaled_demos):
        print(f"demo_idx {demo_idx} 超出范围，共 {len(scaled_demos)} 条训练数据")
        return

    data = scaled_demos[demo_idx]   # shape: (T, 16)
    T = data.shape[0]
    t = np.arange(T)

    fig, axes = plt.subplots(4, 2, figsize=(10, 15))
    fig.suptitle(
        f'Scaled State Visualization (StandardScaler) — train demo #{demo_idx}',
        fontsize=13, fontweight='bold'
    )

    # Row 0: Position
    for col, label, color in zip([0, 1, 2], ['x', 'y', 'z'],
                                 ['tab:red', 'tab:green', 'tab:blue']):
        axes[0, 0].plot(t, data[:, col],     label=label, color=color, linewidth=0.8)
        axes[0, 1].plot(t, data[:, 8 + col], label=label, color=color, linewidth=0.8)
    for ax, title in zip(axes[0], ['Left Hand — Position (scaled)', 'Right Hand — Position (scaled)']):
        ax.set_ylabel('Standardized value')
        ax.set_title(title)
        ax.legend(loc='upper right', fontsize=8)
        ax.axhline(0, color='k', linewidth=0.4, linestyle='--')
        ax.grid(True, alpha=0.3)

    # Row 1: Velocity3
    for col, label, color in zip([0, 1, 2], ['vx', 'vy', 'vz'],
                                 ['tab:red', 'tab:green', 'tab:blue']):
        axes[1, 0].plot(t, data[:, 3 + col],  label=label, color=color, linewidth=0.8)
        axes[1, 1].plot(t, data[:, 11 + col], label=label, color=color, linewidth=0.8)
    for ax, title in zip(axes[1], ['Left Hand — Velocity3 (scaled)', 'Right Hand — Velocity3 (scaled)']):
        ax.set_ylabel('Standardized value')
        ax.set_title(title)
        ax.legend(loc='upper right', fontsize=8)
        ax.axhline(0, color='k', linewidth=0.4, linestyle='--')
        ax.grid(True, alpha=0.3)

    # Row 2: Velocity scalar
    axes[2, 0].plot(t, data[:, 6],  color='tab:purple', linewidth=0.8)
    axes[2, 1].plot(t, data[:, 14], color='tab:orange', linewidth=0.8)
    for ax, title in zip(axes[2], ['Left Hand — Velocity scalar (scaled)', 'Right Hand — Velocity scalar (scaled)']):
        ax.set_ylabel('Standardized value')
        ax.set_title(title)
        ax.axhline(0, color='k', linewidth=0.4, linestyle='--')
        ax.grid(True, alpha=0.3)

    # Row 3: Gripper
    axes[3, 0].plot(t, data[:, 7],  color='tab:brown', linewidth=0.8)
    axes[3, 0].fill_between(t, data[:, 7],  alpha=0.25, color='tab:brown')
    axes[3, 1].plot(t, data[:, 15], color='tab:cyan',  linewidth=0.8)
    axes[3, 1].fill_between(t, data[:, 15], alpha=0.25, color='tab:cyan')

    # 在 gripper 图角落标注 0↔1 突变时间戳
    for ax, col, title in zip(axes[3], [7, 15],
                              ['Left Hand — Gripper', 'Right Hand — Gripper']):
        ax.set_ylabel('State (0/1)')
        ax.set_title(title)
        ax.set_xlabel('Frame')
        ax.axhline(0, color='k', linewidth=0.4, linestyle='--')
        ax.grid(True, alpha=0.3)

        gripper = data[:, col]
        diff = np.diff(gripper.astype(float), prepend=gripper[0])
        rise_frames = t[diff  > 0.5]   # 0 → 1
        fall_frames = t[diff  < -0.5]  # 1 → 0

        # 竖线标注
        for f in rise_frames:
            ax.axvline(f, color='green', linewidth=0.8, linestyle='--', alpha=0.7)
        for f in fall_frames:
            ax.axvline(f, color='red', linewidth=0.8, linestyle='--', alpha=0.7)

        # 角落文字汇总
        lines = []
        if len(rise_frames):
            lines.append('0->(1):' + ', '.join(str(int(f)) for f in rise_frames))
        if len(fall_frames):
            lines.append('1->(0):' + ', '.join(str(int(f)) for f in fall_frames))
        if lines:
            ax.text(0.01, 0.97, '\n'.join(lines),
                    transform=ax.transAxes,
                    fontsize=7, verticalalignment='top',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='lightyellow',
                              edgecolor='gray', alpha=0.8))

    plt.tight_layout()

    save_dir = "/home/lambda/surgical_robotics_challenge/scripts/surgical_robotics_challenge/Project/Dataset/vis"
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f'scaled_demo{demo_idx}.png')
    plt.savefig(save_path, dpi=150)
    print(f'图像已保存至: {save_path}')

    mgr = plt.get_current_fig_manager()

    # 设置窗口尺寸和位置（TkAgg）
    try:
        mgr.window.wm_geometry("+1800+100")
    except Exception:
        pass

    plt.show(block=False)
    plt.pause(0.1)

    # 保持窗口直到：外部 close_event 置位，或用户手动关闭图窗
    while plt.fignum_exists(fig.number):
        if close_event is not None and close_event.is_set():
            plt.close(fig)
            return
        plt.pause(0.3)

def get_test_demo_id_list(demo_id_list):
    shuffle_list = get_shuffled_demo_ids(shuffle=True, demo_id_list=demo_id_list)
    bound = int(ratio*len(shuffle_list))
    return shuffle_list[bound:]


# 使用示例
if __name__ == '__main__':
    import sys as _sys
    _demo_idx = int(_sys.argv[1]) if len(_sys.argv) > 1 else 0
    
    # demo_id_list = []
    # demonstrations_state = load_demonstrations_state()
    # print(demonstrations_state[0].shape)  # Example output: (number_of_frames, 14)
    # demo_lengths = visualize_demo_lengths()
    
    # label_data = load_demonstrations_label()
    # print(len(label_data[0]))
    # print(len(demonstrations_state), len(label_data))
    # train_label = label_data[:138]
    # average_transition_time, std_transition_time = cal_transition_time()
    # print(average_transition_time)
    # print(std_transition_time)
    # print(demo_id_list[138:]) # [184, 185, 186, 187, 188, 189, 190, 191, 192, 193, 194, 195, 196, 197, 198]

    # visualize_demo_lengths()
    # visualize_scaled_state(demo_idx=21)

    for i in range(82,148):
        visualize_scaled_state(demo_idx=i)
    



