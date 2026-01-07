import json
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from scipy.interpolate import interp1d

from scipy.spatial.transform import Rotation as R
from scipy.spatial.transform import Slerp

import config

# ==================== 数据集划分配置 ====================
# 设置随机种子以确保可重复性
RANDOM_SEED = 35
np.random.seed(RANDOM_SEED)

# 生成并shuffle demo ID列表
demo_id_list = np.arange(65)
demo_id_list = np.random.permutation(demo_id_list)

ratio = 0.9

def load_annotations(json_path):
    with open(json_path, 'r') as f:
        data = json.load(f)
    return data

def get_frame_label(annotation_data, frame_num):
    """
    return the label of the frame, if the frame is not in the annotation, return -1
    """
    for ann in annotation_data['annotations']:
        start = ann['start']
        end = ann['end'] 
        if start <= frame_num <= end + 1:
            return int(ann['label'])
    return -1

def generate_frame_label_map(dir_path, demo_id):
    """
    return a list of labels for each frame , whose length is the same as the total number of frames
    """
    label_path = os.path.join(dir_path, 'label', f'{demo_id}_NeedlePassing_demo_annotations.json')
    # kine_path = os.path.join(dir_path, 'state', f'{demo_id}.txt')
    # with open(kine_path, 'r') as f:
    #     kine_frames = len(f.readlines())
    # states = load_demonstrations_state()
    # kine_frames = len(states[demo_id])
    demo_lengths = np.load(os.path.join(needle_dir_path, 'demo_lengths.npy'))
    kine_frames = int(demo_lengths[demo_id])
    annotation_data = load_annotations(label_path)
    frame_map = []
    for i in range(kine_frames):
        frame_num = i * annotation_data['total_frames'] / kine_frames
        frame_map.append(get_frame_label(annotation_data, frame_num))
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

    left_state = content[:, :8]
    right_state = content[:, 8:16]

    return np.hstack((left_state, right_state))

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


def get_shuffled_demo_ids(shuffle=True):
    """获取统一的demo ID顺序，确保state和label使用相同顺序"""
    np.random.seed(RANDOM_SEED)  # 每次重置种子确保一致性
    demo_id_list = np.arange(65)
    if shuffle:
        demo_id_list = np.random.permutation(demo_id_list)
    return demo_id_list


def load_demonstrations_state(shuffle=True, without_quat=False, resample=False):
    if resample:
        print("Resampling demonstrations!!!")
    demo_id_list = get_shuffled_demo_ids(shuffle)
    demo_states = []
    demo_lengths = np.zeros(len(demo_id_list))
    for demo_id in demo_id_list:
        state = read_demo_kinematics_state(needle_dir_path, demo_id)
        demo_lengths[demo_id] = state.shape[0]
        if without_quat:
            state = state[:, [0,1,2,7,8,9,10,15]]
        if resample:
            state = resample_bimanual_trajectory(state, without_quat=without_quat)
            demo_states.append(state)
        else:
            if state.shape[0] <= 350:
                demo_states.append(state)
            else:
                print(f"too long demo idx:{demo_id}")
                continue
        
        
    print(f"last demo id: {demo_id_list[-1]}")
    #print(f"demo_states shape: {np.array(demo_states).shape}")
    np.save(os.path.join(needle_dir_path, 'demo_lengths.npy'), demo_lengths)

    return demo_states


def load_train_state(without_quat=False, resample=False):
    demo_states_all = load_demonstrations_state(without_quat=without_quat, resample=resample)
    bound = round(ratio*len(demo_states_all))
    demo_states = demo_states_all[:bound]
    scaler = StandardScaler()

    all_data_stacked = np.vstack(demo_states)
    scaler.fit(all_data_stacked)
    scaled_demos = [scaler.transform(arr) for arr in demo_states]
    return scaled_demos

def load_test_state(without_quat=False, resample=False):
    demo_states_all = load_demonstrations_state(without_quat=without_quat, resample=resample)
    bound = round(ratio*len(demo_states_all))
    demo_states = demo_states_all[bound:]
    scaler = StandardScaler()

    all_data_stacked = np.vstack(demo_states)
    scaler.fit(all_data_stacked)
    scaled_demos = [scaler.transform(arr) for arr in demo_states]
    return scaled_demos


def load_demonstrations_label(shuffle=True, resample=False):
    demo_id_list = get_shuffled_demo_ids(shuffle) 
    demo_labels = []
    for demo_id in demo_id_list:
        if resample:
            label = resample_label(demo_id=demo_id)
            demo_labels.append(label)
     
        else:
            label = generate_frame_label_map(needle_dir_path, demo_id)
            if len(label) <= 350 and len(label) > 0:
                demo_labels.append(label)
            else:
                print(f"error demo label idx:{demo_id}")
                continue
        
    return demo_labels


def load_train_label(resample=False):
    """
    load the label of the train demos
    params: resample - True/False
    """
    # 159 / 138
    demo_labels_all = load_demonstrations_label(resample=resample)
    bound = round(ratio*len(demo_labels_all))
    demo_labels = demo_labels_all[:bound]
    return demo_labels

def load_test_label(resample=False):
    demo_labels_all = load_demonstrations_label(resample=resample)
    bound = round(ratio*len(demo_labels_all))
    demo_labels = demo_labels_all[bound:]
    return demo_labels


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

# 使用示例
if __name__ == '__main__':
    #visualize_demo_lengths()
    # demo_id_list = []
    # demonstrations_state = load_demonstrations_state()
    # print(demonstrations_state[0].shape)  # Example output: (number_of_frames, 14)
    demo_lengths = visualize_demo_lengths()
    # label_data = load_demonstrations_label()
    # print(len(label_data[0]))
    # print(len(demonstrations_state), len(label_data))
    # train_label = label_data[:138]
    # average_transition_time, std_transition_time = cal_transition_time()
    # print(average_transition_time)
    # print(std_transition_time)
    # print(demo_id_list[138:]) # [184, 185, 186, 187, 188, 189, 190, 191, 192, 193, 194, 195, 196, 197, 198]
    



