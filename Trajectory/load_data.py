import json
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from scipy.interpolate import interp1d

from scipy.spatial.transform import Rotation as R
from scipy.spatial.transform import Slerp

demo_id_list = np.arange(30)
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
    kine_path = os.path.join(dir_path, 'state', f'{demo_id}.txt')
    with open(kine_path, 'r') as f:
        kine_frames = len(f.readlines())
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


def resample_spatially(trajectory, step_size=0.5):
    """
    对轨迹进行空间等距离重采样。
    """
    # 1. 提取时间和空间坐标
    coords = trajectory

    # 2. 计算相邻点之间的距
    # diffs[i] 是点 i 和点 i+1 之间的向量
    diffs = np.diff(coords, axis=0) 
    # dists[i] 是点 i 和点 i+1 之间的标量距离
    dists = np.linalg.norm(diffs, axis=1)

    # 3. 计算累积距离 (Cumulative Arc Length)
    # 在最前面补0，因为第一个点的累积距离是0
    cum_dist = np.r_[0, np.cumsum(dists)]

    # 4. 生成新的等间距距离网格
    total_dist = cum_dist[-1]
    target_dists = np.arange(0, total_dist, step_size)
    
    # 防止最后一点丢失，如果余数很小可以忽略，或者强制加上终点
    if total_dist - target_dists[-1] > 1e-6:
        target_dists = np.append(target_dists, total_dist)

    # 5. 执行插值 (关键步骤)
    # np.interp(x_new, x_old, y_old)
    Left_px = np.interp(target_dists, cum_dist, coords[:, 0])
    Left_py = np.interp(target_dists, cum_dist, coords[:, 1])
    Left_pz = np.interp(target_dists, cum_dist, coords[:, 2])
    Left_ox = np.interp(target_dists, cum_dist, coords[:, 3])
    Left_oy = np.interp(target_dists, cum_dist, coords[:, 4])
    Left_oz = np.interp(target_dists, cum_dist, coords[:, 5])
    Left_ow = np.interp(target_dists, cum_dist, coords[:, 6])
    Left_grasp = np.interp(target_dists, cum_dist, coords[:, 7])
    Right_px = np.interp(target_dists, cum_dist, coords[:, 8])
    Right_py = np.interp(target_dists, cum_dist, coords[:, 9])
    Right_pz = np.interp(target_dists, cum_dist, coords[:, 10])
    Right_ox = np.interp(target_dists, cum_dist, coords[:, 10])
    Right_oy = np.interp(target_dists, cum_dist, coords[:, 12])
    Right_oz = np.interp(target_dists, cum_dist, coords[:, 13])
    Right_ow = np.interp(target_dists, cum_dist, coords[:, 14])
    Right_grasp = np.interp(target_dists, cum_dist, coords[:, 15])
    
    return np.column_stack((Left_px, Left_py, Left_pz, Left_ox, Left_oy, Left_oz, Left_ow, Left_grasp, Right_px, Right_py, Right_pz, Right_ox, Right_oy, Right_oz, Right_ow, Right_grasp))



def resample_bimanual_trajectory(data, step_size=0.5):
    """
    对双臂16维轨迹进行空间重采样。
    
    参数:
        data: numpy array, shape (N, 17). 
              Col 0: Time
              Col 1-3: L_Pos (x,y,z)
              Col 4-7: L_Quat (qx,qy,qz,qw) *注意scipy默认顺序是xyzw，需确认你的数据顺序*
              Col 8:   L_Gripper
              Col 9-11: R_Pos
              Col 12-15: R_Quat
              Col 16:   R_Gripper
        step_size: float, 重采样的空间步长 (mm)
        
    返回:
        resampled_data: shape (M, 18). 增加了 delta_t 列。
    """
    # --- 1. 数据切片 ---
    # 假设输入四元数顺序是 [w, x, y, z] (常见ROS顺序)，但Scipy需要 [x, y, z, w]
    # 这里需要你根据实际数据调整。下面代码假设输入已经是 [x, y, z, w]
    
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
    # 这就是比简单相加更科学的“联合距离”
    dists = np.linalg.norm(deltas, axis=1)
    
    # 3. 计算累计进度 (Progress Variable)
    s_cumulative = np.concatenate(([0], np.cumsum(dists)))
    total_dist = s_cumulative[-1]
    
    # 4. 生成新的均匀进度网格
    s_new = np.linspace(0, total_dist, step_size)
    
    # 5. 分别对左手和右手进行插值
    # 注意：这里我们用同一个 s_new 对两只手进行插值，
    # 保证了它们在新的时间步上是严格“同步”的。
    
    # 左手插值函数
    f_left = interp1d(s_cumulative, left_traj, axis=0, kind='cubic')
    left_resampled = f_left(s_new)
    
    # 右手插值函数
    f_right = interp1d(s_cumulative, right_traj, axis=0, kind='cubic')
    right_resampled = f_right(s_new)

    # --- 2. 计算混合累积距离 (The Driver) ---
    # 计算左手位移
    l_dist = np.linalg.norm(np.diff(l_pos, axis=0), axis=1)
    # 计算右手位移
    r_dist = np.linalg.norm(np.diff(r_pos, axis=0), axis=1)
    
    # 混合距离：两手位移之和 (或者取最大值 np.maximum)
    combined_step = l_dist + r_dist
    cum_dist = np.r_[0, np.cumsum(combined_step)]
    
    # 生成目标距离网格
    total_dist = cum_dist[-1]
    target_dists = np.arange(0, total_dist, step_size)
    
    # --- 3. 分组插值 ---
    
    # A. 时间 & 夹爪 (标量，线性或最近邻)
    # 对于夹爪，我们用 interpolate 之后 > 0.5 判决，或者直接用 nearest
    # 这里演示通用线性插值后取整
    new_l_grip = np.round(np.interp(target_dists, cum_dist, l_grip)) # 0或1
    new_r_grip = np.round(np.interp(target_dists, cum_dist, r_grip))
    
    # B. 位置 (线性插值)
    new_l_pos = np.zeros((len(target_dists), 3))
    new_r_pos = np.zeros((len(target_dists), 3))
    for i in range(3):
        new_l_pos[:, i] = np.interp(target_dists, cum_dist, l_pos[:, i])
        new_r_pos[:, i] = np.interp(target_dists, cum_dist, r_pos[:, i])
        
    # C. 旋转 (SLERP) - 关键部分
    # Scipy 的 Slerp 需要以 "key_times" (这里是 cum_dist) 初始化
    
    # 左手旋转插值
    l_rot_obj = R.from_quat(l_quat)
    l_slerp = Slerp(cum_dist, l_rot_obj)
    new_l_quat = l_slerp(target_dists).as_quat() # 返回插值后的四元数
    
    # 右手旋转插值
    r_rot_obj = R.from_quat(r_quat)
    r_slerp = Slerp(cum_dist, r_rot_obj)
    new_r_quat = r_slerp(target_dists).as_quat()



    # --- 5. 拼接结果 ---
    # 结果顺序: [Time, L_Pos(3), L_Quat(4), L_Grip(1), R_Pos(3), R_Quat(4), R_Grip(1), Delta_T(1)]
    resampled = np.column_stack((
        new_l_pos, new_l_quat, new_l_grip,
        new_r_pos, new_r_quat, new_r_grip,
    ))
    
    return resampled

def load_demonstrations_state():
    demo_states = []
    for demo_id in demo_id_list:
        state = read_demo_kinematics_state(needle_dir_path, demo_id)
        #state = resample_spatially(state)
        if state.shape[0] <= 350:
            demo_states.append(state)
        else:
            continue
        #demo_states.append(state)

    
    return demo_states


def load_train_state():
    demo_states_all = load_demonstrations_state()
    bound = round(ratio*len(demo_states_all))
    demo_states = demo_states_all[:bound]
    scaler = StandardScaler()

    all_data_stacked = np.vstack(demo_states)
    scaler.fit(all_data_stacked)
    scaled_demos = [scaler.transform(arr) for arr in demo_states]
    return scaled_demos

def load_test_state():
    demo_states_all = load_demonstrations_state()
    bound = round(ratio*len(demo_states_all))
    demo_states = demo_states_all[bound:]
    scaler = StandardScaler()

    all_data_stacked = np.vstack(demo_states)
    scaler.fit(all_data_stacked)
    scaled_demos = [scaler.transform(arr) for arr in demo_states]
    return scaled_demos


def load_demonstrations_label():
    demo_labels = []
    for demo_id in demo_id_list:
        label = generate_frame_label_map(needle_dir_path, demo_id)
        if len(label) <= 350:
            demo_labels.append(label)
        else:
            continue
        # demo_labels.append(label)
    return demo_labels


def load_train_label():
    # 159 / 138
    demo_labels_all = load_demonstrations_label()
    bound = round(ratio*len(demo_labels_all))
    demo_labels = demo_labels_all[:bound]
    return demo_labels

def load_test_label():
    demo_labels_all = load_demonstrations_label()
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
    # demo_id_list = []
    # demonstrations_state = load_demonstrations_state()
    # print(demonstrations_state[0].shape)  # Example output: (number_of_frames, 14)
    demo_lengths = visualize_demo_lengths()
    # train_demo = demonstrations_state[:138]
    # label_data = load_demonstrations_label()
    # train_label = label_data[:138]
    # average_transition_time, std_transition_time = cal_transition_time()
    # print(average_transition_time)
    # print(std_transition_time)
    # print(demo_id_list[138:]) # [184, 185, 186, 187, 188, 189, 190, 191, 192, 193, 194, 195, 196, 197, 198]
    



