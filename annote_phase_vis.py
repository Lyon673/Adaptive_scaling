import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.colors import ListedColormap
from PIL import Image
import cv2
from pathlib import Path
from tqdm import tqdm

# color for phase
# FIXED_COLOR_MAP = {
#     '1': '#d62728',   # 红色
#     '2': '#9467bd',       # 紫色
#     '3': '#e377c2',       # 粉色
#     '4': '#ff7f0e',  # 橙色
# }

# color for action
FIXED_COLOR_MAP = {
    '1': '#a2e1d4', #青绿色
    '2': '#24b2a0', #深绿色
    '3': '#f3a847', #浅橙色
    '4': '#d95f2b', #深橙色
    '5': '#a84a9e', #浅紫色
    '6': '#5e2a91', #深紫色
}

def load_annotations(json_path):
    with open(json_path, 'r') as f:
        data = json.load(f)
    return data

def get_frame_label(annotation_data, frame_num):
    for ann in annotation_data['annotations']:
        start = ann['start']
        end = ann['end'] if ann['end'] != -1 else annotation_data['total_frames'] - 1
        if start <= frame_num <= end:
            return ann['label']
    return None

def generate_frame_label_map(annotation_data):
    frame_map = []
    for frame_num in range(annotation_data['total_frames']):
        frame_map.append(get_frame_label(annotation_data, frame_num))
    return frame_map



def visualize_from_frame_map(frame_label_map, fps=None):
    """基于frame_label_map的时间线可视化"""
    unique_labels = list(set(label for label in frame_label_map if label is not None))
    if not unique_labels:
        print("没有找到有效的标签")
        return

    # 创建颜色映射
    cmap = plt.cm.get_cmap('tab20', len(unique_labels))
    color_map = {label: cmap(i) for i, label in enumerate(sorted(unique_labels))}

    fig, ax = plt.subplots(figsize=(14, 3))

    # 绘制未标注区域（灰色）
    unlabeled = [i for i, label in enumerate(frame_label_map) if label is None]
    if unlabeled:
        ax.bar(unlabeled, [0.5] * len(unlabeled), width=1,
               color='lightgray', label='未标注')

    # 绘制标注区域
    current_label = frame_label_map[0]
    start_frame = 0

    for frame_num, label in enumerate(frame_label_map[1:], 1):
        if label != current_label:
            if current_label is not None:
                duration = frame_num - start_frame
                ax.barh(0, duration, left=start_frame, height=0.5,
                        color=color_map[current_label], label=current_label)
            start_frame = frame_num
            current_label = label

    # 绘制最后一个段
    if current_label is not None:
        duration = len(frame_label_map) - start_frame
        ax.barh(0, duration, left=start_frame, height=0.5,
                color=color_map[current_label], label=current_label)

    # 设置图表属性
    ax.set_yticks([])
    ax.set_xlabel('帧号' if fps is None else '时间 (秒)')

    # 添加时间轴（如果提供了fps）
    if fps:
        ax.set_xticks(ax.get_xticks())
        ax.set_xticklabels([f"{x / fps:.1f}" for x in ax.get_xticks()])

    # 创建图例（去重）
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys(),
              bbox_to_anchor=(1.02, 1), loc='upper left')

    plt.tight_layout()
    plt.show()


def visualize_label_heatmap(frame_label_map,save_path=None,fps=None):
    """
    像素级热图可视化（每帧一个像素）
    参数:
        frame_label_map: 列表形式的帧标签映射
        fps: 帧率（可选，用于显示时间轴）
    """
    # 获取唯一标签
    unique_labels = list(set(label for label in frame_label_map if label is not None))
    if not unique_labels:
        print("没有找到有效的标签")
        return

    color_list = []
    label_order = []

    # 首先添加固定颜色标签
    for label, color in FIXED_COLOR_MAP.items():
        if label in unique_labels and label != '未标注':
            color_list.append(color)
            label_order.append(label)
    cmap = ListedColormap(color_list)

    # 准备数据
    label_to_idx = {label: i + 1 for i, label in enumerate(label_order)}
    data = np.array([label_to_idx.get(label, 0) for label in frame_label_map])


    fig, ax = plt.subplots(figsize=(14, 2))

    # 绘制热图
    img = ax.imshow(data.reshape(1, -1), aspect='auto', cmap=cmap,
                    interpolation='nearest')

    # 设置刻度
    ax.set_yticks([])
    ax.set_xticks([])

    if fps:
        sec_interval = max(1, len(frame_label_map) // (10 * fps))  # 每10秒一个刻度
        ax_sec = ax.twiny()
        ax_sec.set_xlim(ax.get_xlim())
        ax_sec.set_xticks(np.arange(0, len(frame_label_map), sec_interval * fps))
        ax_sec.set_xticklabels([f"{x / fps:.1f}" for x in np.arange(0, len(frame_label_map), sec_interval * fps)])
        ax_sec.set_xlabel('time(second)')

    plt.tight_layout()
    if save_path != None:
        plt.savefig(save_path)
    plt.show()

def count_frame_label(frame_label_map,num_cls):
    count_freq = np.zeros((num_cls))
    for idx in range(1,num_cls+1):
        count_freq[idx-1] = np.array(np.array(frame_label_map) == str(idx)).sum()
    return count_freq

if __name__ == '__main__':

    # 可视化单一标注文件
    # json_path = r"G:\Workflow Analysis IMRNeuron\action_annotation\250402-A-05-process_annotations.json"
    # annotation_data = load_annotations(json_path)
    # frame_label_map = generate_frame_label_map(annotation_data)

    # 查看数据结构
    # print("视频路径:", annotation_data['video_path'])
    # print("总帧数:", annotation_data['total_frames'])
    # print("帧率(FPS):", annotation_data['fps'])
    # print("可用标签:", annotation_data['labels'])
    # print("标注区间:", annotation_data['annotations'])

    # 可视化标注阶段
    # save_path = json_path.replace('.json','.png')
    # visualize_label_heatmap(frame_label_map,save_path)


    # 获取每个阶段对应帧数,数据统计
    src_dir = r'G:\Workflow Analysis IMRNeuron\phase_annotation'
    path = Path(src_dir)
    filelist = list(path.glob('*.json'))
    num_cls = 4
    count_freq = np.zeros((num_cls))
    for idx in tqdm(range(len(filelist))):
        json_file_path = str(filelist[idx])
        annotation_data = load_annotations(json_file_path)
        frame_label_map = generate_frame_label_map(annotation_data)
        count_freq_map = count_frame_label(frame_label_map,num_cls)
        count_freq = count_freq + count_freq_map
    print(count_freq)

    if num_cls == 6 :
        categories = ['target', 'insert', 'decouple', 'retract', 'refocus','idle']
        colors = ['#a2e1d4', '#24b2a0', '#f3a847', '#d95f2b','#a84a9e', '#5e2a91']
    if num_cls == 4:
        categories = ['assembly','desorption','implantation','transition']
        colors = ['#d62728', '#9467bd', '#e377c2', '#ff7f0e',]

    fig, ax = plt.subplots(figsize=(10, 6))
    bars = plt.bar(categories,count_freq/30/60,color=colors)
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2, height + 1,
                 '{:.02f}min'.format(height),
                 ha='center', va='bottom',
                 fontsize=11, fontweight='bold')
    plt.tight_layout()
    plt.show()