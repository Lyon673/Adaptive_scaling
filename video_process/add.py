import cv2
import numpy as np
from collections import deque
import os
import glob

gaze_dir = '/home/lambda/surgical_robotics_challenge/scripts/surgical_robotics_challenge/Project/data'
cap_dir = '/home/lambda/Videos/train'
output_dir = '/home/lambda/surgical_robotics_challenge/scripts/surgical_robotics_challenge/Project/video_process/output'

os.makedirs(output_dir, exist_ok=True)

# 获取cap_dir下所有视频，按编号排序
video_files = glob.glob(os.path.join(cap_dir, '*_NeedlePassing_demo.mp4'))
video_files.sort(key=lambda p: int(os.path.basename(p).split('_')[0]))

def find_gaze_folder(subject_id):
    """根据编号查找对应的注视点数据文件夹（后缀可能为01-18或01-20）"""
    for entry in os.listdir(gaze_dir):
        if entry.startswith(f'{subject_id}_data_'):
            folder = os.path.join(gaze_dir, entry)
            npy_path = os.path.join(folder, 'gazepoint_position_data.npy')
            if os.path.isfile(npy_path):
                return npy_path
    return None

total_videos = len(video_files)
print(f"共找到 {total_videos} 个视频，开始处理...")

for idx, video_path in enumerate(video_files[107:]):
    video_name = os.path.basename(video_path)
    subject_id = video_name.split('_')[0]

    gaze_npy_path = find_gaze_folder(subject_id)
    if gaze_npy_path is None:
        print(f"[{idx+1}/{total_videos}] 跳过 {video_name}：未找到对应注视点数据")
        continue

    output_path = os.path.join(output_dir, f'{subject_id}_output.mp4')
    if os.path.exists(output_path):
        print(f"[{idx+1}/{total_videos}] 跳过 {video_name}：输出文件已存在")
        continue

    gaze_data = np.load(gaze_npy_path, allow_pickle=True)
    total_gaze_points = len(gaze_data)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"[{idx+1}/{total_videos}] 跳过 {video_name}：无法打开视频")
        continue

    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    frame_count = 0
    trail_points = deque(maxlen=15)

    print(f"[{idx+1}/{total_videos}] 处理 {video_name}（{total_frames} 帧，{total_gaze_points} 注视点）...")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        if total_frames > 0 and total_gaze_points > 0:
            gaze_index = int(frame_count * total_gaze_points / total_frames)

            if gaze_index < total_gaze_points:
                current_gaze = gaze_data[gaze_index]
                x = int(current_gaze[0]) * 2
                y = 1080 - int(current_gaze[1]) * 2
                trail_points.append((x, y))

                for i in range(1, len(trail_points)):
                    pt1 = trail_points[i - 1]
                    pt2 = trail_points[i]
                    thickness = int(max(1, i * 3 / len(trail_points)))
                    cv2.line(frame, pt1, pt2, (0, 255, 255), thickness)

                overlay = frame.copy()
                cv2.circle(overlay, (x, y), 10, (0, 0, 255), -1)
                cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

        out.write(frame)
        frame_count += 1

    cap.release()
    out.release()
    trail_points.clear()
    print(f"[{idx+1}/{total_videos}] 完成 -> {output_path}")

print("全部视频处理完成！")
