#!/usr/bin/env python3
"""
固定屏幕位置录屏工具
支持捕获指定区域的屏幕并保存为视频文件

使用独立进程进行截图+编码，避免 GIL 争抢导致主线程卡顿。
"""

import cv2
import numpy as np
import time
from datetime import datetime
import multiprocessing as mp
import threading
import argparse
import os

try:
    from mss import mss
    USE_MSS = True
except ImportError:
    from PIL import ImageGrab
    USE_MSS = False
    print("警告: mss 库未安装，使用 PIL.ImageGrab (速度较慢)")
    print("建议安装: pip install mss")


def _recording_process(monitor_dict, x, y, width, height,
                       out_w, out_h,
                       fps, output_file, stop_event, pause_event):
    """
    独立进程：截图 + 编码写入视频文件。
    在进程内创建一次 mss 实例并复用，避免每帧创建/销毁 X11 连接。
    截图以原始分辨率捕获，然后缩放到 (out_w, out_h) 再编码。
    """
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_file, fourcc, fps, (out_w, out_h))
    need_resize = (out_w != width or out_h != height)
    frame_time = 1.0 / fps
    frame_count = 0

    print(f"开始录制... 捕获 {width}x{height} → 输出 {out_w}x{out_h}")
    start_time = time.time()

    if USE_MSS:
        sct = mss()

    try:
        while not stop_event.is_set():
            if pause_event.is_set():
                time.sleep(0.1)
                continue

            loop_start = time.time()

            if USE_MSS:
                screenshot = sct.grab(monitor_dict)
                frame = np.array(screenshot)
                frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
            else:
                screenshot = ImageGrab.grab(bbox=(x, y, x + width, y + height))
                frame = np.array(screenshot)
                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

            if need_resize:
                frame = cv2.resize(frame, (out_w, out_h),
                                   interpolation=cv2.INTER_AREA)

            out.write(frame)
            frame_count += 1

            elapsed = time.time() - loop_start
            sleep_time = frame_time - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)

            if frame_count % fps == 0:
                duration = time.time() - start_time
                print(f"已录制: {duration:.1f}秒 | 帧数: {frame_count}")
    finally:
        if USE_MSS:
            sct.close()
        out.release()
        duration = time.time() - start_time
        print(f"\n录制完成!")
        print(f"总时长: {duration:.2f}秒")
        print(f"总帧数: {frame_count}")
        print(f"保存至: {output_file}")


class ScreenRecorder:
    def __init__(self, x, y, width, height, fps=60, output_file=None,
                 output_scale=1.0):
        """
        初始化屏幕录制器
        
        参数:
            x: 捕获区域左上角 x 坐标
            y: 捕获区域左上角 y 坐标
            width: 捕获区域宽度
            height: 捕获区域高度
            fps: 录制帧率
            output_file: 输出文件名（默认自动生成）
            output_scale: 输出缩放比例 (0~1)，例如 0.5 表示宽高各缩一半
        """
        self.x = x
        self.y = y
        self.width = width
        self.height = height
        self.fps = fps
        self.output_scale = output_scale
        self.out_w = int(width * output_scale)
        self.out_h = int(height * output_scale)
        
        if output_file is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.output_file = f"screen_record_{timestamp}.mp4"
        else:
            self.output_file = output_file
        
        self.recording = False
        self.paused = False
        self._process = None
        self._stop_event = None
        self._pause_event = None
        self.frame_count = 0
        
        self.monitor = {
            "top": y,
            "left": x,
            "width": width,
            "height": height,
        }
        
        print(f"录制区域: ({x}, {y}) - 捕获: {width}x{height} → 输出: {self.out_w}x{self.out_h}")
        print(f"帧率: {fps} FPS")
        print(f"输出文件: {self.output_file}")
    
    def start(self):
        """开始录制（启动独立进程）"""
        if not self.recording:
            self.recording = True
            self.paused = False
            self._stop_event = mp.Event()
            self._pause_event = mp.Event()
            self._process = mp.Process(
                target=_recording_process,
                args=(self.monitor, self.x, self.y,
                      self.width, self.height,
                      self.out_w, self.out_h,
                      self.fps, self.output_file,
                      self._stop_event, self._pause_event),
                daemon=True,
            )
            self._process.start()
            print("录制已启动 (独立进程)")
        else:
            print("已在录制中")
    
    def pause(self):
        """暂停录制"""
        if self.recording and not self.paused:
            self.paused = True
            if self._pause_event:
                self._pause_event.set()
            print("录制已暂停")
        else:
            print("当前未在录制或已暂停")
    
    def resume(self):
        """恢复录制"""
        if self.recording and self.paused:
            self.paused = False
            if self._pause_event:
                self._pause_event.clear()
            print("录制已恢复")
        else:
            print("当前未暂停")
    
    def stop(self):
        """停止录制"""
        if self.recording:
            print("正在停止录制...")
            self.recording = False
            if self._stop_event:
                self._stop_event.set()
            if self._process and self._process.is_alive():
                self._process.join(timeout=5)
                if self._process.is_alive():
                    self._process.terminate()
            print("录制已停止")
        else:
            print("当前未在录制")
    
    def is_recording(self):
        """检查是否正在录制"""
        return self.recording and not self.paused


def get_screen_info():
    """获取屏幕信息"""
    try:
        if USE_MSS:
            with mss() as sct:
                monitor = sct.monitors[1]  # 主显示器
                print(f"\n主屏幕尺寸: {monitor['width']}x{monitor['height']}")
                print(f"位置: ({monitor['left']}, {monitor['top']})")
        else:
            from PIL import ImageGrab
            screen = ImageGrab.grab()
            print(f"\n屏幕尺寸: {screen.width}x{screen.height}")
    except Exception as e:
        print(f"获取屏幕信息失败: {e}")


def interactive_mode():
    """交互模式：让用户手动选择录制区域"""
    print("\n=== 交互式区域选择 ===")
    print("请输入要录制的屏幕区域坐标:")
    
    try:
        x = int(input("左上角 X 坐标 [默认: 0]: ") or "0")
        y = int(input("左上角 Y 坐标 [默认: 0]: ") or "0")
        width = int(input("宽度 [默认: 1920]: ") or "1920")
        height = int(input("高度 [默认: 1080]: ") or "1080")
        fps = int(input("帧率 (FPS) [默认: 30]: ") or "30")
        duration = float(input("录制时长（秒）[默认: 10]: ") or "10")
        
        return x, y, width, height, fps, duration
    except ValueError:
        print("输入错误，使用默认值")
        return 0, 0, 1920, 1080, 30, 10


def main():

    current_file_path = os.path.abspath(__file__)
    current_dir = os.path.dirname(current_file_path)
    path = os.path.join(current_dir, 'data')
    num_dirs = sum(1 for name in os.listdir(path) if os.path.isdir(os.path.join(path, name)))
    dir_name = f'/home/lambda/Videos/train'
    os.makedirs(dir_name, exist_ok=True)

    output_file = os.path.join(dir_name, f"{num_dirs}_NeedlePassing_demo.mp4")

    parser = argparse.ArgumentParser(description='固定屏幕位置录屏工具')
    parser.add_argument('-x', type=int, default=0, help='捕获区域左上角 X 坐标 (默认: 0)')
    parser.add_argument('-y', type=int, default=0, help='捕获区域左上角 Y 坐标 (默认: 0)')
    parser.add_argument('-w', '--width', type=int, default=1920, help='捕获区域宽度 (默认: 1920)')
    parser.add_argument('-H', '--height', type=int, default=1080, help='捕获区域高度 (默认: 1080)')
    parser.add_argument('-f', '--fps', type=int, default=30, help='录制帧率 (默认: 30)')
    parser.add_argument('-d', '--duration', type=float, default=180, help='录制时长（秒）(默认: 180)')
    parser.add_argument('-o', '--output', type=str, default=output_file, help='输出文件名')
    parser.add_argument('-i', '--interactive', action='store_true', help='交互式模式')
    parser.add_argument('--info', action='store_true', help='显示屏幕信息')
    
    args = parser.parse_args()
    
    # 显示屏幕信息
    if args.info:
        get_screen_info()
        return
    
    # 交互模式
    if args.interactive:
        get_screen_info()
        x, y, width, height, fps, duration = interactive_mode()
    else:
        x, y, width, height, fps, duration = (
            args.x, args.y, args.width, args.height, 
            args.fps, args.duration
        )
    
    # 创建录制器
    recorder = ScreenRecorder(x, y, width, height, fps, args.output)
    
    # 开始录制
    recorder.start()
    
    try:
        # 录制指定时长
        time.sleep(duration)
    except KeyboardInterrupt:
        print("\n用户中断录制")
    finally:
        # 停止录制
        recorder.stop()


if __name__ == "__main__":
    print("=" * 60)
    print("固定屏幕位置录屏工具")
    print("=" * 60)
    
    # 检查依赖
    print(f"使用捕获库: {'mss (快速)' if USE_MSS else 'PIL.ImageGrab (兼容)'}")
    
    main()

