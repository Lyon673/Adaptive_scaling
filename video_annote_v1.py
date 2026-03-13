import os
import sys
import cv2
import json
import tkinter as tk
from tkinter import filedialog, messagebox
from tkinter import ttk
from PIL import Image, ImageTk
import numpy as np


class VideoPhaseAnnotator:
    def __init__(self, root, initial_video_dir=None, annotation_output_dir=None):
        self.root = root
        self.root.title("手术操作阶段标注工具")

        # 路径配置
        self.initial_video_dir = initial_video_dir or os.getcwd()  # 打开视频的初始目录
        self.annotation_output_dir = annotation_output_dir  # JSON导出目录（None表示与视频同目录）
        
        # 初始化变量
        self.video_path = ""
        self.cap = None
        self.total_frames = 0
        self.current_frame = 0
        self.fps = 0
        self.video_width = 0
        self.video_height = 0
        self.playing = False
        self._slider_updating = False   # 程序内部更新滑块时置 True，避免触发暂停
        self.annotations = []
        self.current_label = ""
        # self.labels = []  # 用户定义的标签类别
        self.labels = [0, 1, 2, 3, 4, 5, 6]  # 用户定义的标签类别


        self.new_video = 0

        # 路径（需最先初始化，后续变量依赖它）
        self._project_dir = os.path.dirname(os.path.abspath(__file__))
        self._demo_lengths_path = os.path.join(self._project_dir, 'Dataset', 'demo_lengths.npy')
        # 数据流相关
        self.demo_idx = -1
        self.kine_frames = 0   # 当前 demo 的运动学数据帧数

        # 时间校准：视频帧号对应数据流首/末的视频帧位置
        # 视频侧校准帧号
        self.calib_start = None   # 视频帧号：对应数据流 kine_calib_start
        self.calib_end   = None   # 视频帧号：对应数据流 kine_calib_end
        # 数据流侧校准帧号（用户输入，默认为 0 和 kine_frames-1）
        self.kine_calib_start = 0
        self.kine_calib_end   = 0   # 加载视频后更新为 kine_frames-1
        # 映射: kine_idx = kine_calib_start + (video_frame - calib_start) /
        #                  (calib_end - calib_start) * (kine_calib_end - kine_calib_start)

        # 创建GUI布局
        self.create_widgets()

    def create_widgets(self):
        # 视频显示区域
        # self.video_frame = tk.Frame(self.root, width=800, height=500)
        # self.video_frame.pack(side=tk.TOP, padx=5, pady=5)
        #
        # self.canvas = tk.Canvas(self.video_frame, bg='black', width=800, height=500)
        # self.canvas.pack()

        self.video_frame = tk.Frame(self.root, width=1200, height=8800)
        self.video_frame.pack(side=tk.TOP, padx=5, pady=5)

        self.canvas = tk.Canvas(self.video_frame, bg='black', width=1200, height=800)
        self.canvas.pack()

        # 控制按钮区域
        control_frame = tk.Frame(self.root)
        control_frame.pack(side=tk.TOP, padx=5, pady=5)

        self.open_btn = tk.Button(control_frame, text="打开视频", command=self.open_video)
        self.open_btn.pack(side=tk.LEFT, padx=5)

        self.play_btn = tk.Button(control_frame, text="播放/暂停", command=self.toggle_play, state=tk.DISABLED)
        self.play_btn.pack(side=tk.LEFT, padx=5)

        self.prev_btn = tk.Button(control_frame, text="上一帧", command=self.prev_frame, state=tk.DISABLED)
        self.prev_btn.pack(side=tk.LEFT, padx=5)

        self.next_btn = tk.Button(control_frame, text="下一帧", command=self.next_frame, state=tk.DISABLED)
        self.next_btn.pack(side=tk.LEFT, padx=5)

        # 帧导航区域
        nav_frame = tk.Frame(self.root)
        nav_frame.pack(side=tk.TOP, padx=5, pady=5)

        self.frame_slider = tk.Scale(nav_frame, from_=0, to=0, orient=tk.HORIZONTAL,
                                     command=self.on_slider_change, length=1000)
        self.frame_slider.pack(side=tk.LEFT, padx=5)

        self.frame_entry = tk.Entry(nav_frame, width=6)
        self.frame_entry.pack(side=tk.LEFT, padx=5)
        self.frame_entry.bind("<Return>", self.on_frame_entry)

        self.kine_label_var = tk.StringVar(value="数据帧: — / —")
        self.kine_label = tk.Label(nav_frame, textvariable=self.kine_label_var,
                                   font=("Helvetica", 10), fg="#0055aa")
        self.kine_label.pack(side=tk.LEFT, padx=12)

        # 时间校准区域
        calib_frame = tk.LabelFrame(self.root, text="时间校准（视频↔数据流对齐）")
        calib_frame.pack(side=tk.TOP, padx=5, pady=3, fill=tk.X)

        # ── 第一行：视频帧标记按钮 ──
        calib_row1 = tk.Frame(calib_frame)
        calib_row1.pack(side=tk.TOP, fill=tk.X, pady=2)

        self.calib_start_btn = tk.Button(calib_row1, text="设为视频起始帧",
                                         command=self.set_calib_start, state=tk.DISABLED)
        self.calib_start_btn.pack(side=tk.LEFT, padx=6)

        self.calib_start_var = tk.StringVar(value="视频起始: —")
        tk.Label(calib_row1, textvariable=self.calib_start_var, width=16, anchor='w').pack(side=tk.LEFT)

        self.calib_end_btn = tk.Button(calib_row1, text="设为视频结束帧",
                                       command=self.set_calib_end, state=tk.DISABLED)
        self.calib_end_btn.pack(side=tk.LEFT, padx=6)

        self.calib_end_var = tk.StringVar(value="视频结束: —")
        tk.Label(calib_row1, textvariable=self.calib_end_var, width=16, anchor='w').pack(side=tk.LEFT)

        self.calib_reset_btn = tk.Button(calib_row1, text="重置校准",
                                         command=self.reset_calib, state=tk.DISABLED)
        self.calib_reset_btn.pack(side=tk.LEFT, padx=6)

        self.calib_status_var = tk.StringVar(value="未校准（使用总帧数等比映射）")
        tk.Label(calib_row1, textvariable=self.calib_status_var,
                 fg='gray', font=("Helvetica", 9)).pack(side=tk.LEFT, padx=10)

        # ── 第二行：数据流起止索引输入 ──
        calib_row2 = tk.Frame(calib_frame)
        calib_row2.pack(side=tk.TOP, fill=tk.X, pady=2)

        tk.Label(calib_row2, text="对应数据流起始帧:").pack(side=tk.LEFT, padx=6)
        self.kine_start_entry = tk.Entry(calib_row2, width=8)
        self.kine_start_entry.insert(0, '0')
        self.kine_start_entry.pack(side=tk.LEFT)
        self.kine_start_entry.bind('<Return>', lambda e: self._apply_kine_range())
        self.kine_start_entry.bind('<FocusOut>', lambda e: self._apply_kine_range())

        tk.Label(calib_row2, text="对应数据流结束帧:").pack(side=tk.LEFT, padx=(16, 6))
        self.kine_end_entry = tk.Entry(calib_row2, width=8)
        self.kine_end_entry.insert(0, '0')
        self.kine_end_entry.pack(side=tk.LEFT)
        self.kine_end_entry.bind('<Return>', lambda e: self._apply_kine_range())
        self.kine_end_entry.bind('<FocusOut>', lambda e: self._apply_kine_range())

        tk.Label(calib_row2, text="（输入后按 Enter 或移开焦点生效）",
                 fg='gray', font=("Helvetica", 8)).pack(side=tk.LEFT, padx=8)

        # 标注控制区域
        annotate_frame = tk.Frame(self.root)
        annotate_frame.pack(side=tk.TOP, padx=5, pady=5)

        # 标签管理区域
        label_frame = tk.LabelFrame(self.root, text="标签管理")
        label_frame.pack(side=tk.LEFT, padx=5, pady=5, fill=tk.BOTH, anchor='n')

        self.label_entry = tk.Entry(label_frame, width=5)
        self.label_entry.pack(side=tk.LEFT, padx=5, anchor='n')

        self.add_label_btn = tk.Button(label_frame, text="添加标签", command=self.add_label)
        self.add_label_btn.pack(side=tk.LEFT, padx=5, anchor='n')

        self.label_combobox = ttk.Combobox(label_frame, state="readonly")
        self.label_combobox.pack(side=tk.LEFT, padx=5, anchor='n')
        self.label_combobox.bind("<<ComboboxSelected>>", self.on_label_select)
        
        # 初始化标签列表
        if self.labels:
            self.label_combobox['values'] = self.labels
            self.current_label = self.labels[0]
            self.label_combobox.set(self.current_label)

        # 标注按钮
        self.mark_start_btn = tk.Button(annotate_frame, text="标记起始帧", command=self.mark_start_frame,
                                        state=tk.DISABLED)
        self.mark_start_btn.pack(side=tk.LEFT, padx=5)

        self.mark_end_btn = tk.Button(annotate_frame, text="标记结束帧", command=self.mark_end_frame, state=tk.DISABLED)
        self.mark_end_btn.pack(side=tk.LEFT, padx=5)

        self.clear_btn = tk.Button(annotate_frame, text="清除标注", command=self.clear_annotation, state=tk.DISABLED)
        self.clear_btn.pack(side=tk.LEFT, padx=5)

        # 标注显示区域
        annotation_frame = tk.LabelFrame(self.root, text="当前标注")
        annotation_frame.pack(side=tk.RIGHT, padx=5, pady=5, fill=tk.BOTH)

        self.annotation_text = tk.Text(annotation_frame, height=10, width=30)
        self.annotation_text.pack(fill=tk.BOTH, expand=True)

        # 保存按钮
        save_frame = tk.Frame(self.root)
        save_frame.pack(side=tk.BOTTOM, padx=5, pady=5)

        self.save_btn = tk.Button(save_frame, text="保存标注", command=self.save_annotations, state=tk.DISABLED)
        self.save_btn.pack(side=tk.LEFT, padx=5)

        # 状态栏
        self.status_var = tk.StringVar()
        self.status_var.set("就绪")
        self.status_bar = tk.Label(self.root, textvariable=self.status_var, relief=tk.SUNKEN, anchor=tk.W)
        self.status_bar.pack(side=tk.BOTTOM, fill=tk.X)

    def open_video(self):
        file_path = filedialog.askopenfilename(
            initialdir=self.initial_video_dir,
            title="选择视频文件",
            filetypes=[
                ("MP4 files", "*.mp4"),
                ("AVI files", "*.avi"),
                ("MOV files", "*.mov"),
                ("All video files", "*.mp4 *.avi *.mov *.mkv *.flv"),
                ("All files", "*.*")
            ]
        )
        
        if file_path:
            self.video_path = file_path
            self.cap = cv2.VideoCapture(file_path)

            if not self.cap.isOpened():
                messagebox.showerror("错误", "无法打开视频文件")
                return

            self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
            self.fps = self.cap.get(cv2.CAP_PROP_FPS)
            self.video_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            self.video_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

            # 更新UI
            self.frame_slider.config(to=self.total_frames - 1)
            self.current_frame = 0
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.current_frame)

            # 启用控件
            self.play_btn.config(state=tk.NORMAL)
            self.prev_btn.config(state=tk.NORMAL)
            self.next_btn.config(state=tk.NORMAL)
            self.mark_start_btn.config(state=tk.NORMAL)
            self.mark_end_btn.config(state=tk.NORMAL)
            self.clear_btn.config(state=tk.NORMAL)
            self.save_btn.config(state=tk.NORMAL)
            self.calib_start_btn.config(state=tk.NORMAL)
            self.calib_end_btn.config(state=tk.NORMAL)
            self.calib_reset_btn.config(state=tk.NORMAL)

            # 从文件名提取 demo_idx，如 "21_NeedlePassing_demo.mp4" → 21
            try:
                self.demo_idx = int(os.path.basename(file_path).split('_')[0])
            except (ValueError, IndexError):
                self.demo_idx = -1

            # 加载运动学数据帧数
            self.kine_frames = 0
            if self.demo_idx >= 0 and os.path.exists(self._demo_lengths_path):
                try:
                    demo_lengths = np.load(self._demo_lengths_path)
                    if self.demo_idx < len(demo_lengths):
                        self.kine_frames = int(demo_lengths[self.demo_idx])
                except Exception:
                    pass

            # 初始化数据流起止输入框
            # 默认：右手 gripper 第一次 0→1 / 1→0 的时间戳；找不到时退回全段
            ks_default, ke_default = self._detect_gripper_transitions(self.demo_idx)
            if ks_default is None:
                ks_default = 0
            if ke_default is None:
                ke_default = max(0, self.kine_frames - 1)
            self.kine_calib_start = ks_default
            self.kine_calib_end   = ke_default
            self.kine_start_entry.delete(0, tk.END)
            self.kine_start_entry.insert(0, str(self.kine_calib_start))
            self.kine_end_entry.delete(0, tk.END)
            self.kine_end_entry.insert(0, str(self.kine_calib_end))

            # 重置校准
            self.reset_calib()

            # 同步更新数据帧标签
            self._update_kine_label()

            # 加载已有标注文件
            self.load_annotations()

            # 显示第一帧
            self.update_frame()

            self.status_var.set(
                f"已加载视频: {os.path.basename(file_path)} | 总帧数: {self.total_frames} | FPS: {self.fps:.2f}")




    def frame_to_kine_idx(self, video_frame):
        """将视频帧号映射到运动学数据流索引（已校准时使用线性拉伸）"""
        if self.kine_frames <= 0:
            return 0
        cs = self.calib_start
        ce = self.calib_end
        ks = self.kine_calib_start
        ke = self.kine_calib_end
        if cs is not None and ce is not None and ce > cs:
            ratio = (video_frame - cs) / (ce - cs)
            kine_idx = ks + ratio * (ke - ks)
        elif self.total_frames > 0:
            ratio = video_frame / self.total_frames
            kine_idx = ks + ratio * (ke - ks)
        else:
            return 0
        return max(0, min(int(round(kine_idx)), self.kine_frames - 1))

    def _detect_gripper_transitions(self, demo_idx):
        """从 Dataset/state/{demo_idx}.txt 读取右手 gripper（第16列，列索引15），
        返回 (第一次 0→1 的帧号, 第一次 1→0 的帧号)。
        任一方向找不到时对应返回 None。"""
        if demo_idx < 0:
            return None, None
        state_path = os.path.join(self._project_dir, 'Dataset', 'state', f'{demo_idx}.txt')
        if not os.path.exists(state_path):
            return None, None
        try:
            data = np.loadtxt(state_path)          # shape (T, 16)
            gripper = data[:, 15].astype(float)
            diff = np.diff(gripper, prepend=gripper[0])
            rise = np.where(diff > 0.5)[0]   # 0 → 1
            fall = np.where(diff < -0.5)[0]  # 1 → 0
            ks = int(rise[0]) if len(rise) > 0 else None
            ke = int(fall[0]) if len(fall) > 0 else None
            return ks, ke
        except Exception:
            return None, None

    def _apply_kine_range(self):
        """读取输入框中的数据流起止索引并更新映射"""
        try:
            ks = int(self.kine_start_entry.get())
            ke = int(self.kine_end_entry.get())
            ks = max(0, min(ks, self.kine_frames - 1))
            ke = max(0, min(ke, self.kine_frames - 1))
            self.kine_calib_start = ks
            self.kine_calib_end   = ke
            self._refresh_calib_status()
            self._update_kine_label()
        except ValueError:
            pass

    def _update_kine_label(self):
        """刷新 GUI 上的数据帧显示"""
        if self.kine_frames > 0:
            kine_idx = self.frame_to_kine_idx(self.current_frame)
            self.kine_label_var.set(f"数据帧: {kine_idx} / {self.kine_frames - 1}")
        else:
            self.kine_label_var.set("数据帧: — / —")

    def set_calib_start(self):
        """将当前视频帧设为数据流 kine_calib_start 对应的视频帧"""
        self.calib_start = self.current_frame
        self.calib_start_var.set(f"视频起始: 第{self.current_frame}帧")
        self._refresh_calib_status()
        self._update_kine_label()

    def set_calib_end(self):
        """将当前视频帧设为数据流 kine_calib_end 对应的视频帧"""
        self.calib_end = self.current_frame
        self.calib_end_var.set(f"视频结束: 第{self.current_frame}帧")
        self._refresh_calib_status()
        self._update_kine_label()

    def reset_calib(self):
        """清除视频侧校准，恢复等比映射；数据流侧保留"""
        self.calib_start = None
        self.calib_end   = None
        self.calib_start_var.set("视频起始: —")
        self.calib_end_var.set("视频结束: —")
        self._refresh_calib_status()
        self._update_kine_label()

    def _refresh_calib_status(self):
        cs, ce = self.calib_start, self.calib_end
        ks, ke = self.kine_calib_start, self.kine_calib_end
        if cs is not None and ce is not None and ce > cs:
            self.calib_status_var.set(
                f"已校准  视频[{cs}→{ce}] ↔ 数据流[{ks}→{ke}]  视频跨度:{ce-cs}帧  数据跨度:{ke-ks}帧")
        elif cs is not None or ce is not None:
            self.calib_status_var.set("校准中…（请同时设置视频起始和结束帧）")
        else:
            self.calib_status_var.set(f"未校准  数据流范围:[{ks}→{ke}]（等比映射）")

    def load_annotations(self):
        """尝试加载同名的标注文件"""
        # 导入新视频时先清空历史标注和显示
        self.annotations = []
        self.annotation_text.delete(1.0, tk.END)

        video_basename = os.path.basename(self.video_path)
        video_name = os.path.splitext(video_basename)[0]
        
        # 首先尝试从自定义输出目录加载
        json_paths = []
        if self.annotation_output_dir:
            json_paths.append(os.path.join(self.annotation_output_dir, f"{video_name}_annotations.json"))
        
        # 然后尝试从视频同目录加载
        base_path = os.path.splitext(self.video_path)[0]
        json_paths.append(f"{base_path}_annotations.json")
        
        # 尝试加载第一个存在的标注文件
        for json_path in json_paths:
            if os.path.exists(json_path):
                with open(json_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    self.annotations = data.get('annotations', [])
                    self.labels = data.get('labels', [])

                    # 恢复校准参数
                    self.calib_start      = data.get('calib_start',      None)
                    self.calib_end        = data.get('calib_end',         None)
                    self.kine_calib_start = data.get('kine_calib_start',  0)
                    self.kine_calib_end   = data.get('kine_calib_end',
                                                     max(0, self.kine_frames - 1))
                    cs_txt = f"视频起始: 第{self.calib_start}帧" if self.calib_start is not None else "视频起始: —"
                    ce_txt = f"视频结束: 第{self.calib_end}帧"   if self.calib_end   is not None else "视频结束: —"
                    self.calib_start_var.set(cs_txt)
                    self.calib_end_var.set(ce_txt)
                    self.kine_start_entry.delete(0, tk.END)
                    self.kine_start_entry.insert(0, str(self.kine_calib_start))
                    self.kine_end_entry.delete(0, tk.END)
                    self.kine_end_entry.insert(0, str(self.kine_calib_end))
                    self._refresh_calib_status()

                    # 更新标签选择框
                    self.label_combobox['values'] = self.labels
                    if self.labels:
                        self.current_label = self.labels[0]
                        self.label_combobox.set(self.current_label)

                    # 更新标注显示
                    self.update_annotation_display()

                self.status_var.set(f"已加载已有标注文件: {os.path.basename(json_path)}")
                break

    def update_frame(self):
        """更新当前帧显示"""
        if self.cap is not None:
            ret, frame = self.cap.read()
            if ret:
                # 转换为RGB格式
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                # 检查当前帧是否有标注
                current_annotation = self.get_annotation_for_frame(self.current_frame)
                if current_annotation:
                    # 在帧上绘制标注信息
                    label = current_annotation['label']
                    cv2.putText(frame, f"Label: {label}", (10, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)

                # 调整大小以适应显示区域
                canvas_width = self.canvas.winfo_width()
                canvas_height = self.canvas.winfo_height()

                if canvas_width > 0 and canvas_height > 0:
                    img = Image.fromarray(frame)
                    img.thumbnail((canvas_width, canvas_height), Image.LANCZOS)

                    self.photo = ImageTk.PhotoImage(image=img)
                    self.canvas.create_image(canvas_width // 2, canvas_height // 2, image=self.photo, anchor=tk.CENTER)

                # 更新UI显示（用标志位防止滑块回调误触发暂停）
                self._slider_updating = True
                self.frame_slider.set(self.current_frame)
                self._slider_updating = False
                self.frame_entry.delete(0, tk.END)
                self.frame_entry.insert(0, str(self.current_frame))
                self._update_kine_label()

                # 检查是否播放结束
                if self.playing and self.current_frame >= self.total_frames - 1:
                    self.playing = False
                    self.play_btn.config(text="播放")

    def toggle_play(self):
        """切换播放/暂停状态"""
        if self.cap is not None:
            self.playing = not self.playing
            self.play_btn.config(text="暂停" if self.playing else "播放")
            if self.playing:
                # 确保 cap 从当前帧开始顺序读取
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.current_frame)
                self.play_video()

    def play_video(self):
        """播放视频（独立读帧循环，不经过 next_frame/on_slider_change）"""
        if not self.playing or self.cap is None:
            return

        # 读取下一帧（cap 内部指针已在正确位置，不需要 seek）
        ret, frame = self.cap.read()
        if not ret:
            # 读帧失败（已到末尾或解码错误）
            self.playing = False
            self.play_btn.config(text="播放")
            return

        self.current_frame += 1

        # 渲染帧
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        ann = self.get_annotation_for_frame(self.current_frame)
        if ann:
            cv2.putText(frame, f"Label: {ann['label']}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)

        cw = self.canvas.winfo_width()
        ch = self.canvas.winfo_height()
        if cw > 0 and ch > 0:
            img = Image.fromarray(frame)
            img.thumbnail((cw, ch), Image.LANCZOS)
            self.photo = ImageTk.PhotoImage(image=img)
            self.canvas.create_image(cw // 2, ch // 2, image=self.photo, anchor=tk.CENTER)

        # 更新 UI（用标志位防止 slider command 回调误触发暂停）
        self._slider_updating = True
        self.frame_slider.set(self.current_frame)
        self._slider_updating = False
        self.frame_entry.delete(0, tk.END)
        self.frame_entry.insert(0, str(self.current_frame))
        self._update_kine_label()

        # 到末尾则停止，否则继续调度
        if self.current_frame >= self.total_frames - 1:
            self.playing = False
            self.play_btn.config(text="播放")
        else:
            delay = int(1000 / self.fps) if self.fps > 0 else 33
            self.root.after(delay, self.play_video)

    def prev_frame(self):
        """跳转到上一帧"""
        if self.cap is not None:
            # 只有在非播放状态下才执行
            if not self.playing:
                self.current_frame = max(0, self.current_frame - 1)
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.current_frame)
                self.update_frame()

    def next_frame(self):
        """跳转到下一帧"""
        if self.cap is not None:
            # 在播放和非播放状态下都可以执行
            self.current_frame = min(self.total_frames - 1, self.current_frame + 1)
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.current_frame)
            self.update_frame()

    def on_slider_change(self, value):
        """滑块变化事件处理"""
        if self.cap is not None:
            # 程序内部更新滑块位置时不触发暂停
            if self._slider_updating:
                return
            # 用户手动拖动滑块时暂停播放
            was_playing = self.playing
            if was_playing:
                self.playing = False
                self.play_btn.config(text="播放")

            self.current_frame = int(value)
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.current_frame)
            self.update_frame()

    def on_frame_entry(self, event):
        """帧号输入框事件处理"""
        if self.cap is not None:
            try:
                frame_num = int(self.frame_entry.get())
                frame_num = max(0, min(self.total_frames - 1, frame_num))

                # 暂停播放（用户输入帧号时）
                was_playing = self.playing
                if was_playing:
                    self.playing = False
                    self.play_btn.config(text="播放")

                self.current_frame = frame_num
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.current_frame)
                self.update_frame()
            except ValueError:
                pass

    def add_label(self):
        """添加新标签"""
        self.label_combobox['values'] = self.labels
        new_label = self.label_entry.get().strip()
        if new_label and new_label not in self.labels:
            self.labels.append(new_label)
            self.label_combobox['values'] = self.labels
            self.current_label = new_label
            self.label_combobox.set(new_label)
            self.label_entry.delete(0, tk.END)
            self.status_var.set(f"已添加标签: {new_label}")

    def on_label_select(self, event):
        """标签选择事件处理"""
        self.current_label = self.label_combobox.get()

    def mark_start_frame(self):
        """标记起始帧"""
        if not self.current_label:
            messagebox.showwarning("警告", "请先选择或添加标签")
            return

        # 检查是否已有标注覆盖当前帧
        overlapping = self.find_overlapping_annotation(self.current_frame)
        if overlapping is not None:
            response = messagebox.askyesno("确认", f"当前帧已被标注为'{overlapping['label']}'，是否覆盖？")
            if not response:
                return

        # 查找是否已有相同标签的未结束的标注
        for i, ann in enumerate(self.annotations):
            if ann['label'] == self.current_label and ann['end'] == -1:
                self.annotations[i]['start'] = self.current_frame
                self.annotations[i]['kine_start'] = self.frame_to_kine_idx(self.current_frame)
                self.update_annotation_display()
                self.status_var.set(f"已更新标注起始帧: {self.current_frame} "
                                    f"[数据帧 {self.annotations[i]['kine_start']}] ({self.current_label})")
                return

        # 创建新标注
        self.annotations.append({
            'label': self.current_label,
            'start': self.current_frame,
            'kine_start': self.frame_to_kine_idx(self.current_frame),
            'end': -1,
            'kine_end': -1,
        })

        # 按开始帧排序
        self.annotations.sort(key=lambda x: x['start'])

        self.update_annotation_display()
        self.update_frame()
        self.status_var.set(f"已标记起始帧: {self.current_frame} ({self.current_label})")

    def mark_end_frame(self):
        """标记结束帧"""
        if not self.current_label:
            messagebox.showwarning("警告", "请先选择或添加标签")
            return

        # 查找未结束的标注
        unclosed = None
        for i, ann in enumerate(self.annotations):
            if ann['label'] == self.current_label and ann['end'] == -1:
                unclosed = i
                break

        if unclosed is None:
            messagebox.showwarning("警告", f"没有找到未结束的'{self.current_label}'标注")
            return

        # 检查结束帧是否小于起始帧
        if self.current_frame < self.annotations[unclosed]['start']:
            messagebox.showerror("错误", "结束帧不能小于起始帧")
            return

        # 检查是否与其他标注重叠
        for i, ann in enumerate(self.annotations):
            if i != unclosed and self.check_overlap(
                    self.annotations[unclosed]['start'], self.current_frame,
                    ann['start'], ann['end'] if ann['end'] != -1 else self.total_frames - 1
            ):
                response = messagebox.askyesno("确认",
                                               f"标注区间与'{ann['label']}'重叠，是否继续？")
                if not response:
                    return

        # 更新结束帧
        self.annotations[unclosed]['end'] = self.current_frame
        self.annotations[unclosed]['kine_end'] = self.frame_to_kine_idx(self.current_frame)

        self.update_annotation_display()
        self.update_frame()
        self.status_var.set(f"已标记结束帧: {self.current_frame} "
                            f"[数据帧 {self.annotations[unclosed]['kine_end']}] ({self.current_label})")

    def clear_annotation(self):
        """清除当前帧的标注"""
        to_remove = []

        for i, ann in enumerate(self.annotations):
            if ann['start'] <= self.current_frame and (
                    ann['end'] >= self.current_frame or ann['end'] == -1):
                to_remove.append(i)

        if not to_remove:
            messagebox.showinfo("信息", "当前帧没有标注")
            return

        if len(to_remove) > 1 or messagebox.askyesno("确认", "确定要删除当前帧的标注吗？"):
            # 反向删除以避免索引问题
            for i in sorted(to_remove, reverse=True):
                del self.annotations[i]

            self.update_annotation_display()
            self.update_frame()
            self.status_var.set("已清除当前帧的标注")

    def update_annotation_display(self):
        """更新标注信息显示"""
        self.annotation_text.delete(1.0, tk.END)

        if not self.annotations:
            self.annotation_text.insert(tk.END, "暂无标注")
            return

        for ann in self.annotations:
            end_str      = str(ann['end'])      if ann['end']      != -1 else "未结束"
            kine_end_str = str(ann.get('kine_end', '—')) if ann.get('kine_end', -1) != -1 else "未结束"
            self.annotation_text.insert(tk.END, f"标签: {ann['label']}\n")
            self.annotation_text.insert(tk.END,
                f"起始帧: {ann['start']}  [数据帧: {ann.get('kine_start', '—')}]\n")
            self.annotation_text.insert(tk.END,
                f"结束帧: {end_str}  [数据帧: {kine_end_str}]\n")
            self.annotation_text.insert(tk.END, "-" * 30 + "\n")

        self.annotation_text.see(tk.END)

    def save_annotations(self):
        """保存标注到文件"""
        if not self.video_path:
            return

        # 确定保存路径
        video_basename = os.path.basename(self.video_path)
        video_name = os.path.splitext(video_basename)[0]
        
        if self.annotation_output_dir:
            # 使用自定义输出目录
            os.makedirs(self.annotation_output_dir, exist_ok=True)
            json_path = os.path.join(self.annotation_output_dir, f"{video_name}_annotations.json")
        else:
            # 使用视频同目录
            base_path = os.path.splitext(self.video_path)[0]
            json_path = f"{base_path}_annotations.json"

        data = {
            'video_path': self.video_path,
            'total_frames': self.total_frames,
            'fps': self.fps,
            'demo_idx': self.demo_idx,
            'kine_frames': self.kine_frames,
            'calib_start':      self.calib_start,
            'calib_end':        self.calib_end,
            'kine_calib_start': self.kine_calib_start,
            'kine_calib_end':   self.kine_calib_end,
            'labels': self.labels,
            'annotations': self.annotations,
        }

        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

        self.status_var.set(f"标注已保存到: {os.path.basename(json_path)}")
        messagebox.showinfo("成功", f"标注已保存到:\n{json_path}")

    def get_annotation_for_frame(self, frame_num):
        """获取指定帧的标注"""
        for ann in self.annotations:
            if ann['start'] <= frame_num and (
                    ann['end'] >= frame_num or ann['end'] == -1):
                return ann
        return None

    def find_overlapping_annotation(self, frame_num):
        """查找与指定帧重叠的标注"""
        for ann in self.annotations:
            if ann['start'] <= frame_num and (
                    ann['end'] >= frame_num or ann['end'] == -1):
                return ann
        return None

    def check_overlap(self, start1, end1, start2, end2):
        """检查两个区间是否重叠"""
        return not (end1 < start2 or end2 < start1)

    def on_close(self):
        """关闭窗口时的清理工作"""
        if self.cap is not None:
            self.cap.release()
        self.root.destroy()


if __name__ == "__main__":
    # ========== 配置区域 ==========
    # 设置打开视频时的初始文件夹（None 或不设置则使用当前工作目录）
    current_dir = os.path.dirname(os.path.abspath(__file__))
    INITIAL_VIDEO_DIR = "/home/lambda/surgical_robotics_challenge/scripts/surgical_robotics_challenge/Project/video_process/output"
    
    # 设置JSON标注文件的导出目录（None 或不设置则保存在视频同目录）
    ANNOTATION_OUTPUT_DIR = os.path.join(current_dir, 'Dataset', 'label')

    # ==============================
    
    root = tk.Tk()
    # 格式: "宽x高+距屏幕左边距+距屏幕上边距"
    # 只设置位置而不强制尺寸: "+x+y"
    root.geometry("1400x1440+0+0")   # 左上角
    app = VideoPhaseAnnotator(
        root, 
        initial_video_dir=INITIAL_VIDEO_DIR,
        annotation_output_dir=ANNOTATION_OUTPUT_DIR
    )
    root.protocol("WM_DELETE_WINDOW", app.on_close)
    root.mainloop()