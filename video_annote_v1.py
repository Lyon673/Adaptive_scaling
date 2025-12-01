import os
import cv2
import json
import tkinter as tk
from tkinter import filedialog, messagebox
from tkinter import ttk
from PIL import Image, ImageTk


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
        self.annotations = []
        self.current_label = ""
        # self.labels = []  # 用户定义的标签类别
        self.labels = [0, 1, 2, 3, 4, 5]  # 用户定义的标签类别


        self.new_video = 0

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
                                     command=self.on_slider_change, length=1800)
        self.frame_slider.pack(side=tk.LEFT, padx=5)

        self.frame_entry = tk.Entry(nav_frame, width=6)
        self.frame_entry.pack(side=tk.LEFT, padx=5)
        self.frame_entry.bind("<Return>", self.on_frame_entry)

        # 标注控制区域
        annotate_frame = tk.Frame(self.root)
        annotate_frame.pack(side=tk.TOP, padx=5, pady=5)

        # 标签管理区域
        label_frame = tk.LabelFrame(self.root, text="标签管理")
        label_frame.pack(side=tk.LEFT, padx=5, pady=5, fill=tk.BOTH)

        self.label_entry = tk.Entry(label_frame, width=20)
        self.label_entry.pack(side=tk.LEFT, padx=5)

        self.add_label_btn = tk.Button(label_frame, text="添加标签", command=self.add_label)
        self.add_label_btn.pack(side=tk.LEFT, padx=5)

        self.label_combobox = ttk.Combobox(label_frame, state="readonly")
        self.label_combobox.pack(side=tk.LEFT, padx=5)
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

        self.annotation_text = tk.Text(annotation_frame, height=10, width=40)
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

            # 加载已有标注文件
            self.load_annotations()

            # 显示第一帧
            self.update_frame()

            self.status_var.set(
                f"已加载视频: {os.path.basename(file_path)} | 总帧数: {self.total_frames} | FPS: {self.fps:.2f}")

    def load_annotations(self):
        """尝试加载同名的标注文件"""
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

                # 更新UI显示
                self.frame_slider.set(self.current_frame)
                self.frame_entry.delete(0, tk.END)
                self.frame_entry.insert(0, str(self.current_frame))

                # 检查是否播放结束
                if self.playing and self.current_frame >= self.total_frames - 1:
                    self.playing = False
                    self.play_btn.config(text="播放")

    def toggle_play(self):
        """切换播放/暂停状态"""
        if self.cap is not None:
            self.playing = not self.playing
            self.play_btn.config(text="暂停" if self.playing else "播放")
            self.play_video()

    def play_video(self):
        """播放视频"""
        if self.playing and self.cap is not None:
            self.next_frame()
            self.root.after(int(1000 / self.fps), self.play_video)

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
            # 暂停播放（用户拖动滑块时）
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
                # 更新已有标注的起始帧
                self.annotations[i]['start'] = self.current_frame
                self.update_annotation_display()
                self.status_var.set(f"已更新标注起始帧: {self.current_frame} ({self.current_label})")
                return

        # 创建新标注
        self.annotations.append({
            'label': self.current_label,
            'start': self.current_frame,
            'end': -1  # -1表示尚未标记结束
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

        self.update_annotation_display()
        self.update_frame()
        self.status_var.set(f"已标记结束帧: {self.current_frame} ({self.current_label})")

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
            start = ann['start']
            end = ann['end'] if ann['end'] != -1 else "未结束"
            self.annotation_text.insert(tk.END, f"标签: {ann['label']}\n")
            self.annotation_text.insert(tk.END, f"起始帧: {start}\n")
            self.annotation_text.insert(tk.END, f"结束帧: {end}\n")
            self.annotation_text.insert(tk.END, "-" * 30 + "\n")

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
            'labels': self.labels,
            'annotations': self.annotations
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
    INITIAL_VIDEO_DIR = "/home/lambda/Videos/train"
    
    # 设置JSON标注文件的导出目录（None 或不设置则保存在视频同目录）
    ANNOTATION_OUTPUT_DIR = os.path.join(current_dir, 'train', 'label')

    # ==============================
    
    root = tk.Tk()
    app = VideoPhaseAnnotator(
        root, 
        initial_video_dir=INITIAL_VIDEO_DIR,
        annotation_output_dir=ANNOTATION_OUTPUT_DIR
    )
    root.protocol("WM_DELETE_WINDOW", app.on_close)
    root.mainloop()