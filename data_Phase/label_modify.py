import os
import numpy as np
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import matplotlib
matplotlib.use('TkAgg')
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
import matplotlib.pyplot as plt
from matplotlib.widgets import SpanSelector

class PhaseLabelEditor:
    def __init__(self, root):
        self.root = root
        self.root.title("手术阶段标签修正工具 (Surgical Phase Label Editor)")
        self.root.geometry("1200x700")
        
        # 数据相关
        self.file_path = None
        self.labels = None
        self.history = []  # 用于撤销操作
        
        # 选择相关
        self.start_idx = 0
        self.end_idx = 0
        self.span = None
        
        # 阶段定义与颜色映射
        self.phase_map = {
            -1: ("热身期 (Warm-up)", "gray"),
            0: ("粗大阶段 0 (Coarse)", "lightcoral"),
            1: ("精细阶段 1 (Fine)", "mediumseagreen"),
            2: ("粗大阶段 2 (Coarse)", "sandybrown"),
            3: ("精细阶段 3 (Fine)", "mediumaquamarine"),
            4: ("粗大阶段 4 (Coarse)", "darkorange"),
            5: ("精细阶段 5 (Fine)", "lightseagreen"),
            6: ("粗大阶段 6 (Coarse)", "gold")
        }
        
        self._setup_ui()
        
    def _setup_ui(self):
        # ─── 左侧控制面板 ──────────────────────────────────────────
        control_frame = ttk.Frame(self.root, width=300, padding=10)
        control_frame.pack(side=tk.LEFT, fill=tk.Y, expand=False)
        
        # 文件操作区
        file_lf = ttk.LabelFrame(control_frame, text="文件操作", padding=10)
        file_lf.pack(fill=tk.X, pady=10)
        
        ttk.Button(file_lf, text="加载 phase_labels.npy", command=self.load_file).pack(fill=tk.X, pady=5)
        ttk.Button(file_lf, text="保存修改 (Save)", command=self.save_file).pack(fill=tk.X, pady=5)
        
        self.file_label = ttk.Label(file_lf, text="未加载文件", wraplength=250, foreground="blue")
        self.file_label.pack(fill=tk.X, pady=5)
        
        # 标签修改区
        edit_lf = ttk.LabelFrame(control_frame, text="标签修正", padding=10)
        edit_lf.pack(fill=tk.X, pady=10)
        
        ttk.Label(edit_lf, text="1. 在右侧图表中拖拽鼠标选中区间", foreground="dimgray").pack(anchor=tk.W, pady=2)
        
        self.range_var = tk.StringVar(value="当前选中帧: [0, 0]")
        ttk.Label(edit_lf, textvariable=self.range_var, font=("", 10, "bold")).pack(anchor=tk.W, pady=5)
        
        ttk.Label(edit_lf, text="2. 选择目标阶段标签:").pack(anchor=tk.W, pady=5)
        
        self.target_phase_var = tk.IntVar(value=1)
        for phase_val, (phase_name, _) in self.phase_map.items():
            ttk.Radiobutton(edit_lf, text=f"[{phase_val}] {phase_name}", 
                            variable=self.target_phase_var, value=phase_val).pack(anchor=tk.W)
                            
        ttk.Button(edit_lf, text="应用修改 (Apply)", command=self.apply_change).pack(fill=tk.X, pady=15)
        ttk.Button(edit_lf, text="撤销上一步 (Undo)", command=self.undo).pack(fill=tk.X, pady=5)
        
        # 统计信息
        self.stats_lf = ttk.LabelFrame(control_frame, text="序列信息", padding=10)
        self.stats_lf.pack(fill=tk.X, pady=10)
        self.stats_label = ttk.Label(self.stats_lf, text="总帧数: 0")
        self.stats_label.pack(anchor=tk.W)

        # ─── 右侧绘图面板 ──────────────────────────────────────────
        plot_frame = ttk.Frame(self.root, padding=10)
        plot_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        
        self.fig, self.ax = plt.subplots(figsize=(8, 5))
        self.canvas = FigureCanvasTkAgg(self.fig, master=plot_frame)
        
        # 添加 Matplotlib 原生工具栏（用于缩放、平移）
        toolbar = NavigationToolbar2Tk(self.canvas, plot_frame)
        toolbar.update()
        self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)

    def load_file(self):
        # 获取当前脚本所在的文件夹目录
        current_dir = os.path.dirname(os.path.abspath(__file__))
        # (如果你希望是终端运行时的当前工作目录，可以换成 current_dir = os.getcwd())
        
        filepath = filedialog.askopenfilename(
            title="选择标签文件",
            initialdir=current_dir,  # <--- 添加这一行，指定初始目录
            filetypes=[("Numpy Arrays", "*.npy"), ("All Files", "*.*")]
        )
        
        if not filepath:
            return
            
        try:
            data = np.load(filepath)
            if data.ndim != 1:
                messagebox.showerror("错误", "该 npy 文件不是一维数组，请检查！")
                return
            
            self.file_path = filepath
            # 确保类型是 int32 以兼容你现有的系统
            self.labels = data.astype(np.int32)
            self.history = []  # 清空历史
            
            self.file_label.config(text=f"当前文件:\n{os.path.basename(filepath)}")
            self.stats_label.config(text=f"总帧数: {len(self.labels)}")
            
            self.start_idx, self.end_idx = 0, 0
            self.range_var.set("当前选中帧: [0, 0]")
            
            self._update_plot()
            
        except Exception as e:
            messagebox.showerror("加载失败", f"无法加载文件:\n{e}")

    def save_file(self):
        if self.labels is None:
            messagebox.showwarning("警告", "没有可保存的数据！")
            return
            
        filepath = filedialog.asksaveasfilename(
            title="保存标签文件",
            defaultextension=".npy",
            initialfile=os.path.basename(self.file_path) if self.file_path else "phase_labels_fixed.npy",
            filetypes=[("Numpy Arrays", "*.npy")]
        )
        if not filepath:
            return
            
        try:
            np.save(filepath, self.labels)
            messagebox.showinfo("成功", f"文件已成功保存至:\n{filepath}")
            self.file_path = filepath
            self.file_label.config(text=f"当前文件:\n{os.path.basename(filepath)}")
        except Exception as e:
            messagebox.showerror("保存失败", f"无法保存文件:\n{e}")

    def _find_segments(self, label_array, target_label):
        """ 查找连续阶段片段用于绘制背景 """
        in_phase = (label_array == target_label)
        if not np.any(in_phase): return []
        changes = np.diff(in_phase.astype(np.int8))
        starts = list(np.where(changes == 1)[0] + 1)
        ends = list(np.where(changes == -1)[0] + 1)
        if in_phase[0]: starts = [0] + starts
        if in_phase[-1]: ends = ends + [len(label_array)]
        return list(zip(starts, ends))

    def _update_plot(self):
        self.ax.clear()
        if self.labels is None:
            self.canvas.draw()
            return
            
        frames = np.arange(len(self.labels))
        
        # 1. 绘制带有颜色的背景区域
        for phase_val, (phase_name, color) in self.phase_map.items():
            segments = self._find_segments(self.labels, phase_val)
            for i, (s, e) in enumerate(segments):
                label_str = phase_name if i == 0 else None
                self.ax.axvspan(s, e, color=color, alpha=0.3, label=label_str)
                
        # 2. 绘制步进曲线 (阶梯图)
        self.ax.step(frames, self.labels, where='post', color='black', linewidth=1.5)
        
        self.ax.set_ylim(-1.5, 6.5)
        self.ax.set_xlim(0, len(self.labels))
        self.ax.set_yticks(list(self.phase_map.keys()))
        self.ax.set_ylabel("Phase Label")
        self.ax.set_xlabel("Kinematic Frame Index")
        self.ax.set_title("Surgical Phase Timeline", fontweight='bold')
        
        # 图例放到图表外面
        self.ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=4, fontsize=9)
        self.fig.tight_layout()
        
        # 3. 重新绑定 SpanSelector (鼠标拖拽区间选择)
        self.span = SpanSelector(
            self.ax, self._on_select, 'horizontal', useblit=True,
            props=dict(alpha=0.4, facecolor='blue'), interactive=True
        )
        
        self.canvas.draw()

    def _on_select(self, xmin, xmax):
        """ 鼠标拖拽结束后的回调 """
        # 限制在合法帧数范围内
        self.start_idx = max(0, int(np.floor(xmin)))
        self.end_idx = min(len(self.labels) - 1, int(np.ceil(xmax)))
        
        if self.start_idx > self.end_idx:
            self.start_idx, self.end_idx = self.end_idx, self.start_idx
            
        self.range_var.set(f"当前选中帧: [{self.start_idx}, {self.end_idx}]")

    def apply_change(self):
        if self.labels is None:
            return
        if self.start_idx == self.end_idx and self.start_idx == 0:
            messagebox.showinfo("提示", "请先在右侧图表中按住鼠标左键拖拽以选中需要修改的区间。")
            return
            
        target_val = self.target_phase_var.get()
        
        # 保存当前状态到历史 (深度拷贝)
        self.history.append(self.labels.copy())
        # 限制历史记录长度，防止内存占用过大
        if len(self.history) > 20:
            self.history.pop(0)
            
        # 修改标签
        self.labels[self.start_idx:self.end_idx] = target_val
        
        # 刷新视图
        self._update_plot()
        
    def undo(self):
        if not self.history:
            messagebox.showinfo("提示", "没有可以撤销的操作了。")
            return
            
        # 弹出上一步状态并恢复
        self.labels = self.history.pop()
        self._update_plot()

if __name__ == "__main__":
    root = tk.Tk()
    # 根据操作系统适配高分辨率屏幕
    try:
        root.tk.call('tk', 'scaling', 1.5)
    except:
        pass
    app = PhaseLabelEditor(root)
    root.mainloop()