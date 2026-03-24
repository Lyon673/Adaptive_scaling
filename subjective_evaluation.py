"""
subjective_evaluation.py
========================
Standalone NASA-TLX style subjective evaluation GUI with objective metrics.
Reads the latest data folder, computes objective performance indicators
(gracefulness, smoothness, clutch times, total distance, total time),
and saves both subjective ratings and objective metrics to a JSON file.
"""

import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import json
import os
import datetime
import numpy as np

from gracefulness import cal_GS, get_latest_data_dir
import params.config as config


class SubjectiveEvaluationGUI:
    # Rating dimensions: (key, label, low_anchor, high_anchor)
    DIMENSIONS = [
        ("physical_demand",  "1. Physical Demand",            "Easy",  "Difficult"),
        ("temporal_demand",  "2. Temporal Demand",            "Easy",  "Difficult"),
        ("controllability",  "3. Controllability",            "Good",  "Poor"),
        ("performance",      "4. Performance",                "Good",  "Poor"),
        ("mental_demand",    "5. Mental Demand",              "Easy",  "Difficult"),
        ("effort",           "6. Effort",                     "Easy",  "Difficult"),
        ("frustration",      "7. Frustration / Distractions", "Low",   "High"),
    ]

    # Score bounds from config
    SCORE_BOUNDS = config.scoreParams_bound

    def __init__(self, root):
        self.root = root
        self.root.title("Subjective Evaluation (NASA-TLX) + Objective Metrics")
        self.root.geometry("2560x1440+2560+0")
        self.root.minsize(900, 700)

        # Rating variables
        self.vars: dict[str, tk.DoubleVar] = {}
        for key, *_ in self.DIMENSIONS:
            self.vars[key] = tk.DoubleVar(value=5.0)

        self.participant_var = tk.StringVar(value="")
        self.note_var = tk.StringVar(value="")

        # JSON path
        self.json_path: str | None = None
        self.json_path_var = tk.StringVar(value="(not selected)")

        # Objective metrics cache
        self.obj_metrics: dict | None = None
        self.data_dir_var = tk.StringVar(value="(not loaded)")

        self._configure_styles()
        self._build_ui()
        self._load_latest_data()

# ── Styles ────────────────────────────────────────────────────────────

    def _configure_styles(self):
        self.style = ttk.Style(self.root)
        self.style.theme_use("clam")

        # 【核心修改】大幅调小字号，对抗系统的高DPI放大
        self.font_base  = ("Noto Sans", 13)
        self.font_title = ("Noto Sans", 15, "bold")
        self.font_small = ("Noto Sans", 11)

        bg = "#f0f0f0"
        self.style.configure("TFrame",           background=bg)
        self.style.configure("TLabel",           font=self.font_base, background=bg)
        self.style.configure("TButton",          font=self.font_base, padding=4) # 极限压缩按钮内边距
        self.style.configure("TLabelframe",      font=self.font_base, padding=6, background=bg)
        self.style.configure("TLabelframe.Label", font=self.font_title, background=bg)
        self.style.configure("Path.TLabel",      font=self.font_small, background=bg, foreground="#555")

    # ── UI layout ─────────────────────────────────────────────────────────

    def _build_ui(self):
        outer = ttk.Frame(self.root, padding=10) # 压缩最外层边距
        outer.pack(fill=tk.BOTH, expand=True)

        # ── Top: file selector ────────────────────────────────────────────
        file_frame = ttk.LabelFrame(outer, text="Save Location")
        file_frame.pack(fill=tk.X, pady=(0, 4)) # 压缩所有板块的纵向间距到极限

        btn_row = ttk.Frame(file_frame)
        btn_row.pack(fill=tk.X, pady=(0, 2))

        ttk.Button(btn_row, text="Auto New JSON",
                   command=self._auto_new_json).pack(side=tk.LEFT, padx=(0, 10))
        ttk.Button(btn_row, text="Select Existing JSON …",
                   command=self._select_json).pack(side=tk.LEFT, padx=(0, 10))

        ttk.Label(file_frame, textvariable=self.json_path_var,
                  style="Path.TLabel").pack(anchor=tk.W, pady=(2, 0))

        # ── Data folder + Objective metrics (auto-loaded) ─────────────────
        obj_frame = ttk.LabelFrame(outer, text="Objective Metrics  (auto-loaded)")
        obj_frame.pack(fill=tk.X, pady=(0, 4))

        data_hdr = ttk.Frame(obj_frame)
        data_hdr.pack(fill=tk.X, pady=(0, 2))

        ttk.Label(data_hdr, text="Data folder:", font=self.font_small,
                  foreground="#555").pack(side=tk.LEFT)
        ttk.Label(data_hdr, textvariable=self.data_dir_var,
                  font=self.font_base).pack(side=tk.LEFT, padx=(8, 20))
        ttk.Button(data_hdr, text="Reload",
                   command=self._load_latest_data).pack(side=tk.LEFT)

        metrics_grid = ttk.Frame(obj_frame)
        metrics_grid.pack(fill=tk.X, padx=10, pady=(0, 2))

        self.obj_labels: dict[str, ttk.Label] = {}
        obj_items = [
            ("gracefulness", "Gracefulness (G)"),
            ("smoothness",   "Smoothness (S)"),
            ("clutch_L",     "Clutch Times (L)"),
            ("clutch_R",     "Clutch Times (R)"),
            ("total_dist",   "Total Distance"),
            ("total_time",   "Total Time (s)"),
        ]
        for col, (mkey, mtext) in enumerate(obj_items):
            ttk.Label(metrics_grid, text=mtext, font=self.font_small,
                      foreground="#555").grid(row=0, column=col, padx=12, sticky="w")
            lbl = ttk.Label(metrics_grid, text="—", font=self.font_base)
            lbl.grid(row=1, column=col, padx=12, sticky="w")
            self.obj_labels[mkey] = lbl

        score_grid = ttk.Frame(obj_frame)
        score_grid.pack(fill=tk.X, padx=10, pady=(2, 2))

        self.score_labels: dict[str, ttk.Label] = {}
        score_items = [
            ("g_score",    "G Score (5)"),
            ("s_score",    "S Score (5)"),
            ("clutch_sc",  "Clutch Score (30)"),
            ("dist_sc",    "Distance Score (40)"),
            ("time_sc",    "Time Score (20)"),
            ("obj_total",  "Obj Total (100)"),
        ]
        for col, (skey, stext) in enumerate(score_items):
            ttk.Label(score_grid, text=stext, font=self.font_small,
                      foreground="#555").grid(row=0, column=col, padx=12, sticky="w")
            lbl = ttk.Label(score_grid, text="—", font=self.font_base)
            lbl.grid(row=1, column=col, padx=12, sticky="w")
            self.score_labels[skey] = lbl

        # ── Participant / Note ────────────────────────────────────────────
        info_frame = ttk.Frame(outer)
        info_frame.pack(fill=tk.X, pady=(0, 4))

        ttk.Label(info_frame, text="Participant ID:").pack(side=tk.LEFT)
        ttk.Entry(info_frame, textvariable=self.participant_var,
                  width=12, font=self.font_base).pack(side=tk.LEFT, padx=(6, 20))
        ttk.Label(info_frame, text="Note:").pack(side=tk.LEFT)
        ttk.Entry(info_frame, textvariable=self.note_var,
                  width=30, font=self.font_base).pack(side=tk.LEFT, padx=(6, 0))

        # ── Rating scales ─────────────────────────────────────────────────
        scale_frame = ttk.LabelFrame(outer, text="NASA-TLX Subjective Rating (0 – 10)")
        scale_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 4))

        ttk.Label(scale_frame,
                  text="Please rate the following items based on your teleoperation experience:",
                  font=self.font_title).pack(anchor=tk.W, pady=(2, 6))

        for key, label, lo, hi in self.DIMENSIONS:
            self._add_scale_row(scale_frame, key, label, lo, hi)

        # ── Bottom buttons ────────────────────────────────────────────────
        btn_frame = ttk.Frame(outer)
        btn_frame.pack(fill=tk.X, pady=(6, 0))

        ttk.Button(btn_frame, text="Reset",
                   command=self._reset_scales).pack(side=tk.RIGHT, padx=10)
        ttk.Button(btn_frame, text="Submit & Save",
                   command=self._submit).pack(side=tk.RIGHT, padx=10)

        # ── Status bar ────────────────────────────────────────────────────
        self.status_var = tk.StringVar(value="Ready")
        ttk.Label(outer, textvariable=self.status_var,
                  font=self.font_small, foreground="#666").pack(anchor=tk.W, pady=(4, 0))

    def _add_scale_row(self, parent, key, label_text, low_anchor, high_anchor):
        row = ttk.Frame(parent)
        row.pack(fill=tk.X, pady=4, padx=10) # 【核心修改】行间距压到 2 像素

        lbl = ttk.Label(row, text=label_text, width=28, anchor="w")
        lbl.grid(row=0, column=0, sticky="w")

        lo_lbl = ttk.Label(row, text=low_anchor, width=10, anchor="e",
                           font=self.font_small, foreground="#888")
        lo_lbl.grid(row=0, column=1, padx=(10, 6))

        var = self.vars[key]
        # 【核心修改】对于水平Scale，width代表"垂直高度"，将其从28缩减到14
        scale = tk.Scale(row, from_=0, to=10, orient=tk.HORIZONTAL,
                         variable=var, length=800, resolution=0.1,
                         width=28, sliderlength=30, showvalue=False, 
                         bg="#f0f0f0", troughcolor="#ccc", highlightthickness=0,
                         font=self.font_small)
        scale.grid(row=0, column=2, padx=6, sticky="ew")
        row.columnconfigure(2, weight=1)

        hi_lbl = ttk.Label(row, text=high_anchor, width=10, anchor="w",
                           font=self.font_small, foreground="#888")
        hi_lbl.grid(row=0, column=3, padx=(6, 10))

        val_lbl = ttk.Label(row, text="5.0", width=4, anchor="e")
        val_lbl.grid(row=0, column=4, padx=(10, 0))

        def _update(*_a, v=var, vl=val_lbl):
            vl.config(text=f"{v.get():.1f}")
        var.trace_add("write", _update)

    # ── Objective metrics ────────────────────────────────────────────────

    def _load_latest_data(self):
        """Read the latest data folder and compute all objective metrics."""
        try:
            current_dir = os.path.dirname(os.path.abspath(__file__))
            data_base_dir = os.path.join(current_dir, 'data')
            latest_dir = get_latest_data_dir(data_base_dir)
        except FileNotFoundError as e:
            
            return

        folder_name = os.path.basename(latest_dir)
        self.data_dir_var.set(folder_name)

        try:
            gracefulness, smoothness = cal_GS(use_left=True, use_right=True)
            clutch_times = np.load(os.path.join(latest_dir, 'clutch_times.npy'),
                                   allow_pickle=True)
            total_distance = np.load(os.path.join(latest_dir, 'total_distance.npy'),
                                     allow_pickle=True)
            total_time = np.load(os.path.join(latest_dir, 'total_time.npy'),
                                  allow_pickle=True)[0]
        except Exception as e:
            messagebox.showerror("Metric Error", f"Failed to compute metrics:\n{e}")
            return

        bounds = self.SCORE_BOUNDS
        g_score = 5 * np.clip(
            (bounds['gracefulness_max'] - gracefulness) /
            (bounds['gracefulness_max'] - bounds['gracefulness_min']), 0, 1)
        s_score = 5 * np.clip(
            (bounds['smoothness_max'] - smoothness) /
            (bounds['smoothness_max'] - bounds['smoothness_min']), 0, 1)
        clutch_score = 30 * np.clip(
            (bounds['clutch_times_max'] - clutch_times[0] - clutch_times[1]) /
            bounds['clutch_times_max'], 0, 1)
        dist_score = 40 * np.clip(
            (bounds['total_distance_max'] - total_distance[0]) /
            bounds['total_distance_max'], 0, 1)
        time_score = 20 * np.clip(
            (bounds['total_time_max'] - total_time) /
            bounds['total_time_max'], 0, 1)
        obj_total = float(g_score + s_score + clutch_score + dist_score + time_score)

        self.obj_metrics = {
            "data_folder": folder_name,
            "gracefulness": float(gracefulness),
            "smoothness": float(smoothness),
            "clutch_times_L": float(clutch_times[0]),
            "clutch_times_R": float(clutch_times[1]),
            "total_distance": float(total_distance[0]),
            "total_time": float(total_time),
            "gracefulness_score": float(g_score),
            "smoothness_score": float(s_score),
            "clutch_times_score": float(clutch_score),
            "total_distance_score": float(dist_score),
            "total_time_score": float(time_score),
            "objective_total": obj_total,
        }

        self.obj_labels["gracefulness"].config(text=f"{gracefulness:.4f}")
        self.obj_labels["smoothness"].config(text=f"{smoothness:.4f}")
        self.obj_labels["clutch_L"].config(text=f"{clutch_times[0]:.2f}")
        self.obj_labels["clutch_R"].config(text=f"{clutch_times[1]:.2f}")
        self.obj_labels["total_dist"].config(text=f"{total_distance[0]:.4f}")
        self.obj_labels["total_time"].config(text=f"{total_time:.2f}")

        self.score_labels["g_score"].config(text=f"{g_score:.2f}")
        self.score_labels["s_score"].config(text=f"{s_score:.2f}")
        self.score_labels["clutch_sc"].config(text=f"{clutch_score:.2f}")
        self.score_labels["dist_sc"].config(text=f"{dist_score:.2f}")
        self.score_labels["time_sc"].config(text=f"{time_score:.2f}")
        self.score_labels["obj_total"].config(text=f"{obj_total:.2f}")

        self.status_var.set(f"Loaded data: {folder_name}  |  Obj Total = {obj_total:.2f}")

    # ── File helpers ──────────────────────────────────────────────────────

    def _auto_new_json(self):
        current_dir = os.path.dirname(os.path.abspath(__file__))
        save_dir = os.path.join(current_dir, "TLX_evaluation_results")
        os.makedirs(save_dir, exist_ok=True)

        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        path = os.path.join(save_dir, f"subjective_{ts}.json")

        with open(path, "w", encoding="utf-8") as f:
            json.dump([], f)

        self.json_path = path
        self.json_path_var.set(path)
        self.status_var.set(f"Created: {os.path.basename(path)}")

    def _select_json(self):
        path = filedialog.askopenfilename(
            title="Select JSON file",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")],
            initialdir=os.path.dirname(os.path.abspath(__file__)),
        )
        if path:
            self.json_path = path
            self.json_path_var.set(path)
            self.status_var.set(f"Selected: {os.path.basename(path)}")

    # ── Submit ────────────────────────────────────────────────────────────

    def _submit(self):
        if self.json_path is None:
            messagebox.showwarning("No File", "Please select or create a JSON file first.")
            return

        self._load_latest_data()

        if self.obj_metrics is None:
            messagebox.showwarning(
                "No Data",
                "No objective metrics available.\n"
                "Check that the data folder exists, then click Reload.",
            )
            return

        ratings = {k: round(v.get(), 2) for k, v in self.vars.items()}

        inv = {k: 1.0 - v / 10.0 for k, v in ratings.items()}
        sub_score = (
            15 * inv["physical_demand"]
            + 15 * inv["temporal_demand"]
            + 30 * inv["controllability"]
            + 30 * inv["performance"]
            +  3 * inv["mental_demand"]
            +  3 * inv["effort"]
            +  4 * inv["frustration"]
        )

        obj_total = self.obj_metrics["objective_total"]
        total_score = 0.5 * obj_total + 0.5 * sub_score

        entry = {
            "timestamp": datetime.datetime.now().isoformat(timespec="seconds"),
            "participant": self.participant_var.get().strip() or None,
            "note": self.note_var.get().strip() or None,
            "ratings": ratings,
            "subjective_score": round(sub_score, 4),
            "objective_metrics": self.obj_metrics,
            "total_score": round(total_score, 4),
        }

        # read → append → write
        data = []
        if os.path.exists(self.json_path):
            try:
                with open(self.json_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
            except (json.JSONDecodeError, Exception):
                data = []

        data.append(entry)

        try:
            with open(self.json_path, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=4)
        except Exception as e:
            messagebox.showerror("Save Error", str(e))
            return

        n = len(data)
        self.status_var.set(
            f"Saved #{n}  |  Sub={sub_score:.2f}  Obj={obj_total:.2f}  "
            f"Total={total_score:.2f}  |  {os.path.basename(self.json_path)}"
        )
        messagebox.showinfo(
            "Saved",
            f"Rating #{n} saved successfully.\n\n"
            f"Subjective: {sub_score:.2f}\n"
            f"Objective:  {obj_total:.2f}\n"
            f"Total (50/50): {total_score:.2f}\n\n"
            f"Data folder: {self.obj_metrics['data_folder']}\n"
            f"File: {self.json_path}",
        )

    # ── Reset ─────────────────────────────────────────────────────────────

    def _reset_scales(self):
        for v in self.vars.values():
            v.set(5.0)
        self.status_var.set("Scales reset to default (5.0)")


def main():
    root = tk.Tk()
    SubjectiveEvaluationGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
