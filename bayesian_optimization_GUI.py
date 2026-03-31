"""
factors: distance_GP, velocity_psm, IPAL, IPAR
"""

import tkinter as tk
from tkinter import ttk, messagebox
from bayes_opt import BayesianOptimization
from bayes_opt import UtilityFunction
from bayes_opt import SequentialDomainReductionTransformer
from bayes_opt.logger import JSONLogger
from bayes_opt.event import Events
from bayes_opt.util import load_logs

import params.config as config

from gracefulness import cal_GS
from gracefulness import get_latest_data_dir
import numpy as np
import os
import threading
import sys
import json
import random

mode = 2

class BayesianOptimizationGUI:
	# def __init__(self, root):
	# 	self.root = root
	# 	self.root.title("Bayesian Optimization System")
	# 	# Increased window size for better layout
	# 	self.root.geometry("2560x1440+2560+0")
		
	# 	# Initialize variables
	# 	self.current_iteration = 0
	# 	self.max_iterations = 10
	# 	self.optimizer = None
	# 	self.next_point = None
	# 	self.mental_demand = tk.DoubleVar(value=5)
	# 	self.physical_demand = tk.DoubleVar(value=5)
	# 	self.controllability = tk.DoubleVar(value=5)
	# 	self.temporal_demand = tk.DoubleVar(value=5)
	# 	self.performance = tk.DoubleVar(value=5)
	# 	self.effort = tk.DoubleVar(value=5)
	# 	self.frustration = tk.DoubleVar(value=5)
	# 	# Create file path
	# 	current_dir = os.path.dirname(os.path.abspath(__file__))
	# 	self.params_file = os.path.join(current_dir, 'params', 'params.txt')
	# 	self.log_file = os.path.join(current_dir, 'BayesianLog', config.logname)
		
	# 	# Configure styles for a modern look
	# 	self._configure_styles()
		
	# 	# Create the user interface
	# 	self._create_ui()
		
	# 	# Initialize the optimizer
	# 	self._initialize_optimizer()

	# 	self.assigner = RandomOutputAssigner()
	# 	self.AFflag = 0

	# def _configure_styles(self):
	# 	"""Configure ttk styles for the application for a better look and feel."""
	# 	self.style = ttk.Style(self.root)
	# 	self.base_font = ('Noto Sans', 20, 'normal')
	# 	self.title_font = ('Noto Sans', 24, 'bold')
	# 	self.small_font = ('Noto Sans', 18)

	# 	self.style.theme_use('clam')

	# 	self.style.configure('TFrame', background='#f0f0f0')
	# 	self.style.configure('TLabel', font=self.base_font, background='#f0f0f0')
	# 	self.style.configure('TButton', font=('Noto Sans', 20), padding=12)
	# 	self.style.configure('TLabelframe', font=self.base_font, padding=20, background='#f0f0f0')
	# 	self.style.configure('TLabelframe.Label', font=self.title_font, background='#f0f0f0')
	# 	self.style.configure('TNotebook', font=self.base_font, padding=10)
	# 	self.style.configure('TNotebook.Tab', font=('Noto Sans', 22), padding=[20, 10])
	# 	self.style.configure('TProgressbar', thickness=36)


	# def _create_ui(self):
	# 	# Create the main frame
	# 	main_frame = ttk.Frame(self.root, padding="24")
	# 	main_frame.pack(fill=tk.BOTH, expand=True)
		
	# 	# Create the tab control
	# 	self.tab_control = ttk.Notebook(main_frame)
	# 	tab1 = ttk.Frame(self.tab_control, padding=20)  # Optimization Settings & Progress
	# 	tab2 = ttk.Frame(self.tab_control, padding=20)  # NASA-TLX Rating
	# 	tab3 = ttk.Frame(self.tab_control, padding=20)  # Results Display
		
	# 	self.tab_control.add(tab1, text="Optimization Settings")
	# 	self.tab_control.add(tab2, text="Subjective Rating")
	# 	self.tab_control.add(tab3, text="Results Display")
	# 	self.tab_control.pack(expand=True, fill=tk.BOTH)
		
	# 	# Tab 1: Optimization Settings
	# 	self._create_optimization_tab(tab1)
		
	# 	# Tab 2: NASA-TLX Rating
	# 	self._create_nasa_tlx_tab(tab2)
		
	# 	# Tab 3: Results Display
	# 	self._create_results_tab(tab3)
		
	# 	# Bottom control buttons
	# 	control_frame = ttk.Frame(main_frame)
	# 	control_frame.pack(fill=tk.X, pady=16, side=tk.BOTTOM)
		
	# 	# Align buttons to the right
	# 	self.stop_button = ttk.Button(control_frame, text="Stop", command=self.stop_optimization, state=tk.DISABLED)
	# 	self.stop_button.pack(side=tk.RIGHT, padx=12)
		
	# 	self.next_button = ttk.Button(control_frame, text="Next Step", command=self.next_step, state=tk.DISABLED)
	# 	self.next_button.pack(side=tk.RIGHT, padx=12)

	# 	self.start_button = ttk.Button(control_frame, text="Start Optimization", command=self.start_optimization)
	# 	self.start_button.pack(side=tk.RIGHT, padx=12)

	# def _create_optimization_tab(self, parent):
	# 	global mode
	# 	parent.columnconfigure(0, weight=1)

	# 	# Optimization settings UI
	# 	settings_frame = ttk.LabelFrame(parent, text="Optimization Settings")
	# 	settings_frame.grid(row=0, column=0, sticky="ew", padx=16, pady=12)
		
	# 	# Number of iterations setting
	# 	ttk.Label(settings_frame, text="Number of Iterations:").grid(row=0, column=0, sticky=tk.W, padx=10, pady=10)
	# 	# set initial iteration times
	# 	if mode == 3:
	# 		self.iter_var = tk.IntVar(value=10)
	# 	else:
	# 		self.iter_var = tk.IntVar(value=config.iter_times)
	# 	iter_entry = ttk.Spinbox(settings_frame, from_=1, to=100, textvariable=self.iter_var, width=10, font=self.base_font)
	# 	iter_entry.grid(row=0, column=1, sticky=tk.W, padx=10, pady=10)
		
	# 	# Progress information
	# 	progress_frame = ttk.LabelFrame(parent, text="Optimization Progress")
	# 	progress_frame.grid(row=1, column=0, sticky="ew", padx=16, pady=12)
	# 	progress_frame.columnconfigure(1, weight=1)
		
	# 	ttk.Label(progress_frame, text="Current Iteration:").grid(row=0, column=0, sticky=tk.W, padx=10, pady=10)
	# 	self.current_iter_label = ttk.Label(progress_frame, text="0/0", font=self.base_font)
	# 	self.current_iter_label.grid(row=0, column=1, sticky=tk.W, padx=10, pady=10)
		
	# 	# Progress bar
	# 	ttk.Label(progress_frame, text="Overall Progress:").grid(row=1, column=0, sticky=tk.W, padx=10, pady=10)
	# 	self.progress_bar = ttk.Progressbar(progress_frame, orient=tk.HORIZONTAL, length=800, mode='determinate')
	# 	self.progress_bar.grid(row=1, column=1, columnspan=3, sticky="ew", padx=10, pady=10)
		
	# 	# Current parameters
	# 	params_frame = ttk.LabelFrame(parent, text="Current Test Parameters")
	# 	params_frame.grid(row=2, column=0, sticky="nsew", padx=16, pady=12)
	# 	parent.rowconfigure(2, weight=1)
		
	# 	self.params_text = tk.Text(params_frame, height=15, width=60, font=self.base_font)
	# 	self.params_text.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
	# 	self.params_text.config(state=tk.DISABLED, background='#ffffff')
	
	# def _create_nasa_tlx_tab(self, parent):
	# 	parent.columnconfigure(0, weight=1)

	# 	# NASA-TLX rating UI
	# 	nasa_frame = ttk.LabelFrame(parent, text="NASA-TLX Subjective Rating")
	# 	nasa_frame.grid(row=0, column=0, sticky="ew", padx=16, pady=12)
		
	# 	# Instruction text
	# 	instr_text = "Please rate the following items based on your teleoperation experience (0-10):"
	# 	ttk.Label(nasa_frame, text=instr_text, font=self.title_font).pack(anchor=tk.W, pady=(10, 24))
		
	# 	# Rating scales
	# 	self._create_scale(nasa_frame, "1. Physical Demand (0=easy, 10=difficult):", self.physical_demand)
	# 	self._create_scale(nasa_frame, "2. Temporal Demand (0=easy, 10=difficult):", self.temporal_demand)
	# 	self._create_scale(nasa_frame, "3. Controllability (0=good, 10=poor):", self.controllability)
	# 	self._create_scale(nasa_frame, "4. Performance (0=good, 10=poor):", self.performance)
	# 	self._create_scale(nasa_frame, "5. Mental Demand (0=easy, 10=difficult):", self.mental_demand)
	# 	self._create_scale(nasa_frame, "6. Effort (0=easy, 10=difficult):", self.effort)
	# 	self._create_scale(nasa_frame, "7. Frustration/Distractions (0=low, 10=high):", self.frustration)
		
	# 	# Submit button
	# 	submit_button = ttk.Button(nasa_frame, text="Submit Ratings", command=self.submit_scores)
	# 	submit_button.pack(pady=30)
	
	# def _create_scale(self, parent, label_text, variable):
	# 	frame = ttk.Frame(parent)
	# 	frame.pack(fill=tk.X, pady=18, padx=16)
	# 	frame.columnconfigure(1, weight=1)

	# 	label = ttk.Label(frame, text=label_text, width=50, anchor="w")
	# 	label.grid(row=0, column=0, sticky="w")

	# 	scale = tk.Scale(frame, from_=0, to=10, orient=tk.HORIZONTAL,
	# 					 variable=variable, length=1000, resolution=0.1,
	# 					 width=36, sliderlength=50, showvalue=False,
	# 					 bg="#f0f0f0", troughcolor="#ccc", highlightthickness=0,
	# 					 font=self.small_font)
	# 	scale.grid(row=0, column=1, padx=10, sticky="ew")

	# 	value_label = ttk.Label(frame, text="5.0", width=6, anchor="e")

	# 	def update_label(*args):
	# 		value_label.config(text=f"{variable.get():.1f}")
	# 	variable.trace_add("write", update_label)
	# 	value_label.grid(row=0, column=2, padx=(12, 16), sticky="e")

	# def _create_results_tab(self, parent):
	# 	parent.columnconfigure(0, weight=1)
	# 	parent.columnconfigure(1, weight=1)
	# 	parent.rowconfigure(1, weight=1)

	# 	# Performance metrics display
	# 	metrics_frame = ttk.LabelFrame(parent, text="Performance Metrics")
	# 	metrics_frame.grid(row=0, column=0, padx=16, pady=12, sticky="nsew")
		
	# 	metrics = [
	# 		("Gracefulness:", "gracefulness_value"), 
	# 		("Smoothness:", "smoothness_value"),
	# 		("Clutch Times:", "clutch_times_value"),
	# 		("Total Distance:", "total_distance_value"),
	# 		("Total Time:", "total_time_value")
	# 	]
		
	# 	for i, (label_text, attr_name) in enumerate(metrics):
	# 		ttk.Label(metrics_frame, text=label_text).grid(row=i, column=0, sticky=tk.W, padx=12, pady=8)
	# 		setattr(self, attr_name, ttk.Label(metrics_frame, text="-", font=self.base_font + ('bold',)))
	# 		getattr(self, attr_name).grid(row=i, column=1, sticky=tk.W, padx=12, pady=8)
		
	# 	# Scores display
	# 	scores_frame = ttk.LabelFrame(parent, text="Scores")
	# 	scores_frame.grid(row=0, column=1, padx=16, pady=12, sticky="nsew")
		
	# 	scores = [
	# 		("Gracefulness Score (10):", "gracefulness_score"), 
	# 		("Smoothness Score (10):", "smoothness_score"),
	# 		("Clutch Times Score (30):", "clutch_times_score"),
	# 		("Total Distance Score (40):", "total_distance_score"),
	# 		("Total Time Score (20):", "total_time_score"),
	# 		("Total Score:", "total_score")
	# 	]
		
	# 	for i, (label_text, attr_name) in enumerate(scores):
	# 		ttk.Label(scores_frame, text=label_text).grid(row=i, column=0, sticky=tk.W, padx=12, pady=8)
	# 		setattr(self, attr_name, ttk.Label(scores_frame, text="-", font=self.base_font + ('bold',)))
	# 		getattr(self, attr_name).grid(row=i, column=1, sticky=tk.W, padx=12, pady=8)
		
	# 	# Best results display
	# 	best_frame = ttk.LabelFrame(parent, text="Best Result")
	# 	best_frame.grid(row=1, column=0, columnspan=2, padx=16, pady=12, sticky="nsew")
		
	# 	self.best_results_text = tk.Text(best_frame, height=10, width=60, font=self.base_font)
	# 	self.best_results_text.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
	# 	self.best_results_text.config(state=tk.DISABLED, background='#ffffff')

	def __init__(self, root):
		self.root = root
		self.root.title("Bayesian Optimization System")
		# 适配 1080p 屏幕的初始窗口大小及位置，预留任务栏空间
		self.root.geometry("2560x1440+2560+0")
		self.root.minsize(1024, 768)
		
		# Initialize variables
		self.current_iteration = 0
		self.max_iterations = 10
		self.optimizer = None
		self.next_point = None
		self.mental_demand = tk.DoubleVar(value=5)
		self.physical_demand = tk.DoubleVar(value=5)
		self.controllability = tk.DoubleVar(value=5)
		self.temporal_demand = tk.DoubleVar(value=5)
		self.performance = tk.DoubleVar(value=5)
		self.effort = tk.DoubleVar(value=5)
		self.frustration = tk.DoubleVar(value=5)
		# Create file path
		current_dir = os.path.dirname(os.path.abspath(__file__))
		self.params_file = os.path.join(current_dir, 'params', 'params.txt')
		self.log_file = os.path.join(current_dir, 'BayesianLog', config.logname)
		
		# Configure styles for a modern look
		self._configure_styles()
		
		# Create the user interface
		self._create_ui()
		
		# Initialize the optimizer
		self._initialize_optimizer()

		self.assigner = RandomOutputAssigner()
		self.AFflag = 0

	def _configure_styles(self):
		"""Configure ttk styles for the application for a better look and feel."""
		self.style = ttk.Style(self.root)
		# 全局下调字号以适配 1080p
		self.base_font = ('Noto Sans', 13, 'normal')
		self.title_font = ('Noto Sans', 16, 'bold')
		self.small_font = ('Noto Sans', 11)

		self.style.theme_use('clam')

		self.style.configure('TFrame', background='#f0f0f0')
		self.style.configure('TLabel', font=self.base_font, background='#f0f0f0')
		self.style.configure('TButton', font=('Noto Sans', 13), padding=6) # 压缩按钮内边距
		self.style.configure('TLabelframe', font=self.base_font, padding=10, background='#f0f0f0') # 压缩框架内边距
		self.style.configure('TLabelframe.Label', font=self.title_font, background='#f0f0f0')
		self.style.configure('TNotebook', font=self.base_font, padding=6)
		self.style.configure('TNotebook.Tab', font=('Noto Sans', 14), padding=[12, 6])
		self.style.configure('TProgressbar', thickness=20) # 减小进度条厚度


	def _create_ui(self):
		# Create the main frame
		main_frame = ttk.Frame(self.root, padding="10")
		main_frame.pack(fill=tk.BOTH, expand=True)
		
		# Create the tab control
		self.tab_control = ttk.Notebook(main_frame)
		tab1 = ttk.Frame(self.tab_control, padding=10)  # Optimization Settings & Progress
		tab2 = ttk.Frame(self.tab_control, padding=10)  # NASA-TLX Rating
		tab3 = ttk.Frame(self.tab_control, padding=10)  # Results Display
		
		self.tab_control.add(tab1, text="Optimization Settings")
		self.tab_control.add(tab2, text="Subjective Rating")
		self.tab_control.add(tab3, text="Results Display")
		self.tab_control.pack(expand=True, fill=tk.BOTH)
		
		# Tab 1: Optimization Settings
		self._create_optimization_tab(tab1)
		
		# Tab 2: NASA-TLX Rating
		self._create_nasa_tlx_tab(tab2)
		
		# Tab 3: Results Display
		self._create_results_tab(tab3)
		
		# Bottom control buttons
		control_frame = ttk.Frame(main_frame)
		control_frame.pack(fill=tk.X, pady=10, side=tk.BOTTOM)
		
		# Align buttons to the right
		self.stop_button = ttk.Button(control_frame, text="Stop", command=self.stop_optimization, state=tk.DISABLED)
		self.stop_button.pack(side=tk.RIGHT, padx=12)
		
		self.next_button = ttk.Button(control_frame, text="Next Step", command=self.next_step, state=tk.DISABLED)
		self.next_button.pack(side=tk.RIGHT, padx=12)

		self.start_button = ttk.Button(control_frame, text="Start Optimization", command=self.start_optimization)
		self.start_button.pack(side=tk.RIGHT, padx=12)

	def _create_optimization_tab(self, parent):
		global mode
		parent.columnconfigure(0, weight=1)

		# Optimization settings UI
		settings_frame = ttk.LabelFrame(parent, text="Optimization Settings")
		settings_frame.grid(row=0, column=0, sticky="ew", padx=10, pady=6)
		
		# Number of iterations setting
		ttk.Label(settings_frame, text="Number of Iterations:").grid(row=0, column=0, sticky=tk.W, padx=10, pady=5)
		# set initial iteration times
		if mode == 3:
			self.iter_var = tk.IntVar(value=10)
		else:
			self.iter_var = tk.IntVar(value=config.iter_times)
		iter_entry = ttk.Spinbox(settings_frame, from_=1, to=100, textvariable=self.iter_var, width=10, font=self.base_font)
		iter_entry.grid(row=0, column=1, sticky=tk.W, padx=10, pady=5)
		
		# Progress information
		progress_frame = ttk.LabelFrame(parent, text="Optimization Progress")
		progress_frame.grid(row=1, column=0, sticky="ew", padx=10, pady=6)
		progress_frame.columnconfigure(1, weight=1)
		
		ttk.Label(progress_frame, text="Current Iteration:").grid(row=0, column=0, sticky=tk.W, padx=10, pady=5)
		self.current_iter_label = ttk.Label(progress_frame, text="0/0", font=self.base_font)
		self.current_iter_label.grid(row=0, column=1, sticky=tk.W, padx=10, pady=5)

		# 阶段指示标签
		ttk.Label(progress_frame, text="Current Phase:").grid(row=0, column=2, sticky=tk.W, padx=20, pady=5)
		self.phase_label = ttk.Label(progress_frame,
			text="Phase 1 — Optimizing: K_g, C_base",
			font=self.base_font, foreground='#1a6fbd')
		self.phase_label.grid(row=0, column=3, sticky=tk.W, padx=10, pady=5)

		# Progress bar
		ttk.Label(progress_frame, text="Overall Progress:").grid(row=1, column=0, sticky=tk.W, padx=10, pady=5)
		self.progress_bar = ttk.Progressbar(progress_frame, orient=tk.HORIZONTAL, length=500, mode='determinate') # 减小进度条长度
		self.progress_bar.grid(row=1, column=1, columnspan=3, sticky="ew", padx=10, pady=5)
		
		# Current parameters
		params_frame = ttk.LabelFrame(parent, text="Current Test Parameters")
		params_frame.grid(row=2, column=0, sticky="nsew", padx=10, pady=6)
		parent.rowconfigure(2, weight=1)
		
		# 减小文本显示区域的高度
		self.params_text = tk.Text(params_frame, height=10, width=60, font=self.base_font) 
		self.params_text.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
		self.params_text.config(state=tk.DISABLED, background='#ffffff')
	
	def _create_nasa_tlx_tab(self, parent):
		parent.columnconfigure(0, weight=1)

		# NASA-TLX rating UI
		nasa_frame = ttk.LabelFrame(parent, text="NASA-TLX Subjective Rating")
		nasa_frame.grid(row=0, column=0, sticky="ew", padx=10, pady=6)
		
		# Instruction text
		instr_text = "Please rate the following items based on your teleoperation experience (0-10):"
		ttk.Label(nasa_frame, text=instr_text, font=self.title_font).pack(anchor=tk.W, pady=(6, 12))
		
		# Rating scales
		self._create_scale(nasa_frame, "1. Physical Demand (0=easy, 10=difficult):", self.physical_demand)
		self._create_scale(nasa_frame, "2. Temporal Demand (0=easy, 10=difficult):", self.temporal_demand)
		self._create_scale(nasa_frame, "3. Controllability (0=good, 10=poor):", self.controllability)
		self._create_scale(nasa_frame, "4. Performance (0=good, 10=poor):", self.performance)
		self._create_scale(nasa_frame, "5. Mental Demand (0=easy, 10=difficult):", self.mental_demand)
		self._create_scale(nasa_frame, "6. Effort (0=easy, 10=difficult):", self.effort)
		self._create_scale(nasa_frame, "7. Frustration/Distractions (0=low, 10=high):", self.frustration)
		
		# Submit button
		submit_button = ttk.Button(nasa_frame, text="Submit Ratings", command=self.submit_scores)
		submit_button.pack(pady=15)
	
	def _create_scale(self, parent, label_text, variable):
		frame = ttk.Frame(parent)
		frame.pack(fill=tk.X, pady=4, padx=10) # 极限压缩每一行的间距
		frame.columnconfigure(1, weight=1)

		label = ttk.Label(frame, text=label_text, width=45, anchor="w")
		label.grid(row=0, column=0, sticky="w")

		# 将长度缩减至 600，厚度缩减至 14，滑块长度缩减至 30
		scale = tk.Scale(frame, from_=0, to=10, orient=tk.HORIZONTAL,
						 variable=variable, length=600, resolution=0.1,
						 width=28, sliderlength=30, showvalue=False,
						 bg="#f0f0f0", troughcolor="#ccc", highlightthickness=0,
						 font=self.small_font)
		scale.grid(row=0, column=1, padx=10, sticky="ew")

		value_label = ttk.Label(frame, text="5.0", width=5, anchor="e")

		def update_label(*args):
			value_label.config(text=f"{variable.get():.1f}")
		variable.trace_add("write", update_label)
		value_label.grid(row=0, column=2, padx=(10, 10), sticky="e")

	def _create_results_tab(self, parent):
		parent.columnconfigure(0, weight=1)
		parent.columnconfigure(1, weight=1)
		parent.rowconfigure(1, weight=1)

		# Performance metrics display
		metrics_frame = ttk.LabelFrame(parent, text="Performance Metrics")
		metrics_frame.grid(row=0, column=0, padx=10, pady=6, sticky="nsew")
		
		metrics = [
			("Gracefulness:", "gracefulness_value"), 
			("Smoothness:", "smoothness_value"),
			("Clutch Times:", "clutch_times_value"),
			("Total Distance:", "total_distance_value"),
			("Total Time:", "total_time_value")
		]
		
		for i, (label_text, attr_name) in enumerate(metrics):
			ttk.Label(metrics_frame, text=label_text).grid(row=i, column=0, sticky=tk.W, padx=10, pady=5)
			setattr(self, attr_name, ttk.Label(metrics_frame, text="-", font=self.base_font + ('bold',)))
			getattr(self, attr_name).grid(row=i, column=1, sticky=tk.W, padx=10, pady=5)
		
		# Scores display
		scores_frame = ttk.LabelFrame(parent, text="Scores")
		scores_frame.grid(row=0, column=1, padx=10, pady=6, sticky="nsew")
		
		scores = [
			("Gracefulness Score (10):", "gracefulness_score"), 
			("Smoothness Score (10):", "smoothness_score"),
			("Clutch Times Score (30):", "clutch_times_score"),
			("Total Distance Score (40):", "total_distance_score"),
			("Total Time Score (20):", "total_time_score"),
			("Total Score:", "total_score")
		]
		
		for i, (label_text, attr_name) in enumerate(scores):
			ttk.Label(scores_frame, text=label_text).grid(row=i, column=0, sticky=tk.W, padx=10, pady=5)
			setattr(self, attr_name, ttk.Label(scores_frame, text="-", font=self.base_font + ('bold',)))
			getattr(self, attr_name).grid(row=i, column=1, sticky=tk.W, padx=10, pady=5)
		
		# Best results display
		best_frame = ttk.LabelFrame(parent, text="Best Result")
		best_frame.grid(row=1, column=0, columnspan=2, padx=10, pady=6, sticky="nsew")
		
		# 减小文本显示区域高度
		self.best_results_text = tk.Text(best_frame, height=6, width=60, font=self.base_font)
		self.best_results_text.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
		self.best_results_text.config(state=tk.DISABLED, background='#ffffff')
	
	def _initialize_optimizer(self):
		global mode

		# ── 两阶段参数分组 ──────────────────────────────────────────────────────
		# Phase 1 (前 70%)：只优化 K_g 和 C_base
		self.pbounds_phase1 = {
			k: v for k, v in config.optimization_range.items()
			if k in ('K_g', 'C_base')
		}
		# Phase 2 (后 30%)：只优化其余参数，K_g / C_base 锁定为 Phase 1 最优值
		self.pbounds_phase2 = {
			k: v for k, v in config.optimization_range.items()
			if k in ('A_theta', 'A_gp', 'A_v', 'B_safety')
		}

		# 阶段状态（start_optimization 时会重置）
		self.current_phase   = 1
		self.phase1_iters    = 1
		self.phase2_iters    = 1
		self.best_phase1_params = None   # Phase 1 结束时从 optimizer_phase1.max 写入

		if mode != 3:
			log_p1 = self.log_file.replace('.json', '_phase1.json')
			log_p2 = self.log_file.replace('.json', '_phase2.json')

			# ── Phase 1 优化器 ────────────────────────────────────────────────
			self.optimizer_phase1 = BayesianOptimization(
				f=None,
				pbounds=self.pbounds_phase1,
				verbose=0,
				random_state=1,
				bounds_transformer=SequentialDomainReductionTransformer(minimum_window=0.1),
			)
			if os.path.exists(log_p1):
				load_logs(self.optimizer_phase1, logs=[log_p1])
				print(f"<LYON> Phase-1 optimizer loaded {len(self.optimizer_phase1.space)} points.")
			else:
				print("<LYON> Phase-1: no past logs, starting fresh.")
			self.optimizer_phase1.subscribe(
				Events.OPTIMIZATION_STEP, JSONLogger(path=log_p1, reset=False)
			)

			# ── Phase 2 优化器 ────────────────────────────────────────────────
			self.optimizer_phase2 = BayesianOptimization(
				f=None,
				pbounds=self.pbounds_phase2,
				verbose=0,
				random_state=1,
				bounds_transformer=SequentialDomainReductionTransformer(minimum_window=0.1),
			)
			if os.path.exists(log_p2):
				load_logs(self.optimizer_phase2, logs=[log_p2])
				print(f"<LYON> Phase-2 optimizer loaded {len(self.optimizer_phase2.space)} points.")
			else:
				print("<LYON> Phase-2: no past logs, starting fresh.")
			self.optimizer_phase2.subscribe(
				Events.OPTIMIZATION_STEP, JSONLogger(path=log_p2, reset=False)
			)

			# self.optimizer 始终指向当前活跃优化器（供 stop 等位置使用）
			self.optimizer = self.optimizer_phase1

		# Set utility function
		self.utility = UtilityFunction(kind="ei", xi=0.0)

		# Define maximum values
		self.gracefulness_max = config.scoreParams_bound['gracefulness_max']
		self.gracefulness_min = config.scoreParams_bound['gracefulness_min']
		self.smoothness_max   = config.scoreParams_bound['smoothness_max']
		self.smoothness_min   = config.scoreParams_bound['smoothness_min']
		self.clutch_times_max = config.scoreParams_bound['clutch_times_max']
		self.total_distance_max = config.scoreParams_bound['total_distance_max']
		self.total_time_max   = config.scoreParams_bound['total_time_max']
	
	def start_optimization(self):
		"""Starts the optimization process."""
		self.max_iterations = self.iter_var.get()
		self.current_iteration = 0

		# ── 两阶段边界计算 ────────────────────────────────────────────────────
		# 前 70% 迭代优化 K_g / C_base，后 30% 优化其余参数
		self.phase1_iters = max(1, int(self.max_iterations * config.phase_rate))
		self.phase2_iters = self.max_iterations - self.phase1_iters
		self.current_phase      = 1
		self.best_phase1_params = None
		if mode != 3:
			self.optimizer = self.optimizer_phase1

		self.progress_bar['maximum'] = self.max_iterations
		self.progress_bar['value'] = 0
		self.current_iter_label.config(text=f"0/{self.max_iterations}")
		self.phase_label.config(
			text=f"Phase 1 ({self.phase1_iters} iters) — Optimizing: K_g, C_base",
			foreground='#1a6fbd',
		)

		self.start_button.config(state=tk.DISABLED)
		self.next_button.config(state=tk.NORMAL)
		self.stop_button.config(state=tk.NORMAL)

		self.next_step()
	
	def next_step(self):
		"""Executes the next optimization step."""
		if self.current_iteration < self.max_iterations:
			self.current_iteration += 1
			self.current_iter_label.config(text=f"{self.current_iteration}/{self.max_iterations}")
			self.progress_bar['value'] = self.current_iteration

			if mode == 3:
				adaptive_params = config.adaptive
				adaptive_params["AFflag"] = 0
				fixed_params = config.fixed
				output = self.assigner.assign_random_output(adaptive=adaptive_params, fixed=fixed_params)
				self.AFflag = output["AFflag"]
				self.save_params_to_txt(output["params"])
			else:
				# ── 判断当前处于哪个阶段 ─────────────────────────────────────
				if self.current_iteration <= self.phase1_iters:
					# Phase 1：优化 K_g / C_base，其余取 init_params 初始值
					self.current_phase  = 1
					active_opt          = self.optimizer_phase1
					self.optimizer      = active_opt
					fixed_complement    = {
						k: config.init_params[k]
						for k in ('A_theta', 'A_gp', 'A_v', 'B_safety')
					}
					phase_label_text = (
						f"Phase 1 iter {self.phase1_iters}  —  "
						f"Optimizing: K_g, C_base "
					)
					phase_color = '#1a6fbd'
				else:
					# ── Phase 1 → Phase 2 过渡：锁定 Phase 1 最优参数 ────────
					if self.best_phase1_params is None:
						if len(self.optimizer_phase1.res) > 0:
							self.best_phase1_params = dict(self.optimizer_phase1.max['params'])
						else:
							self.best_phase1_params = {
								k: config.init_params[k] for k in ('K_g', 'C_base')
							}
						print(f"<LYON> Phase 1 ended. Best params locked: {self.best_phase1_params}")

					# Phase 2：优化 A_theta / A_gp / A_v / B_safety，K_g / C_base 锁定
					self.current_phase  = 2
					active_opt          = self.optimizer_phase2
					self.optimizer      = active_opt
					fixed_complement    = self.best_phase1_params
					p2_iter             = self.current_iteration - self.phase1_iters
					phase_label_text = (
						f"Phase 2 iter {self.phase2_iters}  —  "
						f"Optimizing: A_theta, A_gp, A_v, B_safety  |  "
						f"Fixed: K_g={fixed_complement['K_g']:.4f}, "
						f"C_base={fixed_complement['C_base']:.4f}"
					)
					phase_color = '#b06000'

				self.phase_label.config(text=phase_label_text, foreground=phase_color)

				# 从活跃优化器获取下一个探测点（仅含本阶段待优化参数）
				self.next_point = active_opt.suggest(self.utility)

				# 构造写入 params.txt 的完整参数字典
				self.next_point_full = {**self.next_point, **fixed_complement}

				# 更新参数显示
				self.params_text.config(state=tk.NORMAL)
				self.params_text.delete(1.0, tk.END)
				self.params_text.insert(
					tk.END,
					f"Iteration {self.current_iteration}/{self.max_iterations}"
					f"  (Phase {self.current_phase})\n\n"
				)
				self.params_text.insert(tk.END, "[Optimizing]\n")
				for key in self.next_point:
					self.params_text.insert(tk.END, f"  {key:<12} = {self.next_point_full[key]:.6f}\n")
				self.params_text.insert(tk.END, "\n[Fixed]\n")
				for key in fixed_complement:
					self.params_text.insert(tk.END, f"  {key:<12} = {self.next_point_full[key]:.6f}\n")
				self.params_text.config(state=tk.DISABLED)

				# 保存完整参数到文件（main.py 从此文件读取所有参数）
				self.save_params_to_txt(self.next_point_full)

			# 切换到 NASA-TLX 评分 Tab
			self.tab_control.select(1)
		else:
			# 优化完成
			messagebox.showinfo("Optimization Complete", "The Bayesian optimization process has finished!")
			self.next_button.config(state=tk.DISABLED)
			self.stop_button.config(state=tk.DISABLED)
			self.start_button.config(state=tk.NORMAL)

			if mode != 3:
				self.show_best_result()
	
	def submit_scores(self):
		"""Submits NASA-TLX scores and calculates the target value."""
		# Calculate subjective score
		mental_demand = 1-self.mental_demand.get()/10
		physical_demand = 1-self.physical_demand.get()/10	
		temporal_demand = 1-self.temporal_demand.get()/10
		controllability = 1-self.controllability.get()/10
		performance = 1-self.performance.get()/10
		effort = 1-self.effort.get()/10
		frustration = 1-self.frustration.get()/10
		
		sub_score = 15*physical_demand + 15*temporal_demand + 30*controllability + 30*performance + 3*mental_demand + 3*effort + 4*frustration
		
		# Switch to the results tab
		self.tab_control.select(2)  # Select the 3rd tab (Results Display)
		
		# Calculate other metrics
		self.save_and_display_metrics(sub_score)
	
	@staticmethod
	def calculate_metrics():
		"""
		计算性能指标（静态方法，可以不实例化类直接调用）
		Call the original performance calculation function
		"""
		gracefulness, smoothness = cal_GS()
		current_file_path = os.path.abspath(__file__)
		current_dir = os.path.dirname(current_file_path)
		data_base_dir = os.path.join(current_dir, 'data')
		latest_dir = get_latest_data_dir(data_base_dir)
		clutch_times = np.load(os.path.join(latest_dir, 'clutch_times.npy'), allow_pickle=True)
		total_distance = np.load(os.path.join(latest_dir, 'total_distance.npy'), allow_pickle=True)
		total_time = np.load(os.path.join(latest_dir, 'total_time.npy'), allow_pickle=True)[0]
		
		return gracefulness, smoothness, clutch_times, total_distance, total_time


	def save_and_display_metrics(self, sub_score):
		"""Calculates and displays performance metrics and scores."""
		
		gracefulness, smoothness, clutch_times, total_distance, total_time = self.calculate_metrics()
		# Calculate individual scores
		# group1 G S
		gracefulness_score =  5 * np.clip((self.gracefulness_max - gracefulness) / (self.gracefulness_max - self.gracefulness_min), 0, 1)
		smoothness_score = 5 * np.clip((self.smoothness_max - smoothness) / (self.smoothness_max - self.smoothness_min), 0, 1)
		# group2 fast
		clutch_times_score = 30 * np.clip((self.clutch_times_max - clutch_times[0] - clutch_times[1]) / self.clutch_times_max, 0, 1) 
		total_time_score = 20 * np.clip((self.total_time_max - total_time) / self.total_time_max, 0, 1)
		#group3 slow
		total_distance_score = 40 * np.clip((self.total_distance_max - total_distance[0]) / self.total_distance_max, 0, 1)
		

		obj_score = gracefulness_score + smoothness_score + clutch_times_score + total_distance_score + total_time_score
		
		# Calculate the total score
		total_score = 0.5 * obj_score + 0.5 * sub_score

		current_file_path = os.path.abspath(__file__)
		current_dir = os.path.dirname(current_file_path)
		output_json_path = os.path.join(current_dir, 'BayesianLog',config.scorefilename)

		new_entry = {
			"AFflag": float(self.AFflag),
			"subscore": float(sub_score),
			"gracefulness": float(gracefulness),
			"smoothness": float(smoothness),
			"Lclutch_times": float(clutch_times[0]),
			"Rclutch_times": float(clutch_times[1]),
			"total_distance": float(total_distance),
			"total_time": float(total_time),
			"gracefulness_score": float(gracefulness_score),
			"smoothness_score": float(smoothness_score),
			"clutch_times_score": float(clutch_times_score),
			"total_distance_score": float(total_distance_score),
			"total_time_score": float(total_time_score),
			"total_score": float(total_score),
			"fixed_scale": float(config.fixed["fixed_scale"]) if mode == 3 else None,
		}
		
		existing_data = []
		if os.path.exists(output_json_path):
			try:
				with open(output_json_path, 'r', encoding='utf-8') as f:
					existing_data = json.load(f)
			except json.JSONDecodeError:
				
				existing_data = []
			except Exception as e:
				print(f"JSON error: {e}")
				existing_data = []

		existing_data.append(new_entry)

		try:
			with open(output_json_path, 'w', encoding='utf-8') as f:
				json.dump(existing_data, f, ensure_ascii=False, indent=4)
			print(f"scores are saved in : {output_json_path}")
		except Exception as e:
			print(f"JSON error: {e}")
	

		# Update the UI display
		self.gracefulness_value.config(text=f"{gracefulness:.4f}")
		self.smoothness_value.config(text=f"{smoothness:.4f}")
		self.clutch_times_value.config(text=f"{clutch_times[0]:.4f},{clutch_times[1]:.4f}")
		self.total_distance_value.config(text=f"{total_distance[0]:.4f}")
		self.total_time_value.config(text=f"{total_time:.4f}")
		
		self.gracefulness_score.config(text=f"{gracefulness_score:.4f}")
		self.smoothness_score.config(text=f"{smoothness_score:.4f}")
		self.clutch_times_score.config(text=f"{clutch_times_score:.4f}")
		self.total_distance_score.config(text=f"{total_distance_score:.4f}")
		self.total_time_score.config(text=f"{total_time_score:.4f}")
		self.total_score.config(text=f"{total_score:.4f}")
		
		if mode != 3:
			# 向当前阶段的优化器注册结果（只传入该阶段负责的参数子集）
			if self.current_phase == 1:
				self.optimizer_phase1.register(params=self.next_point, target=total_score)
			else:
				self.optimizer_phase2.register(params=self.next_point, target=total_score)

			# 控制按钮状态
			if self.current_iteration < self.max_iterations:
				self.next_button.config(state=tk.NORMAL)
			else:
				self.show_best_result()
				self.next_button.config(state=tk.DISABLED)
				self.stop_button.config(state=tk.DISABLED)
				self.start_button.config(state=tk.NORMAL)

	def show_best_result(self):
		"""Displays the best result of the two-phase optimization."""
		self.best_results_text.config(state=tk.NORMAL)
		self.best_results_text.delete(1.0, tk.END)

		# ── Phase 1 最优（K_g / C_base）────────────────────────────────────
		if len(self.optimizer_phase1.res) > 0:
			best_p1       = dict(self.optimizer_phase1.max['params'])
			best_score_p1 = self.optimizer_phase1.max['target']
		else:
			best_p1       = {k: config.init_params[k] for k in ('K_g', 'C_base')}
			best_score_p1 = float('nan')

		# ── Phase 2 最优（A_theta / A_gp / A_v / B_safety）─────────────────
		if len(self.optimizer_phase2.res) > 0:
			best_p2       = dict(self.optimizer_phase2.max['params'])
			best_score_p2 = self.optimizer_phase2.max['target']
		else:
			best_p2       = {k: config.init_params[k] for k in ('A_theta', 'A_gp', 'A_v', 'B_safety')}
			best_score_p2 = float('nan')

		# ── 合并最优全参数集 ─────────────────────────────────────────────────
		best_full = {**best_p1, **best_p2}

		self.best_results_text.insert(tk.END, "=== Optimization Complete ===\n\n")

		self.best_results_text.insert(tk.END,
			f"[Phase 1]  Best Score: {best_score_p1:.6f}\n"
			f"  Optimized params (K_g, C_base):\n"
		)
		for k in ('K_g', 'C_base'):
			self.best_results_text.insert(tk.END, f"    {k:<12} : {best_p1[k]:.6f}\n")

		self.best_results_text.insert(tk.END, "\n")
		self.best_results_text.insert(tk.END,
			f"[Phase 2]  Best Score: {best_score_p2:.6f}\n"
			f"  Optimized params (A_theta, A_gp, A_v, B_safety):\n"
		)
		for k in ('A_theta', 'A_gp', 'A_v', 'B_safety'):
			self.best_results_text.insert(tk.END, f"    {k:<12} : {best_p2[k]:.6f}\n")

		self.best_results_text.insert(tk.END, "\n--- Combined Best Parameter Set ---\n")
		for k, v in best_full.items():
			self.best_results_text.insert(tk.END, f"  {k:<12} : {v:.6f}\n")

		self.best_results_text.config(state=tk.DISABLED)
	
	def stop_optimization(self):
		"""Stops the optimization process."""
		if messagebox.askyesno("Confirm", "Are you sure you want to stop the optimization process?"):
			self.next_button.config(state=tk.DISABLED)
			self.stop_button.config(state=tk.DISABLED)
			self.start_button.config(state=tk.NORMAL)
			
			# If there are already some results, show the best one so far
			if hasattr(self, 'optimizer') and self.optimizer and len(self.optimizer.res) > 0 and mode!=3:
				self.show_best_result()
	
	def save_params_to_txt(self, dic):
		"""Saves parameters to a text file."""
		# Ensure the directory exists
		os.makedirs(os.path.dirname(self.params_file), exist_ok=True)
		
		with open(self.params_file, 'w') as f:
			for key, value in dic.items():
				f.write(f"{key}={value}\n")
		
		print(f"Parameters saved to {self.params_file}")

class RandomOutputAssigner:
    def __init__(self):
        # 初始化计数器，记录adaptive和fixed被选择的次数
        self.adaptive_count = 0
        self.fixed_count = 0
        # 初始化一个列表，用于存储接下来10次运行的赋值顺序
        self.assignment_order = []
        self._generate_assignment_order() # 在初始化时生成一次顺序

    def _generate_assignment_order(self):
        """内部方法：生成10次运行的随机赋值顺序，确保adaptive和fixed各5次"""
        self.assignment_order = ['adaptive'] * 5 + ['fixed'] * 5
        random.shuffle(self.assignment_order) # 随机打乱顺序
        # 重置计数器，以便新的10次循环开始
        self.adaptive_count = 0
        self.fixed_count = 0

    def assign_random_output(self, adaptive, fixed):

        # 如果已经完成了10次循环，则重新生成顺序
        if self.adaptive_count + self.fixed_count >= 10:
            self._generate_assignment_order()

        # 从预生成的顺序中获取当前次序的赋值类型
        current_assignment_type = self.assignment_order[self.adaptive_count + self.fixed_count]

        if current_assignment_type == 'adaptive':
            self.adaptive_count += 1
            output = { "params":adaptive, "AFflag": 0}
            print(f"select 'adaptive' ({self.adaptive_count}/5)")
        else: # current_assignment_type == 'fixed'
            self.fixed_count += 1
            output = { "params":fixed, "AFflag": 1}
            print(f"select 'fixed' ({self.fixed_count}/5)")

        return output

def main():
	args = sys.argv[1:]

	if len(args) > 0 :
		global mode
		mode = float(args[0])

	
	root = tk.Tk()
	app = BayesianOptimizationGUI(root)
	root.mainloop()


if __name__ == "__main__":
	main()