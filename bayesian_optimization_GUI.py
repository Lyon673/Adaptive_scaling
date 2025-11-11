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
	def __init__(self, root):
		self.root = root
		self.root.title("Bayesian Optimization System")
		# Increased window size for better layout
		self.root.geometry("1980x1600+0+0")
		
		# Initialize variables
		self.current_iteration = 0
		self.max_iterations = 10
		self.optimizer = None
		self.next_point = None
		self.mental_demand = tk.DoubleVar(value=10)
		self.physical_demand = tk.DoubleVar(value=10)
		self.temporal_demand = tk.DoubleVar(value=10)
		self.performance = tk.DoubleVar(value=10)
		self.effort = tk.DoubleVar(value=10)
		
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
		# Changed the font to Calibri
		self.base_font = ('Noto Sans', 10, 'normal')
		self.title_font = ('Noto Sans', 12, 'bold')

		self.style.theme_use('clam') # A modern theme

		self.style.configure('TFrame', background='#f0f0f0')
		self.style.configure('TLabel', font=self.base_font, background='#f0f0f0')
		self.style.configure('TButton', font=self.base_font, padding=5)
		self.style.configure('TLabelframe', font=self.base_font, padding=10, background='#f0f0f0')
		self.style.configure('TLabelframe.Label', font=self.title_font, background='#f0f0f0')
		self.style.configure('TNotebook', font=self.base_font, padding=5)
		self.style.configure('TNotebook.Tab', font=('Noto Sans', 12), padding=[10, 5])
		self.style.configure('TProgressbar', thickness=20)


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
		self.stop_button.pack(side=tk.RIGHT, padx=5)
		
		self.next_button = ttk.Button(control_frame, text="Next Step", command=self.next_step, state=tk.DISABLED)
		self.next_button.pack(side=tk.RIGHT, padx=5)

		self.start_button = ttk.Button(control_frame, text="Start Optimization", command=self.start_optimization)
		self.start_button.pack(side=tk.RIGHT, padx=5)

	def _create_optimization_tab(self, parent):
		parent.columnconfigure(0, weight=1)

		# Optimization settings UI
		settings_frame = ttk.LabelFrame(parent, text="Optimization Settings")
		settings_frame.grid(row=0, column=0, sticky="ew", padx=10, pady=10)
		
		# Number of iterations setting
		ttk.Label(settings_frame, text="Number of Iterations:").grid(row=0, column=0, sticky=tk.W, padx=5, pady=5)
		# set initial iteration times
		self.iter_var = tk.IntVar(value=3)
		iter_entry = ttk.Spinbox(settings_frame, from_=1, to=100, textvariable=self.iter_var, width=10, font=self.base_font)
		iter_entry.grid(row=0, column=1, sticky=tk.W, padx=5, pady=5)
		
		# Progress information
		progress_frame = ttk.LabelFrame(parent, text="Optimization Progress")
		progress_frame.grid(row=1, column=0, sticky="ew", padx=10, pady=10)
		progress_frame.columnconfigure(1, weight=1) # Make progress bar expandable
		
		ttk.Label(progress_frame, text="Current Iteration:").grid(row=0, column=0, sticky=tk.W, padx=5, pady=5)
		self.current_iter_label = ttk.Label(progress_frame, text="0/0", font=self.base_font)
		self.current_iter_label.grid(row=0, column=1, sticky=tk.W, padx=5, pady=5)
		
		# Progress bar
		ttk.Label(progress_frame, text="Overall Progress:").grid(row=1, column=0, sticky=tk.W, padx=5, pady=5)
		self.progress_bar = ttk.Progressbar(progress_frame, orient=tk.HORIZONTAL, length=400, mode='determinate')
		self.progress_bar.grid(row=1, column=1, columnspan=3, sticky="ew", padx=5, pady=5)
		
		# Current parameters
		params_frame = ttk.LabelFrame(parent, text="Current Test Parameters")
		params_frame.grid(row=2, column=0, sticky="nsew", padx=10, pady=10)
		parent.rowconfigure(2, weight=1) # Allow this frame to expand vertically
		
		self.params_text = tk.Text(params_frame, height=15, width=60, font=self.base_font)
		self.params_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
		self.params_text.config(state=tk.DISABLED, background='#ffffff')
	
	def _create_nasa_tlx_tab(self, parent):
		parent.columnconfigure(0, weight=1)

		# NASA-TLX rating UI
		nasa_frame = ttk.LabelFrame(parent, text="NASA-TLX Subjective Rating")
		nasa_frame.grid(row=0, column=0, sticky="ew", padx=10, pady=10)
		
		# Instruction text
		instr_text = "Please rate the following items based on your teleoperation experience (0-20):"
		ttk.Label(nasa_frame, text=instr_text, font=self.title_font).pack(anchor=tk.W, pady=(5, 15))
		
		# Rating scales
		self._create_scale(nasa_frame, "1. Mental Demand (0=low, 20=high):", self.mental_demand)
		self._create_scale(nasa_frame, "2. Physical Demand (0=low, 20=high):", self.physical_demand)
		self._create_scale(nasa_frame, "3. Temporal Demand (0=low, 20=high):", self.temporal_demand)
		self._create_scale(nasa_frame, "4. Performance (0=good, 20=poor):", self.performance)
		self._create_scale(nasa_frame, "5. Effort (0=low, 20=high):", self.effort)
		
		# Submit button
		submit_button = ttk.Button(nasa_frame, text="Submit Ratings", command=self.submit_scores)
		submit_button.pack(pady=20)
	
	# def _create_scale(self, parent, label_text, variable):
	# 	frame = ttk.Frame(parent)
	# 	frame.pack(fill=tk.X, pady=8, padx=10)
	# 	frame.columnconfigure(1, weight=1) # Let the scale expand
		
	# 	ttk.Label(frame, text=label_text).grid(row=0, column=0, sticky="w")
	# 	scale = ttk.Scale(frame, from_=0, to=20, orient=tk.HORIZONTAL, variable=variable, length=300)
	# 	scale.grid(row=0, column=1, padx=10)
		
	# 	value_label = ttk.Label(frame, textvariable=variable, width=4)
	# 	# Format value to show integer part
	# 	variable.trace_add("write", lambda *args, var=variable, lbl=value_label: lbl.config(text=f"{int(var.get())}"))
	# 	value_label.grid(row=0, column=2, padx=5, sticky="e")
	
	def _create_scale(self, parent, label_text, variable):
		frame = ttk.Frame(parent)
		frame.pack(fill=tk.X, pady=8, padx=10)

		label = ttk.Label(frame, text=label_text, width=50, anchor="w")  
		label.grid(row=0, column=0, sticky="w")

		scale = ttk.Scale(frame, from_=0, to=20, orient=tk.HORIZONTAL, variable=variable, length=700)
		scale.grid(row=0, column=1, padx=10, sticky="ew")

		value_label = ttk.Label(frame, text="10.0", width=6, anchor="e")
		
		# Update label text when variable changes
		def update_label(*args):
			value_label.config(text=f"{variable.get():.1f}")
		variable.trace_add("write", update_label)
		value_label.grid(row=0, column=2, padx=(10, 15), sticky="e")

	def _create_results_tab(self, parent):
		parent.columnconfigure(0, weight=1)
		parent.columnconfigure(1, weight=1)
		parent.rowconfigure(1, weight=1) # Allow best results to expand

		# Performance metrics display
		metrics_frame = ttk.LabelFrame(parent, text="Performance Metrics")
		metrics_frame.grid(row=0, column=0, padx=10, pady=10, sticky="nsew")
		
		metrics = [
			("gracefulness:", "gracefulness_value"), 
			("Smoothness:", "smoothness_value"),
			("Clutch Times:", "clutch_times_value"),
			("Total Distance:", "total_distance_value"),
			("Total Time:", "total_time_value")
		]
		
		for i, (label_text, attr_name) in enumerate(metrics):
			ttk.Label(metrics_frame, text=label_text).grid(row=i, column=0, sticky=tk.W, padx=5, pady=2)
			setattr(self, attr_name, ttk.Label(metrics_frame, text="-", font=self.base_font + ('bold',)))
			getattr(self, attr_name).grid(row=i, column=1, sticky=tk.W, padx=5, pady=2)
		
		# Scores display
		scores_frame = ttk.LabelFrame(parent, text="Scores")
		scores_frame.grid(row=0, column=1, padx=10, pady=10, sticky="nsew")
		
		scores = [
			("gracefulness Score(5):", "gracefulness_score"), 
			("Smoothness Score(5):", "smoothness_score"),
			("Clutch Times Score(15):", "clutch_times_score"),
			("Total Distance Score(15):", "total_distance_score"),
			("Total Time Score(10):", "total_time_score"),
			("Total Score:", "total_score")
		]
		
		for i, (label_text, attr_name) in enumerate(scores):
			ttk.Label(scores_frame, text=label_text).grid(row=i, column=0, sticky=tk.W, padx=5, pady=2)
			setattr(self, attr_name, ttk.Label(scores_frame, text="-", font=self.base_font + ('bold',)))
			getattr(self, attr_name).grid(row=i, column=1, sticky=tk.W, padx=5, pady=2)
		
		# Best results display
		best_frame = ttk.LabelFrame(parent, text="Best Result")
		best_frame.grid(row=1, column=0, columnspan=2, padx=10, pady=10, sticky="nsew")
		
		self.best_results_text = tk.Text(best_frame, height=10, width=60, font=self.base_font)
		self.best_results_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
		self.best_results_text.config(state=tk.DISABLED, background='#ffffff')
	
	def _initialize_optimizer(self):
		global mode
		if mode ==1:
		# Set parameter bounds
			self.pbounds = config.oneHanded_range
		else:
			self.pbounds = config.twoHanded_range

			
		bounds_transformer = SequentialDomainReductionTransformer(minimum_window=0.1)

		

		if mode != 3:
			# Initialize optimizer
			self.optimizer = BayesianOptimization(
				f=None,
				pbounds=self.pbounds,
				verbose=0,
				random_state=1,
				bounds_transformer=bounds_transformer
			)
			# load the past optimization logs if they exist
			if os.path.exists(self.log_file):
				load_logs(self.optimizer, logs=[self.log_file])
				print("<LYON> The optimizer is now aware of {} points.".format(len(self.optimizer.space)))
			else: 
				print("<LYON> No past optimization logs found, starting fresh.")

			logger = JSONLogger(path=self.log_file, reset=False)
			self.optimizer.subscribe(Events.OPTIMIZATION_STEP, logger)

		
		# Set utility function
		self.utility = UtilityFunction(kind="ei", xi=0.0)
		
		# Define maximum values
		self.gracefulness_max = config.scoreParams_bound['gracefulness_max']
		self.gracefulness_min = config.scoreParams_bound['gracefulness_min']
		self.smoothness_max = config.scoreParams_bound['smoothness_max']
		self.smoothness_min = config.scoreParams_bound['smoothness_min']
		self.clutch_times_max = config.scoreParams_bound['clutch_times_max']
		self.total_distance_max = config.scoreParams_bound['total_distance_max']
		self.total_time_max = config.scoreParams_bound['total_time_max']
	
	def start_optimization(self):
		"""Starts the optimization process."""
		self.max_iterations = self.iter_var.get()
		self.current_iteration = 0
		self.progress_bar['maximum'] = self.max_iterations
		self.progress_bar['value'] = 0
		self.current_iter_label.config(text=f"0/{self.max_iterations}")
		
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

				# Get the next point to probe
				self.next_point = self.optimizer.suggest(self.utility)
				
				# Update parameter display
				self.params_text.config(state=tk.NORMAL)
				self.params_text.delete(1.0, tk.END)
				self.params_text.insert(tk.END, f"Iteration {self.current_iteration}/{self.max_iterations}\n\n")
				for key, value in self.next_point.items():
					self.params_text.insert(tk.END, f"{key:<12} = {value:.6f}\n")
				self.params_text.config(state=tk.DISABLED)
				
				# Save parameters
				self.save_params_to_txt(self.next_point)
			
			# Switch to the NASA-TLX rating tab
			self.tab_control.select(1)  # Select the 2nd tab (NASA-TLX rating)
		else:
			# Optimization finished
			messagebox.showinfo("Optimization Complete", "The Bayesian optimization process has finished!")
			self.next_button.config(state=tk.DISABLED)
			self.stop_button.config(state=tk.DISABLED)
			self.start_button.config(state=tk.NORMAL)
			
			# Show the best result
			if mode!=3:
				self.show_best_result()
	
	def submit_scores(self):
		"""Submits NASA-TLX scores and calculates the target value."""
		# Calculate subjective score
		mental_demand = self.mental_demand.get()
		physical_demand = self.physical_demand.get()
		temporal_demand = self.temporal_demand.get()
		performance = self.performance.get()
		effort = self.effort.get()
		
		raw_tlx = mental_demand + physical_demand + temporal_demand + performance + effort
		sub_score = 100 - raw_tlx
		
		# Switch to the results tab
		self.tab_control.select(2)  # Select the 3rd tab (Results Display)
		
		# Calculate other metrics
		self.calculate_and_display_metrics(sub_score)
	
	def calculate_and_display_metrics(self, sub_score):
		"""Calculates and displays performance metrics and scores."""
		try:
			# Call the original performance calculation function
			gracefulness, smoothness = cal_GS()
			current_file_path = os.path.abspath(__file__)
			current_dir = os.path.dirname(current_file_path)
			data_base_dir = os.path.join(current_dir, 'data')
			latest_dir = get_latest_data_dir(data_base_dir)
			clutch_times = np.load(os.path.join(latest_dir, 'clutch_times.npy'), allow_pickle=True)[0]
			total_distance = np.load(os.path.join(latest_dir, 'total_distance.npy'), allow_pickle=True)[0]
			total_time = np.load(os.path.join(latest_dir, 'total_time.npy'), allow_pickle=True)[0]
			
			# Calculate individual scores
			gracefulness_score =  5 * np.clip((self.gracefulness_max - gracefulness) / (self.gracefulness_max - self.gracefulness_min), 0, 1)
			smoothness_score = 5 * np.clip((self.smoothness_max - smoothness) / (self.smoothness_max - self.smoothness_min), 0, 1)
			clutch_times_score = 15 * np.clip((self.clutch_times_max - clutch_times + 1) / self.clutch_times_max, 0, 1)
			total_distance_score = 15 * np.clip((self.total_distance_max - total_distance) / self.total_distance_max, 0, 1)
			total_time_score = 10 * np.clip((self.total_time_max - total_time) / self.total_time_max, 0, 1)
			
			# Calculate the total score
			total_score = 0.5 * sub_score + gracefulness_score + smoothness_score + clutch_times_score + total_distance_score + total_time_score

			output_json_path = os.path.join(current_dir, 'BayesianLog',config.scorefilename)

			new_entry = {
				"AFflag": float(self.AFflag),
				"subscore": float(sub_score),
				"gracefulness": float(gracefulness),
				"smoothness": float(smoothness),
				"clutch_times": float(clutch_times),
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
			self.clutch_times_value.config(text=f"{clutch_times:.4f}")
			self.total_distance_value.config(text=f"{total_distance:.4f}")
			self.total_time_value.config(text=f"{total_time:.4f}")
			
			self.gracefulness_score.config(text=f"{gracefulness_score:.4f}")
			self.smoothness_score.config(text=f"{smoothness_score:.4f}")
			self.clutch_times_score.config(text=f"{clutch_times_score:.4f}")
			self.total_distance_score.config(text=f"{total_distance_score:.4f}")
			self.total_time_score.config(text=f"{total_time_score:.4f}")
			self.total_score.config(text=f"{total_score:.4f}")
			
			if mode != 3:
				# Register the result with the optimizer
				self.optimizer.register(params=self.next_point, target=total_score)
			
				# If there are more iterations, enable the "Next" button
				if self.current_iteration < self.max_iterations:
					self.next_button.config(state=tk.NORMAL)
				else:
					self.show_best_result()
					self.next_button.config(state=tk.DISABLED)
					self.stop_button.config(state=tk.DISABLED)
					self.start_button.config(state=tk.NORMAL)
				
		except Exception as e:
			messagebox.showerror("Error", f"Error calculating performance metrics: {str(e)}")
	
	def show_best_result(self):
		"""Displays the best result of the optimization."""
		self.best_results_text.config(state=tk.NORMAL)
		self.best_results_text.delete(1.0, tk.END)
		self.best_results_text.insert(tk.END, "Optimization complete! Best result:\n\n")
		
		for key, value in self.optimizer.max['params'].items():
			self.best_results_text.insert(tk.END, f"{key:<12} : {value:.6f}\n")
		
		self.best_results_text.insert(tk.END, f"\nBest Score: {self.optimizer.max['target']:.6f}")
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