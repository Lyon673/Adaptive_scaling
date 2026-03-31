# ── System settings ──────────────────────────────────────────────────
resolution_x = 960
resolution_y = 540

#open_phase_segmentation = True



# —— Adaptive formula settings ──────────────────────────────────────────────────

fixed_scale = 10


# the parameters in the scaling calculation formula
feature_bound = {
    'd_min': 0.015, 'd_max': 0.065,
	'p_min': 0.95, 'p_max': 1,
	'v_min': 0.001, 'v_max': 0.10,
    's_min': 0.01, 's_max': 0.06,
}

# the initial params for the optimization
init_params = {
    'K_g': 10.0,
    'K_p': 2.0,
    'C_base': 9.0,

    'A_theta': 5,
    'A_gp': 2.0,
    'A_v': 4.3,
    'B_safety': 3.8,

	'fixed_scale': 1.2,  
	'AFflag': 0 # 0-adaptive, 1-fixed
}

# feature filter params
velocity_queue_length = 14 # direction
ipa_window_length = 63
ipa_polyorder = 3
velocity_window_length = 9
velocity_polyorder = 5



# —— Comparison Experiments settings ──────────────────────────────────────────────────

comparison_experiment_mode = 0 # 0-phase adaptive 1-adaptive  2-lower fixed, 3-lower and upper fixed

lower_fixed_scale = 10

upper_fixed_scale = 18

# the params of the best adaptive frame
# adaptive = {"K_g":10.0, "K_p":2.0, "C_base":9.0, "A_theta":2, "A_gp":2.0, "A_v":4.3, "B_safety":1.0, "AFflag":0}


# fixed = {"fixed_scale": 1.2, "AFflag": 1}






# —— Bayesian Optimization settings ──────────────────────────────────────────────────

iter_times = 10

phase_rate = 0.6

optimization_range = {
    'K_g': (8.0, 12.0),
    'C_base': (6.0, 15.0),

    'A_theta': (3.0, 7.0),
    'A_gp': (1.0, 4.0),
    'A_v': (3.0, 5.5), 
    'B_safety': (3.6, 4.0),
}

# the optimization params set history
logname ='logs.log.json'

# the score log of the optimazation tests
scorefilename = 'scores.json'

# parameters in the score calculation formula
scoreParams_bound = {
    'gracefulness_max': 5,
    'gracefulness_min': 2,
	'smoothness_max': 6,
    'smoothness_min': 2,
    'clutch_times_max': 4,
    'total_distance_max': 0.4,
    'total_time_max': 50
}




# —— Screen recording settings ──────────────────────────────────────────────────

enable_screen_recording = False # Set to False to disable screen recording

screen_recording_params = {
    'x': 2560,                                   # Left-top corner X coordinate
    'y': 0,                                      # Left-top corner Y coordinate
    'width': 2560,                               # Capture area width
    'height': 1440,                              # Capture area height
    'fps': 15,                                   # Frames per second
    'output_scale': 0.375,                         # Output scale (0.5 → 1280x720)
    'output_dir': '/home/lambda/Videos/train',   # Video output directory
}











# —— Gaze filter settings ──────────────────────────────────────────────────
gaze_filter_params = {
    'heatmap_size_x': 192,
    'heatmap_size_y': 108,
    'scale_params': {
        'd_min': 0,      # Minimum distance
        'd_max': 0.12,    # Maximum distance
    },

    # Outlier filtering parameters - more focused on significant outliers
    'filter_params': {
        'attention_threshold': 0.2, # heatmap value
        'window_seconds': 2.0, # time window
        'jump_threshold': 0.5, # 2d distance
        'min_neighbors': 3, # number of neighbors
        'velocity_threshold': 1, # 2d velocity
    },

    # Fixed window configuration 
    'fixed_window_config': {
        'enabled': True,           # whether to enable the fixed window
        'activation_offset': 2.5   # how many seconds before the end to activate the fixed window
    },

    # the previous scale factor
    'prev_scale_factor': [1.0, 1.0],

    'gaussian_kernel_sigma': 16,

    'temporal_exponent_decay': 0.8,

}



