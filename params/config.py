# resolution of windows
resolution_x = 960
resolution_y = 540

# the parameters in the scaling calculation formula
feature_bound = {
    'd_min': 0.00, 'd_max': 0.08,
	'p_min': 0.95, 'p_max': 1,
	'v_min': 0, 'v_max': 0.08,
    's_min': 0.01, 's_max': 0.1,
    'C_offset': 0.05
}

# the parameters in Bayesian optimization
# optimization_range = { 
#     # threshold
#     'tau_d': (0.7, 1),
#     'tau_p': (0.7, 1),
#     'tau_v': (0.7, 1),
#     'tau_s': (0.7, 1),  

#     # gains
#     # 'A_d': (1, 3),
#     # 'A_p': (1, 10),
#     # 'A_v': (1, 10),
#     # 'A_s': (1, 10),

#     # weights
#     'Y_base': (0.01, 0.15),  # base
#     'W_d': (0.1, 5),	 
#     'W_p': (0.1, 5),	   
#     'W_v': (0.1, 3),	
#     'W_s': (0.1, 5),
#     'W_dps': (0.1, 1.5),	  
#     'W_dvs': (0.1, 1.5),	 
#     'W_pvs': (0.1, 1.5),	 
#     'W_dpv': (0.1, 1.5),
#     'W_dpvs': (0.1, 1.5),   
# }

optimization_range = {
}

# parameters in the score calculation formula
scoreParams_bound = {
    'gracefulness_max': 5,
    'gracefulness_min': 1.5,
	'smoothness_max': 8,
    'smoothness_min': 4,
    'clutch_times_max': 10,
    'total_distance_max': 0.6,
    'total_time_max': 120
}

# the optimization params set history
logname ='logs.log.json'

# the score log of the optimazation tests
scorefilename = 'scores.json'

# for the command params (not used)
exflag = 1

# the params of the best adaptive frame
adaptive = {"W_d": 0.6557110261229989, "W_dp": 1.5489982389822312, "W_dv": 1.661834391297523, "W_p": 1.0, "W_pv": 1.340468616420758, "W_v": 0.4538660096723739, "Y_base": 0.03331922812540701, "tau_d": 0.7, "tau_p": 0.7, "tau_v": 1.0}

fixed = {"fixed_scale": 0.35, "AFflag": 1}

# the initial params for the optimization
init_params = {

	'd_min': feature_bound['d_min'] , 'd_max': feature_bound['d_max'],
	'p_min': feature_bound['p_min'] , 'p_max': feature_bound['p_max'], 
	'v_min': feature_bound['v_min'] , 'v_max': feature_bound['v_max'],
	's_min': feature_bound['s_min'] , 's_max': feature_bound['s_max'],
    #"W_d": 0.6557110261229989, "W_dp": 1.5489982389822312, "W_dv": 1.661834391297523, "W_p": 1.0, "W_pv": 1.340468616420758, "W_v": 0.4538660096723739, "Y_base": 0.03331922812540701, "tau_d": 0.7, "tau_p": 0.7, "tau_v": 1.0,
	'tau_d': 0.7,  
	'tau_p': 0.8,  
	'tau_v': 0.6,  
	'tau_s': 0.5,  

	'Y_base': 0.1,  
	'W_d': 1.0,	 
	'W_p': 1.0,	  
	'W_v': 1.0,	 
	'W_s': 1.0,
	'W_dp': 0.5,	
	'W_dv': 0.5,	 
	'W_pv': 0.5,	
	'W_dpv': 0.5,	
	'W_dps': 0.5,	
	'W_dvs': 0.5,	
	'W_pvs': 0.5,	
	'W_dpvs': 0.5, 

	'C_offset': feature_bound['C_offset'],

	'fixed_scale': 1.0,  
	'AFflag': 0 # 0-adaptive, 1-fixed
}

# the params for the gaze filter
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

