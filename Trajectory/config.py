test_demonstration_name = "190.txt"

# TSC parameters

p = 0.9 # regime pruning rate
fp = 0.5 # state-time pruning rate
delta = 0.1 # loop merging threshold
n_regimes = 15 # maximum number of regimes
n_state_clusters = 8 # maximum number of state clusters
n_time_clusters = 6 # maximum number of time clusters

# p = 0.8 # regime pruning rate
# fp = 0.4 # state-time pruning rate
# delta = 0.1 # loop merging threshold
# n_regimes = 15 # maximum number of regimes
# n_state_clusters = 10 # maximum number of state clusters
# n_time_clusters = 6 # maximum number of time clusters

# TSC model path
TSC_model_path = "TSC_model.pkl"

# main parameters
state_probability_threshold = 0.4
time_probability_threshold = 0.4