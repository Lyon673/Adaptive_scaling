"""
calculate the ipa using RIPA method
"""


import math
import pywt
import numpy as np
from scipy.signal import savgol_filter
class Pupil:
    def __init__(self, data, timestamp):
        self.data = data
        self.timestamp = timestamp
    
    def get_timestamp(self):
        return self.timestamp

    def get_data(self):
        return self.data

def ipa_cal(d, threshold=None, flag=True, timestamp=None):
    """
    Calculate the Real-time Index of Pupillary Activity (RIPA) based on Savitzky-Golay filtering method.
    
    Parameters:
        d (list): A list of signal data with timestamp information.
        threshold (float, optional): Custom threshold value. If None, calculated automatically.
        flag (bool): Flag for threshold calculation method.
        timestamp (int, optional): Timestamp for windowing.
    
    Returns:
        tuple: (RIPA_value, position, threshold_used)
            - RIPA_value: The calculated RIPA value (normalized, 0-1 range)
            - position: List of positions where significant activity detected
            - threshold_used: The threshold value used in calculation
    """
    if not d or len(d) < 13:  # Minimum required for Savitzky-Golay filtering
        return None, [], 0
    
    # Extract pupil diameter data
    pupil_data = np.array([i.get_data() for i in d])
    
    # Determine data range for processing
    
    working_data = pupil_data
    
    try:
        # RIPA calculation based on Savitzky-Golay filtering
        
        # 1. Calculate smoothed data x̄(t) using Savitzky-Golay filter
        smooth_window = min(13, len(working_data) if len(working_data) % 2 == 1 else len(working_data) - 1)
        if smooth_window < 3:
            smooth_window = 3
        smoothed_data = savgol_filter(working_data, window_length=smooth_window, polyorder=2, deriv=0)
        
        # 2. Calculate low-frequency derivative response g▽ (applied to smoothed data)
        low_deriv_window = min(13, len(smoothed_data) if len(smoothed_data) % 2 == 1 else len(smoothed_data) - 1)
        if low_deriv_window < 3:
            low_deriv_window = 3
        low_freq_deriv = savgol_filter(smoothed_data, window_length=low_deriv_window, polyorder=2, deriv=1)
        
        # 3. Calculate high-frequency derivative response g△ (applied to original data)
        high_deriv_window = min(13, len(working_data) if len(working_data) % 2 == 1 else len(working_data) - 1)
        if high_deriv_window < 3:
            high_deriv_window = 3
        high_freq_deriv = savgol_filter(working_data, window_length=high_deriv_window, polyorder=10, deriv=1)
        
        # 4. Calculate derivative ratio ẋ(t) = g▽/g△
        lf_hf_deriv_ratio = np.zeros_like(low_freq_deriv)
        for i in range(len(low_freq_deriv)):
            if abs(high_freq_deriv[i]) > 1e-10:  # Avoid division by zero
                lf_hf_deriv_ratio[i] = low_freq_deriv[i] / high_freq_deriv[i]
            else:
                lf_hf_deriv_ratio[i] = 0
        
        # 5. Detect modulus maxima m(t) of the derivative ratio
        modulus_maxima = ripa_modmax(lf_hf_deriv_ratio)
        
        # 6. Calculate threshold λ(t)
        non_zero_maxima = [m for m in modulus_maxima if abs(m) > 1e-10]
        
        if threshold is None:
            if len(non_zero_maxima) > 1:
                sigma_hat = np.std(non_zero_maxima)
                omega = (1/0.6745)**2  # RIPA specific parameter
                threshold = omega * sigma_hat
            else:
                threshold = 0.1  # Default threshold if insufficient data
        
        # 7. Calculate RIPA value C(ẋ)
        if len(non_zero_maxima) > 0:
            median_m = np.median(non_zero_maxima)
        else:
            median_m = 0
        
        upper_bound = median_m + threshold
        lower_bound = median_m - threshold
        
        # Count samples outside the threshold bounds
        outlier_count = 0
        
        for i, ratio in enumerate(lf_hf_deriv_ratio):
            if ratio > upper_bound or ratio < lower_bound:
                outlier_count += 1
                
        # 8. Calculate normalized RIPA value Ĉ(ẋ) = 1 - C(ẋ)/N
        total_samples = len(lf_hf_deriv_ratio)
        if total_samples > 0:
            normalized_ripa = 1 - (outlier_count / total_samples)
        else:
            normalized_ripa = 0
        

        return normalized_ripa
        
    except Exception as e:
        print(f"<LYON> RIPA calculation error: {e}")
        return None, [], 0


def ripa_modmax(d):
    """
    Compute the modulus maxima for RIPA calculation using derivative ratio data.
    
    Parameters:
        d (array): A numpy array of derivative ratio values.
    
    Returns:
        list: A list where local maxima are marked with the original value, others are 0.
    """
    if len(d) < 3:
        return [0.0] * len(d)
    
    # Compute signal modulus (absolute value of the signal)
    m = np.abs(d)
    
    # Initialize the result list with 0.0
    t = [0.0] * len(d)
    
    for i in range(len(d)):
        # Get the neighboring values, ensuring correct boundaries
        ll = m[i - 1] if i >= 1 else m[i] + 1e-10  # Left neighbor
        oo = m[i]  # Current value
        rr = m[i + 1] if i < len(d) - 1 else m[i]  # Right neighbor
        
        # Check if the current value is a local maximum
        if (ll <= oo and oo >= rr) and (ll < oo or oo > rr):
            t[i] = d[i]  # Assign the original value (not absolute) at the local maximum
        else:
            t[i] = 0.0  # Otherwise set it to 0
    
    return t
