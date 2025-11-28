import os
import numpy as np
import matplotlib.pyplot as plt
from collections import deque
from scipy.signal import savgol_filter


def load_ipa_series(data_dir):
    """Load left/right IPA arrays just like visualization.py."""
    # ipa_left = np.load(os.path.join(data_dir, 'ipaL_data.npy'), allow_pickle=True)
    # ipa_right = np.load(os.path.join(data_dir, 'ipaR_data.npy'), allow_pickle=True)
    ipa_left = np.load(os.path.join(data_dir, 'Lpsm_velocity.npy'), allow_pickle=True)
    ipa_right = np.load(os.path.join(data_dir, 'Rpsm_velocity.npy'), allow_pickle=True)
    return np.asarray(ipa_left).ravel(), np.asarray(ipa_right).ravel()


def kalman_filter_1d(measurements, process_var=1e-3, measurement_var=1e-2, initial_estimate=None):
    """
    Basic 1D Kalman filter for scalar IPA signals.

    Args:
        measurements: 1D numpy array of raw IPA readings.
        process_var: Q, expected process noise variance.
        measurement_var: R, measurement noise variance.
        initial_estimate: optional starting value; defaults to first measurement.

    Returns:
        np.ndarray of filtered estimates with same length as measurements.
    """
    if measurements.ndim != 1:
        raise ValueError('Measurements must be a 1D array')
    if len(measurements) == 0:
        return measurements.copy()

    x = initial_estimate if initial_estimate is not None else measurements[0]
    p = 1.0  # initial covariance
    q = process_var
    r = measurement_var
    filtered = np.empty_like(measurements, dtype=float)

    for i, z in enumerate(measurements):
        # Predict
        p += q
        # Update
        k = p / (p + r)
        x = x + k * (z - x)
        p = (1 - k) * p
        filtered[i] = x

    return filtered



def savgol_filter_ipa(measurements, window_length=15, polyorder=5):
    """
    Apply Savitzky-Golay smoothing to IPA data.

    Args:
        measurements: 1D numpy array.
        window_length: odd integer window size.
        polyorder: polynomial order (< window_length).
    """
    if window_length % 2 == 0:
        raise ValueError('window_length must be odd')
    if polyorder >= window_length:
        raise ValueError('polyorder must be less than window_length')
    return savgol_filter(measurements, window_length=window_length, polyorder=polyorder, mode='interp')




class RealTimeSavitzkyGolay:
    """
    Sliding-window Savitzky-Golay filter for streaming IPA samples.

    Keeps the latest `window_length` points in a deque and recomputes the
    filtered value each time a new sample arrives.
    """

    def __init__(self, window_length=15, polyorder=3):
        if window_length % 2 == 0:
            raise ValueError('window_length must be odd')
        if polyorder >= window_length:
            raise ValueError('polyorder must be less than window_length')
        self.window_length = window_length
        self.polyorder = polyorder
        self.buffer = deque(maxlen=window_length)

    def update(self, sample):
        """
        Add a new sample and return the latest smoothed value.
        Returns the raw sample until the buffer is full.
        """
        self.buffer.append(sample)
        if len(self.buffer) < self.window_length:
            return sample
        filtered_window = savgol_filter(
            np.array(self.buffer),
            window_length=self.window_length,
            polyorder=self.polyorder,
            mode='interp',
        )
        return filtered_window[-1]


def get_latest_data_dir(base_dir,num=None):
    """Return newest subdirectory under the data folder."""
    if num is not None:
        subdirs = [
            d for d in os.listdir(base_dir)
            if os.path.isdir(os.path.join(base_dir, d)) and d.split('_')[0] == str(num)
        ]
        if not subdirs:
            raise FileNotFoundError(f'No subdirectories in {base_dir} with number {num}')
        subdirs.sort(key=lambda name: int(name.split('_')[0]) if name.split('_')[0].isdigit() else 0)
        return os.path.join(base_dir, subdirs[-1])
    subdirs = [
        d for d in os.listdir(base_dir)
        if os.path.isdir(os.path.join(base_dir, d))
    ]
    if not subdirs:
        raise FileNotFoundError(f'No subdirectories in {base_dir}')
    subdirs.sort(key=lambda name: int(name.split('_')[0]) if name.split('_')[0].isdigit() else 0)
    return os.path.join(base_dir, subdirs[-1])


def demo_filters(data_dir):
    """
    Load IPA data, apply filters, and return a dict of arrays for downstream use.
    """
    left, right = load_ipa_series(data_dir)
    left_kf = kalman_filter_1d(left)
    right_kf = kalman_filter_1d(right)

    left_sg = savgol_filter_ipa(left)
    right_sg = savgol_filter_ipa(right)

    ipa_avg = (left + right) / 2.0
    avg_kf = kalman_filter_1d(ipa_avg)
    avg_sg = savgol_filter_ipa(ipa_avg)

    return {
        'left_raw': left,
        'right_raw': right,
        'avg_raw': ipa_avg,
        'left_kalman': left_kf,
        'right_kalman': right_kf,
        'avg_kalman': avg_kf,
        'left_savgol': left_sg,
        'right_savgol': right_sg,
        'avg_savgol': avg_sg,
    }


def visualize_filtered_ipa(data_dir, filtered_data=None):
    """Create comparison plots for raw vs filtered IPA signals."""
    data = filtered_data or demo_filters(data_dir)
    timestamps_left = np.arange(len(data['left_raw']))
    timestamps_right = np.arange(len(data['right_raw']))
    timestamps_avg = np.arange(len(data['avg_raw']))

    color_raw = '#95a5a6'
    color_kalman = '#3498db'
    color_savgol = '#e67e22'
    color_avg_kf = '#9b59b6'
    color_avg_sg = '#f1c40f'

    fig, axs = plt.subplots(3, 1, figsize=(16, 14), sharex=False)
    fig.patch.set_facecolor('white')

    axs[0].plot(timestamps_left, data['left_raw'], color=color_raw, alpha=0.5, linewidth=1.2, label='Left Raw')
    axs[0].plot(timestamps_left, data['left_kalman'], color=color_kalman, alpha=0.9, linewidth=2.2, label='Left Kalman')
    axs[0].plot(timestamps_left, data['left_savgol'], color=color_savgol, alpha=0.9, linewidth=2.2, label='Left Savitzky-Golay')
    axs[0].set_title('Left Hand IPA Filtering Comparison', fontsize=13, fontweight='bold', pad=15)
    axs[0].set_ylabel('IPA Value', fontsize=11)
    axs[0].grid(True, alpha=0.2, linestyle='--')
    axs[0].set_facecolor('#f8f9fa')
    axs[0].legend(loc='best', frameon=True, fancybox=True, shadow=True)

    axs[1].plot(timestamps_right, data['right_raw'], color=color_raw, alpha=0.5, linewidth=1.2, label='Right Raw')
    axs[1].plot(timestamps_right, data['right_kalman'], color=color_kalman, alpha=0.9, linewidth=2.2, label='Right Kalman')
    axs[1].plot(timestamps_right, data['right_savgol'], color=color_savgol, alpha=0.9, linewidth=2.2, label='Right Savitzky-Golay')
    axs[1].set_title('Right Hand IPA Filtering Comparison', fontsize=13, fontweight='bold', pad=15)
    axs[1].set_ylabel('IPA Value', fontsize=11)
    axs[1].grid(True, alpha=0.2, linestyle='--')
    axs[1].set_facecolor('#f8f9fa')
    axs[1].legend(loc='best', frameon=True, fancybox=True, shadow=True)

    axs[2].plot(timestamps_avg, data['avg_raw'], color=color_raw, alpha=0.5, linewidth=1.2, label='Average Raw')
    axs[2].plot(timestamps_avg, data['avg_kalman'], color=color_avg_kf, alpha=0.9, linewidth=2.2, label='Average Kalman')
    axs[2].plot(timestamps_avg, data['avg_savgol'], color=color_avg_sg, alpha=0.9, linewidth=2.2, label='Average Savitzky-Golay')
    axs[2].set_title('Average IPA Filtering Comparison', fontsize=13, fontweight='bold', pad=15)
    axs[2].set_xlabel('Frame Index', fontsize=11)
    axs[2].set_ylabel('IPA Value', fontsize=11)
    axs[2].grid(True, alpha=0.2, linestyle='--')
    axs[2].set_facecolor('#f8f9fa')
    axs[2].legend(loc='best', frameon=True, fancybox=True, shadow=True)

    plt.tight_layout()
    plt.show()
    return None


if __name__ == '__main__':
    project_dir = os.path.dirname(os.path.abspath(__file__))
    data_base = os.path.join(project_dir, 'data')
    data_dir = get_latest_data_dir(data_base, num=53)
    print(f'Using data directory: {data_dir}')
    results = demo_filters(data_dir)
    for key, arr in results.items():
        print(f'{key}: mean={arr.mean():.4f}, std={arr.std():.4f}')
    visualize_filtered_ipa(data_dir, filtered_data=results)
