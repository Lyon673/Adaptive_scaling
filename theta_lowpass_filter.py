#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Kalman and Savitzky-Golay filters for theta (angle between velocity and gaze direction) visualization.
Applies two different filtering methods to remove noise from theta signals.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
import os
import sys


class KalmanFilter1D:
    """
    1D Kalman Filter implementation for signal smoothing.
    
    参数调整方向:
    - process_variance (Q): 过程噪声方差
      * 增大: 滤波器更快响应信号变化，跟踪性更好，但噪声抑制较弱
      * 减小: 更平滑的输出，噪声抑制更强，但可能跟不上快速变化
      * 典型范围: 1e-5 到 1e-1
      
    - measurement_variance (R): 测量噪声方差
      * 增大: 输出更平滑，但响应变慢
      * 减小: 更信任测量值，响应更快，但噪声抑制较弱
      * 典型范围: 1e-2 到 1.0
      
    - initial_estimate_error (P): 初始估计误差
      * 影响滤波器收敛速度
      * 典型值: 1.0
    """
    
    def __init__(self, process_variance=1e-3, measurement_variance=0.1, 
                 initial_value=0.0, initial_estimate_error=1.0):
        """
        Initialize Kalman Filter.
        
        Args:
            process_variance: Process noise covariance (Q)
            measurement_variance: Measurement noise covariance (R)
            initial_value: Initial state estimate
            initial_estimate_error: Initial estimate error covariance (P)
        """
        self.process_variance = process_variance  # Q
        self.measurement_variance = measurement_variance  # R
        self.estimate = initial_value  # x̂
        self.estimate_error = initial_estimate_error  # P
    
    def update(self, measurement):
        """
        Update Kalman filter with new measurement.
        
        Args:
            measurement: New measurement value
            
        Returns:
            Updated state estimate
        """
        # Prediction step
        prediction = self.estimate
        prediction_error = self.estimate_error + self.process_variance
        
        # Update step
        kalman_gain = prediction_error / (prediction_error + self.measurement_variance)
        self.estimate = prediction + kalman_gain * (measurement - prediction)
        self.estimate_error = (1 - kalman_gain) * prediction_error
        
        return self.estimate
    
    def filter(self, data):
        """
        Apply Kalman filter to entire data array.
        
        Args:
            data: Input signal (1D array)
            
        Returns:
            Filtered signal
        """
        filtered = np.zeros_like(data)
        self.estimate = data[0]  # Initialize with first measurement
        
        for i, measurement in enumerate(data):
            filtered[i] = self.update(measurement)
        
        return filtered


def savgol_smooth(data, window_length=11, polyorder=3):
    """
    Apply Savitzky-Golay filter to data.
    
    参数调整方向:
    - window_length: 窗口长度（必须是奇数）
      * 增大: 更平滑，但可能过度平滑细节
      * 减小: 保留更多细节，但噪声抑制较弱
      * 典型范围: 5 到 51（奇数）
      * 建议: 从数据点数的 5-10% 开始尝试
      
    - polyorder: 多项式阶数
      * 增大: 更好地拟合复杂曲线，但可能引入伪影
      * 减小: 更平滑，但可能丢失信号特征
      * 典型范围: 2 到 5
      * 约束: polyorder < window_length
      * 建议: 对于平滑信号用2-3，复杂信号用3-4
    
    Args:
        data: Input signal (1D array)
        window_length: Length of filter window (must be odd)
        polyorder: Order of polynomial to fit
    
    Returns:
        Filtered signal
    """
    if window_length % 2 == 0:
        window_length += 1  # Ensure odd number
    
    if window_length > len(data):
        window_length = len(data) if len(data) % 2 == 1 else len(data) - 1
    
    if polyorder >= window_length:
        polyorder = window_length - 1
    
    return savgol_filter(data, window_length, polyorder)


def visualize_theta_with_filters(data_dir, 
                                  kalman_q=1e-3, kalman_r=0.1,
                                  savgol_window=11, savgol_poly=3):
    """
    Load theta data, apply Kalman and Savitzky-Golay filters, and visualize results.
    
    Args:
        data_dir: Directory containing theta.npy
        kalman_q: Kalman filter process variance (Q)
        kalman_r: Kalman filter measurement variance (R)
        savgol_window: Savitzky-Golay window length (must be odd)
        savgol_poly: Savitzky-Golay polynomial order
    """
    # Load theta data
    theta_path = os.path.join(data_dir, 'theta.npy')
    
    if not os.path.exists(theta_path):
        print(f"Error: {theta_path} not found!")
        return
    
    theta_array = np.load(theta_path)
    print(f"Loaded theta data shape: {theta_array.shape}")
    
    # Extract left and right theta values
    if theta_array.ndim > 1 and theta_array.shape[1] >= 2:
        thetaL_rad = theta_array[:, 0]
        thetaR_rad = theta_array[:, 1]
    else:
        thetaL_rad = theta_array
        thetaR_rad = np.zeros_like(thetaL_rad)
    
    # Convert to degrees
    thetaL = np.degrees(thetaL_rad)
    thetaR = np.degrees(thetaR_rad)
    
    # Apply Kalman filter
    kf_L = KalmanFilter1D(process_variance=kalman_q, measurement_variance=kalman_r)
    kf_R = KalmanFilter1D(process_variance=kalman_q, measurement_variance=kalman_r)
    thetaL_kalman = kf_L.filter(thetaL)
    thetaR_kalman = kf_R.filter(thetaR)
    
    # Apply Savitzky-Golay filter
    thetaL_savgol = savgol_smooth(thetaL, window_length=savgol_window, polyorder=savgol_poly)
    thetaR_savgol = savgol_smooth(thetaR, window_length=savgol_window, polyorder=savgol_poly)
    
    # Create timesteps
    timesteps = np.arange(len(thetaL))
    
    # Create figure with 3 rows and 2 columns
    fig = plt.figure(figsize=(16, 14))
    
    # Top row: Original signals
    ax1 = plt.subplot(3, 2, 1)
    ax1.plot(timesteps, thetaL, color='tab:blue', linewidth=1.5, alpha=0.7, label='Original')
    ax1.set_xlabel('Time step', fontsize=12)
    ax1.set_ylabel('Theta (degrees)', fontsize=12)
    ax1.set_title('Left PSM: Original Theta', fontsize=13, fontweight='bold')
    ax1.grid(True, linestyle='--', alpha=0.4)
    ax1.legend(fontsize=11)
    
    ax2 = plt.subplot(3, 2, 2)
    ax2.plot(timesteps, thetaR, color='tab:orange', linewidth=1.5, alpha=0.7, label='Original')
    ax2.set_xlabel('Time step', fontsize=12)
    ax2.set_ylabel('Theta (degrees)', fontsize=12)
    ax2.set_title('Right PSM: Original Theta', fontsize=13, fontweight='bold')
    ax2.grid(True, linestyle='--', alpha=0.4)
    ax2.legend(fontsize=11)
    
    # Middle row: Kalman filtered
    ax3 = plt.subplot(3, 2, 3)
    ax3.plot(timesteps, thetaL, color='tab:blue', linewidth=1.0, alpha=0.3, label='Original')
    ax3.plot(timesteps, thetaL_kalman, color='darkblue', linewidth=2.0, label=f'Kalman (Q={kalman_q:.0e}, R={kalman_r:.0e})')
    ax3.set_xlabel('Time step', fontsize=12)
    ax3.set_ylabel('Theta (degrees)', fontsize=12)
    ax3.set_title('Left PSM: Kalman Filtered', fontsize=13, fontweight='bold')
    ax3.grid(True, linestyle='--', alpha=0.4)
    ax3.legend(fontsize=10)
    
    ax4 = plt.subplot(3, 2, 4)
    ax4.plot(timesteps, thetaR, color='tab:orange', linewidth=1.0, alpha=0.3, label='Original')
    ax4.plot(timesteps, thetaR_kalman, color='darkorange', linewidth=2.0, label=f'Kalman (Q={kalman_q:.0e}, R={kalman_r:.0e})')
    ax4.set_xlabel('Time step', fontsize=12)
    ax4.set_ylabel('Theta (degrees)', fontsize=12)
    ax4.set_title('Right PSM: Kalman Filtered', fontsize=13, fontweight='bold')
    ax4.grid(True, linestyle='--', alpha=0.4)
    ax4.legend(fontsize=10)
    
    # Bottom row: Savitzky-Golay filtered
    ax5 = plt.subplot(3, 2, 5)
    ax5.plot(timesteps, thetaL, color='tab:blue', linewidth=1.0, alpha=0.3, label='Original')
    ax5.plot(timesteps, thetaL_savgol, color='darkgreen', linewidth=2.0, label=f'Savitzky-Golay (win={savgol_window}, poly={savgol_poly})')
    ax5.set_xlabel('Time step', fontsize=12)
    ax5.set_ylabel('Theta (degrees)', fontsize=12)
    ax5.set_title('Left PSM: Savitzky-Golay Filtered', fontsize=13, fontweight='bold')
    ax5.grid(True, linestyle='--', alpha=0.4)
    ax5.legend(fontsize=10)
    
    ax6 = plt.subplot(3, 2, 6)
    ax6.plot(timesteps, thetaR, color='tab:orange', linewidth=1.0, alpha=0.3, label='Original')
    ax6.plot(timesteps, thetaR_savgol, color='darkgreen', linewidth=2.0, label=f'Savitzky-Golay (win={savgol_window}, poly={savgol_poly})')
    ax6.set_xlabel('Time step', fontsize=12)
    ax6.set_ylabel('Theta (degrees)', fontsize=12)
    ax6.set_title('Right PSM: Savitzky-Golay Filtered', fontsize=13, fontweight='bold')
    ax6.grid(True, linestyle='--', alpha=0.4)
    ax6.legend(fontsize=10)
    
    plt.tight_layout()
    
    # Print filter parameters and statistics
    print(f"\n{'='*60}")
    print(f"Filter Parameters:")
    print(f"{'='*60}")
    print(f"\nKalman Filter:")
    print(f"  Process variance (Q): {kalman_q:.0e}")
    print(f"  Measurement variance (R): {kalman_r:.0e}")
    print(f"\nSavitzky-Golay Filter:")
    print(f"  Window length: {savgol_window}")
    print(f"  Polynomial order: {savgol_poly}")
    
    print(f"\n{'='*60}")
    print(f"Data Statistics:")
    print(f"{'='*60}")
    print(f"\nOriginal data range (degrees):")
    print(f"  Left PSM:  [{thetaL.min():.2f}, {thetaL.max():.2f}]")
    print(f"  Right PSM: [{thetaR.min():.2f}, {thetaR.max():.2f}]")
    
    print(f"\nKalman filtered range (degrees):")
    print(f"  Left PSM:  [{thetaL_kalman.min():.2f}, {thetaL_kalman.max():.2f}]")
    print(f"  Right PSM: [{thetaR_kalman.min():.2f}, {thetaR_kalman.max():.2f}]")
    
    print(f"\nSavitzky-Golay filtered range (degrees):")
    print(f"  Left PSM:  [{thetaL_savgol.min():.2f}, {thetaL_savgol.max():.2f}]")
    print(f"  Right PSM: [{thetaR_savgol.min():.2f}, {thetaR_savgol.max():.2f}]")
    
    # Calculate smoothness (variance)
    print(f"\nData variance (lower = smoother):")
    print(f"  Original:        Left={thetaL.var():.2f}, Right={thetaR.var():.2f}")
    print(f"  Kalman:          Left={thetaL_kalman.var():.2f}, Right={thetaR_kalman.var():.2f}")
    print(f"  Savitzky-Golay:  Left={thetaL_savgol.var():.2f}, Right={thetaR_savgol.var():.2f}")
    print(f"{'='*60}\n")
    
    # Save the figure
    # output_path = os.path.join(data_dir, 'theta_filter_comparison.png')
    # plt.savefig(output_path, dpi=150, bbox_inches='tight')
    # print(f"Figure saved to: {output_path}")
    
    plt.show()


if __name__ == "__main__":
    # Default to the latest data directory
    if len(sys.argv) > 1:
        data_dir = sys.argv[1]
    else:
        data_dir = "/home/lambda/surgical_robotics_challenge/scripts/surgical_robotics_challenge/Project/data/78_data_11-27"
    
    print("\n" + "="*60)
    print("Theta Filter Comparison: Kalman vs Savitzky-Golay")
    print("="*60 + "\n")
    
    # ========== 参数调整指南 ==========
    # 
    # Kalman Filter 参数:
    # ------------------
    # kalman_q (过程噪声方差): 控制滤波器对信号变化的响应速度
    #   - 增大 (如 1e-2): 快速跟踪信号变化，但噪声抑制较弱
    #   - 减小 (如 1e-4): 更平滑的输出，但响应较慢
    #   - 推荐范围: 1e-5 到 1e-1
    #   - 默认: 1e-3 (平衡跟踪性和平滑性)
    #
    # kalman_r (测量噪声方差): 控制对测量值的信任程度
    #   - 增大 (如 1.0): 输出更平滑，但响应变慢
    #   - 减小 (如 0.01): 更信任测量值，响应更快
    #   - 推荐范围: 1e-2 到 1.0
    #   - 默认: 0.1
    #
    # 调参策略:
    #   1. 如果信号过于平滑、跟不上变化: 增大 kalman_q 或减小 kalman_r
    #   2. 如果噪声太多、不够平滑: 减小 kalman_q 或增大 kalman_r
    #   3. Q/R 比值决定平滑度: 比值越小越平滑
    #
    # Savitzky-Golay Filter 参数:
    # ---------------------------
    # savgol_window (窗口长度): 控制平滑程度
    #   - 增大 (如 21, 31): 更平滑，但可能过度平滑细节
    #   - 减小 (如 5, 7): 保留更多细节，但噪声抑制较弱
    #   - 必须是奇数，推荐范围: 5 到 51
    #   - 建议: 从数据点数的 5-10% 开始
    #   - 默认: 11
    #
    # savgol_poly (多项式阶数): 控制拟合曲线的复杂度
    #   - 增大 (如 4, 5): 更好地拟合复杂曲线
    #   - 减小 (如 2): 更平滑，适合简单信号
    #   - 约束: polyorder < window_length
    #   - 推荐范围: 2 到 5
    #   - 默认: 3 (适合大多数情况)
    #
    # 调参策略:
    #   1. 先调整 window_length: 窗口越大越平滑
    #   2. 再调整 polyorder: 阶数越高越能保留信号特征
    #   3. 如果信号变化快: 用小窗口 + 高阶数 (如 win=7, poly=4)
    #   4. 如果需要强平滑: 用大窗口 + 低阶数 (如 win=21, poly=2)
    #
    # 两种滤波器的选择:
    # ----------------
    # - Kalman: 适合实时处理、在线滤波，能自适应调整
    # - Savitzky-Golay: 适合离线处理，在保留信号形状方面表现更好
    # - 如果数据有突变: Kalman 可能表现更好
    # - 如果需要保留峰值: Savitzky-Golay 可能表现更好
    
    visualize_theta_with_filters(
        data_dir=data_dir,
        kalman_q=1e-2,         # 过程噪声方差 (try: 1e-4, 1e-3, 1e-2)
        kalman_r=0.1,          # 测量噪声方差 (try: 0.01, 0.1, 1.0)
        savgol_window=11,      # 窗口长度 (try: 7, 11, 15, 21)
        savgol_poly=3          # 多项式阶数 (try: 2, 3, 4)
    )

