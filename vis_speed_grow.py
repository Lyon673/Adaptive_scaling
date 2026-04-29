import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fsolve
import os


# 模拟 config.feature_bound
class Config:
    feature_bound = {'v_min': 0, 'v_max': 0.1} # 假设值，请根据实际情况修改

config = Config()

# 参数设置
k = 2 * 1e-2
K_g = 10
C_base = 14
alpha = 2
beta = 2

def get_fixed_point_function(v, v_m):
    """构建 G(v) - v = 0 的形式用于求解"""
    # 1. Normalize
    v_min = config.feature_bound['v_min']
    v_max = config.feature_bound['v_max']
    N_v = np.clip((v - v_min) / (v_max - v_min), 0.001, 1)
    
    # 2. expFunc & s_t
    f_val = 1 - np.exp(-(alpha**2) * (N_v**beta))
    s_t = K_g * (0.3 + f_val) + C_base
    
    # 3. G(v)
    return k * v_m * s_t

def solve_fixed_point(v_m):
    """针对给定的 v_m 求解 G(v) = v"""
    func = lambda v: get_fixed_point_function(v, v_m) - v
    # 初始猜测值设为 v_m 影响下的基准值
    v_start = k * v_m * (K_g + C_base)
    root = fsolve(func, x0=v_start)
    return root[0]

if __name__ == "__main__":
    # 定义 v_m 的范围
    v_m_range = np.linspace(1, 20, 80)*0.01 # 从 1 到 30 扫描 v_m
    fixed_points = [solve_fixed_point(vm) for vm in v_m_range]

    # 绘图
    plt.figure(figsize=(10, 6))
    plt.plot(v_m_range, fixed_points, color='black', linewidth=2)
    
    plt.title('Relationship between $v_m$ and Fixed Point $v^*$', fontsize=22)
    plt.xlabel('MTM Velocity $v_m$', fontsize=18)
    plt.ylabel('Steady State PSM Velocity $v^*$', fontsize=18)
    plt.grid(True, linestyle='--', alpha=0.7)

    OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Essay_image_results")
    save_path = os.path.join(OUTPUT_DIR, f"velocity_stability.pdf")
    plt.savefig(save_path, format="pdf", dpi=300, bbox_inches='tight', facecolor='white')