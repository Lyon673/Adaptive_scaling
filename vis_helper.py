"""
独立子进程入口：在无 tkinter 的干净环境中渲染特征曲线图。
用法: python vis_helper.py <demo_idx>
"""
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'Trajectory'))

from load_data import visualize_scaled_state

if __name__ == '__main__':
    demo_idx = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    visualize_scaled_state(demo_idx)
