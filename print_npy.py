import numpy as np
import os

def print_npy(npy_path):
    np.set_printoptions(
    precision=15,     # 设置小数点后的位数（最多15位，因为float64的精度限制）
    suppress=False,   # False显示科学计数法，True则正常显示
    linewidth=200     # 每行的字符宽度
    )
    npy = np.load(npy_path)
    print(npy[:10])

if __name__ == '__main__':
    npy_path = os.path.join(os.path.dirname(__file__), 'data', '0_data_12-01', 'Lpsm_pose.npy')
    print_npy(npy_path)