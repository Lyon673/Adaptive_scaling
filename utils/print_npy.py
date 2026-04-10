import numpy as np
import os

# 加载 npy 文件
file_path = os.path.join(os.path.dirname(__file__), os.pardir, "data","174_data_04-09","gaze_data.npy")
data = np.load(file_path)

# 打印数组长度
print("Array length:", len(data))

# 打印前10个元素
print("First 10 elements:", data[:10])