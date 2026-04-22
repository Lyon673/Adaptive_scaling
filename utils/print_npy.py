import numpy as np
import os

# 加载 npy 文件
file_path = os.path.join(os.path.dirname(__file__), os.pardir, "data","190_data_04-09","Lmtm_pose.npy")
data = np.load(file_path)

mtm_speed = np.linalg.norm(np.diff(data[:, :3], axis=0), axis=1)
print(f"min speed: {np.min(mtm_speed)}")
print(f"max speed: {np.max(mtm_speed)}")
print(f"average speed: {np.mean(mtm_speed)}")
# 打印数组长度
print("Array length:", len(data))

# 打印前10个元素
print("First 10 elements:", data[:10])