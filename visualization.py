import numpy as np
import matplotlib.pyplot as plt
from collections import deque
import socket
import Project.ipa as ipa
import pywt

_ipa = np.load('./Project/data/ipaL_data.npy', allow_pickle=True)[90:]

for i in range(len(_ipa)):
    if _ipa[i] < 0.5:
        _ipa[i] = _ipa[i-1]
        
_v = np.load('./Project/data/psm_velocity_data.npy', allow_pickle=True)[90:]
_d = np.load('./Project/data/distance_data.npy', allow_pickle=True)[90:]
_s = np.load('./Project/data/scale_data.npy', allow_pickle=True)[90:]

for i in range(len(_ipa)):
    if _ipa[i] != 1 and _ipa[i-1] == 1:
        print(f"index:{i}")

timestamps = range(len(_ipa))
print(len(timestamps))



fig, axs = plt.subplots(2, 2, figsize=(12, 8))  

# 绘制 _ipa 数据
axs[0, 0].scatter(timestamps, _ipa, c='red', alpha=0.5, s=2)
axs[0, 0].set_title('IPA Data')
axs[0, 0].set_xlabel('Timestamps')
axs[0, 0].set_ylabel('IPA Values')

# 绘制 _v 数据
axs[0, 1].scatter(timestamps, _v, c='blue', alpha=0.5, s=2)
axs[0, 1].set_title('Velocity Data')
axs[0, 1].set_xlabel('Timestamps')
axs[0, 1].set_ylabel('Velocity Values')

# 绘制 _d 数据
axs[1, 0].scatter(timestamps, _d, c='green', alpha=0.5, s=2)
axs[1, 0].set_title('Distance Data')
axs[1, 0].set_xlabel('Timestamps')
axs[1, 0].set_ylabel('Distance Values')

# 绘制 _s 数据
print(len(_s))
axs[1, 1].scatter(range(len(_s)), _s, c='purple', alpha=0.5, s=2)
axs[1, 1].set_title('Scale Data')
axs[1, 1].set_xlabel('Timestamps')
axs[1, 1].set_ylabel('Scale Values')

# 调整子图布局
plt.tight_layout()

# 显示图表
plt.show()

