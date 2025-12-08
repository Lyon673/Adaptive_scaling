import numpy as np
import matplotlib.pyplot as plt
from collections import deque
import socket
import ipa
import pywt


"""
more data required:
    1. long time recording
    2. with a causal gaze period at the beginning
    3. repeat
"""

"""
gaze01 : without gazing, and two gazing later

"""

pupil = np.load('/home/lambda/surgical_robotics_challenge/scripts/surgical_robotics_challenge/iros/gaze/gaze03.npy', allow_pickle=True)

"""l = len(pupil)
pupil = [(pupil[i]+pupil[i+1]+pupil[i+2])/3 for i in range(0, l-2)]
"""

l = len(pupil)
zeropos = []
for i in range(l):
    if pupil[i][0] == 0.0:
        pupil[i][0] = np.mean([pupil[i-1][0],pupil[i-2][0],pupil[i-3][0]])
        #pupil[i][0] = pupil[i-1][0]
        zeropos.append(i)
    if pupil[i][4] == pupil[i-1][4]:
        np.delete(pupil, i)

    
l = len(pupil)

sampling_rate = 60.0  # Hz

"""signal = [i[0] for i in pupil]
# 计算 FFT
fft_result = np.fft.fft(signal)
fft_magnitude = np.abs(fft_result)  
fft_magnitude = fft_magnitude[:len(fft_magnitude) // 2]  

frequencies = np.fft.fftfreq(len(signal), d=1.0 / sampling_rate)
frequencies = frequencies[:len(frequencies) // 2]  

plt.figure(figsize=(10, 6))
plt.plot(frequencies, fft_magnitude, color='blue')
plt.title("Frequency Spectrum")
plt.xlabel("Frequency (Hz)")
plt.ylabel("Amplitude")
plt.grid()
plt.show()"""

# lowpass filter
coeffs = pywt.wavedec([i[0] for i in pupil], 'db4', level=3)

denoised_coeffs = [coeffs[0]]  # 保留近似系数（不做处理）
for c in coeffs[1:]:  # 只对细节系数进行阈值处理
    threshold = np.sqrt(2 * np.log(len(c))) * np.std(c)
    denoised_coeffs.append(pywt.threshold(c, threshold, mode='soft'))
denoised_signal = pywt.waverec(denoised_coeffs, 'db4')

fig, axs = plt.subplots(2,1, figsize=(10, 8))
axs[0].scatter(range(l), [i[0] for i in pupil], c='red', alpha=0.5, s=2)
axs[1].plot(range(l), denoised_signal[:l] , c='blue', alpha=0.5)
plt.show()


for i in range(l):
    pupil[i][0] = denoised_signal[i]

"""# pre process
dilation = []
for i in range(2,l):
    if pupil[i][4] == pupil[i-1][4]:
        continue
    speed = np.abs((pupil[i][0]-pupil[i-1][0])/(pupil[i][4]-pupil[i-1][4])*1e6)
    if len(dilation) > 20.0:
        mad = np.median(np.abs(dilation - np.median(dilation)))
        threshold = np.median(dilation)+ 10 * mad
        print(f"<LYON> speed: {speed}, threshold: {threshold}")
        if speed > threshold:
            pupil[i][0] = np.mean([i[0] for i in pupil[i-3:i]])
            pupil[i+1][0] = np.mean([i[0] for i in pupil[i-2:i+1]])
            pupil[i+2][0] = np.mean([i[0] for i in pupil[i-1:i+2]])
            pupil[i+3][0] = np.mean([i[0] for i in pupil[i:i+3]])
        speed = np.abs((pupil[i][0]-pupil[i-1][0])/(pupil[i][4]-pupil[i-1][4])*1e6)
    dilation.append(speed)
"""
dl = len(denoised_signal)

print(f"<LYON> l: {l}, dl: {dl}")


plt.scatter(range(l), [i[0] for i in pupil], c='red', alpha=0.5, s=2)
plt.show()

"""for i in range(127,l):
    print(pupil[i].timestamp-pupil[i-127].timestamp)"""

timestamp_diff = []

for i in range(64, l-64):
    timestamp_diff.append(pupil[i+63][4]*1e-6 - pupil[i-64][4]*1e-6)


pupildata = np.array([ipa.Pupil(i[0],i[4]*1e-6) for i in pupil])

ipadata, position, threshold = ipa.ipa_cal(pupildata)
print(f"<LYON> whole process threshold: {threshold}")

ipa_L = np.array([])
#ipa 
for i in range(64, l-64):
    try:
        ipa_data,_,_ = ipa.ipa_cal(pupildata, 0, False, timestamp=i)
        ipa_L = np.append(ipa_L, ipa_data)
    except ValueError:
        print("ValueError")
        continue

#ipa_L = np.array([x for x in ipa_L if x>0.2 and x<0.6])



valid = []
for i in range(l):
    if pupil[i][2] and pupil[i][3]:
        valid.append(1)
    else :
        valid.append(0)

"""plt.scatter(range(l), valid, c='green', alpha=0.5, s=2)
plt.show()"""

"""plt.plot(range(l), [i[0] for i in pupil], c='green', alpha=0.5)
plt.show()"""



"""for i in range(l):
    print(f"<LYON> left pupil: {pupil[i][0]}, right pupil: {pupil[i][1]}, left valid: {pupil[i][2]}, right valid: {pupil[i][3]}")
"""



ipadata, position,_ = ipa.ipa_cal(pupildata)
print(f"<LYON> ipadata: {ipadata}, postion: {position}")
pos = np.zeros(l)
pos[position] = 1

ipa_L_pad = np.pad(ipa_L, pad_width=(64, 64), mode='constant', constant_values=0)
zero = np.zeros(l)
zero[zeropos] = 1

fig, axs = plt.subplots(3,1, figsize=(10, 8))
axs[0].scatter(range(l), ipa_L_pad, c='red', alpha=0.5, s=2)
axs[1].plot(range(l), pos , c='blue', alpha=0.5)
axs[2].plot(range(l), zero, c='green', alpha=0.5)
plt.show()


fig, axs = plt.subplots(2,1, figsize=(10, 8))
axs[0].scatter(range(l), [i[0] for i in pupil], c='red', alpha=0.5, s=2)
axs[1].plot(range(l), pos , c='blue', alpha=0.5)
plt.show()

print(f"<LYON> zeropos: {zeropos}")
print(f"<LYON> l: {l}")


"""# 小波分解
coeffs = pywt.wavedec([i[0] for i in pupil], 'sym16', 'per', level=4)

# 遍历每一级的细节系数，逐级重构并绘图
for i in range(1, len(coeffs)):  # 从第 1 层细节系数开始
    # 初始化近似系数为 0
    approx = np.zeros_like(coeffs[0])
    
    # 仅保留当前层的细节系数，其他层置为 0
    detail = [np.zeros_like(c) if j != i else coeffs[j] for j, c in enumerate(coeffs)]
    
    # 重构信号
    reconstructed_signal = pywt.waverec([approx] + detail[1:], 'sym16', 'per')
    
    # 绘制当前层的重构信号
    plt.figure(figsize=(10, 6))
    plt.plot([i[0] for i in pupil], label='Original Signal', alpha=0.5)
    plt.plot(reconstructed_signal, label=f'Reconstructed from Level {i} Detail Coefficients', linestyle='dashed')
    plt.legend()
    plt.title(f"Reconstructed Signal from Level {i} Detail Coefficients")
    plt.xlabel("Time")
    plt.ylabel("Amplitude")
    plt.grid()
    plt.show()"""