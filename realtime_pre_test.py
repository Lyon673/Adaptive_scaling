"""
simulate_latency.py
===================
独立的实时阶段预测延迟评估工具 (Latency Benchmark)。
使用测试集中的真实演示数据，模拟 maincb 的高频逐帧调用，
以评估 LSTM 或 TeCNO 模型在实际部署时的真实耗时与吞吐量上限。
"""

import time
import numpy as np
import matplotlib.pyplot as plt
import os
import sys

# 确保能正确导入 Trajectory 目录下的模块
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

# =====================================================================
# 【模型切换区】: 你可以在这里切换测试 LSTM 还是 TeCNO
# =====================================================================
#from realtime_phase_predictor import PhasePredictor
from realtime_phase_predictor_tecno import PhasePredictor

from Trajectory.load_data import load_test_state
from Trajectory.config import without_quat, resample

# 模型权重路径配置 (请根据你测试的模型修改路径)
MODEL_PATH = os.path.join(
    # current_dir, 'Trajectory', 'LSTM_model', 'lstm_sequence_model.pth'
    current_dir, 'Trajectory', 'TeCNO_model', 'tecno_sequence_model.pth'
)


def run_latency_simulation():
    print("="*50)
    print("1. 正在加载测试集演示数据...")
    # 按照训练时的标准排除特定 Demo，并加载测试集
    excluded = [80, 81, 92, 109, 112, 117, 122, 144, 145]
    demo_id_list = np.delete(np.arange(148), excluded)
    
    test_states = load_test_state(without_quat=without_quat, resample=resample, demo_id_list=demo_id_list)
    
    if not test_states or len(test_states) == 0:
        print("错误: 无法加载测试集数据！")
        return

    # 选取测试集的第一条轨迹进行模拟
    demo_sequence = test_states[0]
    num_frames = len(demo_sequence)
    print(f"成功加载测试序列，总帧数: {num_frames} 帧")

    print("\n2. 正在初始化 PhasePredictor...")
    try:
        predictor = PhasePredictor(model_path=MODEL_PATH, min_frames=3)
    except Exception as e:
        print(f"初始化预测器失败，请检查路径或配置: {e}")
        return

    latencies = []
    
    print("\n3. 开始逐帧模拟并记录耗时 (Latency)...")
    for i in range(num_frames):
        current_state = demo_sequence[i]
        
        # 逆向解析：从处理好的状态向量中，剥离出原始的 Position 和 Gripper 供 update 使用
        # realtime_phase_predictor 内部提取 8 维特征的逻辑是: [0, 1, 2, 7, 8, 9, 10, 15]
        if without_quat:
            # 数据集长度为 8
            L_pos3    = current_state[0:3]
            L_gripper = current_state[3]
            R_pos3    = current_state[4:7]
            R_gripper = current_state[7]
        else:
            # 数据集长度为 16
            L_pos3    = current_state[0:3]
            L_gripper = current_state[7]
            R_pos3    = current_state[8:11]
            R_gripper = current_state[15]

        # ------------------------------------------------
        # 核心计时区 (精确到纳秒级别)
        # ------------------------------------------------
        start_time = time.perf_counter()
        
        phase_label, phase_probs = predictor.update(
            L_pos3=L_pos3,
            R_pos3=R_pos3,
            L_gripper=L_gripper,
            R_gripper=R_gripper
        )
        
        end_time = time.perf_counter()
        # ------------------------------------------------
        
        latency_ms = (end_time - start_time) * 1000.0
        latencies.append(latency_ms)

        if (i + 1) % 50 == 0 or (i + 1) == num_frames:
            print(f"  Processed {i+1:03d}/{num_frames} frames | Current Phase: {phase_label} | Latency: {latency_ms:.3f} ms")

    # ------------------- 统计与可视化 -------------------
    latencies = np.array(latencies)
    avg_latency = np.mean(latencies)
    max_latency = np.max(latencies)
    min_latency = np.min(latencies)
    p99_latency = np.percentile(latencies, 99) # 99% 的帧都在这个时间以内完成
    
    print("\n" + "="*50)
    print(" 性能测试评估报告 (Latency Benchmark Report) ")
    print("="*50)
    print(f"总测试帧数           : {num_frames} 帧")
    print(f"平均单帧耗时 (Avg)   : {avg_latency:.3f} ms")
    print(f"最大单帧耗时 (Max)   : {max_latency:.3f} ms")
    print(f"最小单帧耗时 (Min)   : {min_latency:.3f} ms")
    print(f"99分位耗时   (P99)   : {p99_latency:.3f} ms")
    print(f"理论最高帧率 (FPS)   : {1000.0 / avg_latency:.1f} Hz")
    print("="*50)

    # 绘制耗时分布折线图
    plt.figure(figsize=(12, 6))
    plt.plot(latencies, label='Per-frame Latency (ms)', color='steelblue', alpha=0.8, linewidth=1.5)
    plt.axhline(avg_latency, color='orange', linestyle='--', linewidth=2, label=f'Average: {avg_latency:.2f} ms')
    plt.axhline(p99_latency, color='red', linestyle=':', linewidth=2, label=f'P99: {p99_latency:.2f} ms')
    plt.title('Real-time Phase Predictor Inference Latency', fontsize=14)
    plt.xlabel('Frame Sequence', fontsize=12)
    plt.ylabel('Latency (ms)', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend(fontsize=11)
    plt.tight_layout()
    
    save_path = os.path.join(current_dir, 'latency_benchmark.png')
    plt.savefig(save_path, dpi=150)
    print(f"\n[可视化] 耗时折线图已保存至: {save_path}")

if __name__ == '__main__':
    run_latency_simulation()