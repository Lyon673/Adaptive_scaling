import json
import os

# JSON 数据
data = {"target": 52.03292118102036, "params": {"A_gp": 2.3357273246141084, "A_theta": 4.678460849731229, "A_v": 4.411627055269589, "B_safety": 3.8912488509488146}, "datetime": {"datetime": "2026-04-06 16:32:10", "elapsed": 284.683472, "delta": 70.483562}}

# 提取 params 并转换为 txt 格式
params_txt = "\n".join([f"{key}={value}" for key, value in data["params"].items()])

# 保存到文件
current_file_path = os.path.abspath(__file__)
current_dir = os.path.dirname(current_file_path)
output_path = os.path.join(current_dir, 'params.txt')
with open(output_path, "w") as file:
    file.write(params_txt)

print(f"参数已保存到 {output_path}")