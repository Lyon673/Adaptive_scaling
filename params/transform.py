import json
import os
# JSON 数据
data = {"target": 80.95489420201923, "params": {"A_gp": 4.0, "A_theta": 1.8789074708107931, "A_v": 3.614237437497564, "B_safety": 4.0, "C_base": 10.686741227348275, "K_g": 8.0, "K_p": 1.422413680058934}, "datetime": {"datetime": "2026-01-18 15:37:07", "elapsed": 2072.414197, "delta": 52.91894}}


# 提取 params 并转换为 txt 格式
params_txt = "\n".join([f"{key}={value}" for key, value in data["params"].items()])

# 保存到文件
current_file_path = os.path.abspath(__file__)
current_dir = os.path.dirname(current_file_path)
output_path = os.path.join(current_dir, 'params.txt')
with open(output_path, "w") as file:
    file.write(params_txt)

print(f"参数已保存到 {output_path}")