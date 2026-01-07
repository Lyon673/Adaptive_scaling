import json
import os
# JSON 数据
data = {"target": 72.9122517360759, "params": {"A_gp": 2.9110343838669928, "A_ipa": 1.0, "A_pp": 1.8081177326842737, "A_theta": 9.525454707504819, "A_v": 5.631658854725127, "C_base": 11.523280921852646, "K_g": 9.814210465008008, "K_p": 1.2}, "datetime": {"datetime": "2025-12-01 17:12:29", "elapsed": 124.631024, "delta": 57.08365}}



# 提取 params 并转换为 txt 格式
params_txt = "\n".join([f"{key}={value}" for key, value in data["params"].items()])

# 保存到文件
current_file_path = os.path.abspath(__file__)
current_dir = os.path.dirname(current_file_path)
output_path = os.path.join(current_dir, 'params.txt')
with open(output_path, "w") as file:
    file.write(params_txt)

print(f"参数已保存到 {output_path}")