import json
import os
# JSON 数据
data = {"target": 35.56008638295057, "params": {"A_gp": 2.252163132973055, "A_theta": 4.628285795238272, "A_v": 4.424680868208455, "B_safety": 3.828077489162437}, "datetime": {"datetime": "2026-04-02 13:58:54", "elapsed": 757.222729, "delta": 109.906862}}


# 提取 params 并转换为 txt 格式
params_txt = "\n".join([f"{key}={value}" for key, value in data["params"].items()])

# 保存到文件
current_file_path = os.path.abspath(__file__)
current_dir = os.path.dirname(current_file_path)
output_path = os.path.join(current_dir, 'params.txt')
with open(output_path, "w") as file:
    file.write(params_txt)

print(f"参数已保存到 {output_path}")