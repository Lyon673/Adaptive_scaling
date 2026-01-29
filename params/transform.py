import json
import os
# JSON 数据
data = {"target": 75.49363678783594, "params": {"A_gp": 3.5411088007000284, "A_theta": 1.176563522353645, "A_v": 4.796654829799902, "B_safety": 3.8414050806778333, "C_base": 13.541027439424639, "K_g": 9.876439050136206, "K_p": 1.7858739691608618}, "datetime": {"datetime": "2026-01-20 16:28:01", "elapsed": 528.639971, "delta": 110.925817}}


# 提取 params 并转换为 txt 格式
params_txt = "\n".join([f"{key}={value}" for key, value in data["params"].items()])

# 保存到文件
current_file_path = os.path.abspath(__file__)
current_dir = os.path.dirname(current_file_path)
output_path = os.path.join(current_dir, 'params.txt')
with open(output_path, "w") as file:
    file.write(params_txt)

print(f"参数已保存到 {output_path}")