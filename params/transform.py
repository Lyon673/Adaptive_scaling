import json
import os
# JSON 数据
data = {"target": 83.63687608926733, "params": {
    'K_g': 10.0,
    'K_p': 1.0,
    'C_base': 9.0,

    'A_theta': 10,
    'A_gp': 2.0,
    'A_pp': 3.5,
    'A_v': 4.3,
    'A_ipa': 2.0,

	'fixed_scale': 1.0,  
	'AFflag': 0 # 0-adaptive, 1-fixed
}}
# 提取 params 并转换为 txt 格式
params_txt = "\n".join([f"{key}={value}" for key, value in data["params"].items()])

# 保存到文件
current_file_path = os.path.abspath(__file__)
current_dir = os.path.dirname(current_file_path)
output_path = os.path.join(current_dir, 'params.txt')
with open(output_path, "w") as file:
    file.write(params_txt)

print(f"参数已保存到 {output_path}")