import json
import os
# JSON 数据
data = {"target": 83.63687608926733, "params": {"A_gp": 1.224963017405465, "A_ipa": 1.3985440853709183, "A_pp": 2.950066579949716, "A_theta": 11.90522164789582, "A_v": 5.844552386514674, "C_base": 14.966792790891242, "K_g": 11.921821047619085, "K_p": 0.9763159840958847}}
# 提取 params 并转换为 txt 格式
params_txt = "\n".join([f"{key}={value}" for key, value in data["params"].items()])

# 保存到文件
current_file_path = os.path.abspath(__file__)
current_dir = os.path.dirname(current_file_path)
output_path = os.path.join(current_dir, 'params.txt')
with open(output_path, "w") as file:
    file.write(params_txt)

print(f"参数已保存到 {output_path}")