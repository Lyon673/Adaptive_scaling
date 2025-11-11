import json

# JSON 数据
data = {"target": 83.63687608926733, "params": {"W_d": 0.20714375522588313, "W_dp": 0.5347289313727411, "W_dv": 1.906408808850459, "W_p": 0.23330603251763002, "W_pv": 0.37588053440469194, "W_v": 0.8703180460798396, "Y_base": 0.06099321852596262, "tau_d": 0.8090221786759244, "tau_p": 0.8914474669753404, "tau_v": 0.7960409400325988}, "datetime": {"datetime": "2025-07-11 16:08:48", "elapsed": 217.658285, "delta": 79.215168}}


# 提取 params 并转换为 txt 格式
params_txt = "\n".join([f"{key}={value}" for key, value in data["params"].items()])

# 保存到文件
output_path = "/home/lambda/SurRoL/Project/params/params.txt"
with open(output_path, "w") as file:
    file.write(params_txt)

print(f"参数已保存到 {output_path}")