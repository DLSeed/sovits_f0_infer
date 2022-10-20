import logging
import os
import subprocess

import demjson
import soundfile
import torch

from sovits import infer_tool
from wav_temp import merge

infer_tool.mkdir(["./raw", "./pth", "./results"])
logging.getLogger('numba').setLevel(logging.WARNING)
# 自行下载hubert-soft-0d54a1f4.pt改名为hubert.pt放置于pth文件夹下
# https://github.com/bshall/hubert/releases/tag/v0.1
# pth文件夹，放置hubert、sovits模型
# 可填写音源文件列表、名称不含空格，音源文件格式为wav，放置于raw文件夹下
clean_names = ["变成猫了"]
# 合成多少歌曲时，若半音数量不足、自动补齐相同数量（按第一首歌的半音）
trans = [-3]  # 加减半音数（可为正负）
# 每首歌同时输出的speaker_id
id_list = [4]

model_name = "152_epochs.pth"  # 模型名称（pth文件夹下）
config_name = "nyarumul.json"  # 模型配置（config文件夹下）

# 加载sovits模型、参数
net_g_ms, hubert_soft, feature_input, hps_ms = infer_tool.load_model(f"pth/{model_name}", f"configs/{config_name}")
speakers = demjson.decode_file(f"configs/{config_name}")["speakers"]
target_sample = hps_ms.data.sampling_rate
dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
infer_tool.fill_a_to_b(trans, clean_names)  # 自动补齐
input_wav_path = "./wav_temp/input"
out_wav_path = "./wav_temp/output"
infer_tool.mkdir([input_wav_path, out_wav_path])
print("mis连续超过10%时，考虑升降半音\n")
# 遍历列表
for clean_name, tran in zip(clean_names, trans):
    raw_audio_path = f"./raw/{clean_name}.wav"
    infer_tool.format_wav(raw_audio_path, target_sample)
    for spk_id in id_list:
        # 清除缓存文件
        infer_tool.del_temp_wav("./wav_temp")
        var_list = []
        mis_list = []
        out_audio_name = model_name.split(".")[0] + f"_{clean_name}_{speakers[spk_id]}"

        proc = subprocess.Popen(
            f"python ./sovits/slicer.py {raw_audio_path} --out_name {out_audio_name} --out {input_wav_path}  --db_thresh -30",
            shell=True).wait()
        # shutil.copy(raw_audio_path, f"{input_wav_path}/{out_audio_name}-00.wav")
        count = 0
        file_list = os.listdir(input_wav_path)
        len_file_list = len(file_list)
        for file_name in file_list:
            raw_path = f"{input_wav_path}/{file_name}"
            out_path = f"{out_wav_path}/{file_name}"

            out_audio, out_sr = infer_tool.infer(raw_path, spk_id, tran, net_g_ms, hubert_soft, feature_input)
            soundfile.write(out_path, out_audio, target_sample)

            infer_tool.f0_plt(raw_path, out_path, tran, hubert_soft, feature_input)
            mistake, var = infer_tool.calc_error(raw_path, out_path, tran, feature_input)
            mis_list.append(mistake)
            var_list.append(var)
            count += 1
            print(f"{file_name}: {round(100 * count / len_file_list, 2)}%  mis:{mistake} var:{var}")
        print(
            f"分段误差参考：0.3优秀，0.5左右合理，少量0.8-1可以接受\n若偏差过大，请调整升降半音数；多次调整均过大、说明超出歌手音域\n半音偏差：{mis_list}\n半音方差：{var_list}")
        merge.run(out_audio_name)
        # 清除缓存文件
        infer_tool.del_temp_wav("./wav_temp")
