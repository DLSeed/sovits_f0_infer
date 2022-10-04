import logging
import os
import shutil
import subprocess

import demjson
import soundfile
import torch
import torchaudio

import infer_tool
from wav_temp import merge

infer_tool.mkdir(["./raw", "./pth", "./results"])
logging.getLogger('numba').setLevel(logging.WARNING)
# 自行下载hubert-soft-0d54a1f4.pt改名为hubert.pt放置于pth文件夹下
# https://github.com/bshall/hubert/releases/tag/v0.1
# pth文件夹，放置hubert、sovits模型
# 可填写音源文件列表，音源文件格式为wav，放置于raw文件夹下
clean_names = ["嘘月"]
# 合成多少歌曲时，若半音数量不足、自动补齐相同数量（按第一首歌的半音）
trans = [-9]  # 加减半音数（可为正负）
# 每首歌同时输出的speaker_id
id_list = [0]

model_name = "476_epochs.pth"  # 模型名称（pth文件夹下）
config_name = "sovits_pre.json"  # 模型配置（config文件夹下）

# 加载sovits模型、参数
net_g_ms, hubert_soft, feature_input, hps_ms = infer_tool.load_model(f"pth/{model_name}", f"configs/{config_name}")
speakers = demjson.decode_file(f"configs/{config_name}")["speakers"]
target_sample = hps_ms.data.sampling_rate
dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
infer_tool.fill_a_to_b(trans, clean_names)  # 自动补齐
input_wav_path = "./wav_temp/input"
out_wav_path = "./wav_temp/out"
infer_tool.mkdir([input_wav_path, out_wav_path])
print("mis连续超过10%时，考虑升降半音\n")
# 遍历列表
for clean_name, tran in zip(clean_names, trans):
    infer_tool.format_wav(f'./raw/{clean_name}.wav', target_sample)
    for spk_id in id_list:
        out_audio_name = model_name.split(".")[0] + f"_{clean_name}_{speakers[spk_id]}"
        raw_audio_path = f"./raw/{clean_name}.wav"
        audio, sample_rate = torchaudio.load(raw_audio_path)
        audio_time = audio.shape[-1] / target_sample
        val_list = []
        # 清除缓存文件
        infer_tool.del_temp_wav("./wav_temp")
        # 源音频切割方案
        if audio_time > 1.3 * int(cut_time):
            proc = subprocess.Popen(
                f"python slicer.py {raw_audio_path} --out_name {out_audio_name} --out {input_wav_path}  --db_thresh -30",
                shell=True)
            proc.wait()
        else:
            shutil.copy(f"./raw/{clean_name}.wav", f"{input_wav_path}/{out_audio_name}-00.wav")

        count = 0
        file_list = os.listdir(input_wav_path)
        len_file_list = len(file_list)
        for file_name in file_list:
            raw_path = f"{input_wav_path}/{file_name}"
            out_path = f"{out_wav_path}/{file_name}"

            out_audio, out_sr = infer_tool.infer(raw_path, spk_id, tran, net_g_ms, hubert_soft, feature_input)
            soundfile.write(out_path, out_audio, target_sample)

            mistake = infer_tool.calc_error(raw_path, out_path, tran, hubert_soft, feature_input)
            val_list.append(mistake)
            count += 1
            print(f"{file_name}: {round(100 * count / len_file_list, 2)}%   mis:{mistake}%")
        print(f"\n分段误差参考：1%优秀，3%左右合理，5%-8%可以接受\n{val_list}")
        merge.run(out_audio_name)
