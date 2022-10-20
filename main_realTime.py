import os
import threading
import time

import sounddevice as sd
import torch


import infer_tool
import utils

if not os.path.exists("./record/"):
    os.mkdir("./record/")
infer_tool.del_temp_wav("./record/")


def play(data, fs):
    sd.play(data, fs)


# 这个不改
clean_name = 1
tran = 5
# 改成相应id
speaker_id = 3

# 每次合成长度，建议30s内，太高了爆掉显存(gtx1066一次15s以内）
model_name = "395_epochs.pth"  # 模型名称（pth文件夹下）
config_name = "sovits_pre.json"  # 模型配置（config文件夹下）


# 以下内容无需修改
hps_ms = utils.get_hparams_from_file(f"configs/{config_name}")
dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 获取config参数
target_sample = hps_ms.data.sampling_rate

threading.Thread(target=os.system, args=("python recorder.py",)).start()
jump = False
print("开始录音：")
while True:
    a = time.time()
    if os.path.exists("record/" + str(clean_name) + ".wav"):
        clean_names = [str(clean_name)]
    else:
        continue
    try:
        infer_tool.format_wav(f'./record/{clean_name}.wav', target_sample)
        source_path = f"./record/{clean_name}.wav"
        out_audio, out_sr = infer_tool.infer(source_path, speaker_id, tran)
        threading.Thread(target=play, args=(out_audio, target_sample)).start()
        os.system("del record\\" + str(clean_name) + ".wav")
        jump = False
    except Exception as e:
        print(e)
        os.system("del record\\" + str(clean_name) + ".wav")
        jump = True
    if jump:
        clean_name += 1
        continue

    clean_name += 1

    print("time taken: " + str(time.time() - a))
