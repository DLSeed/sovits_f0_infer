import threading
import wave

import pyaudio

import config
import utils
model_name = "395_epochs.pth"  # 模型名称（pth文件夹下）
config_name = "sovits_pre.json"  # 模型配置（config文件夹下）

# 以下内容无需修改
hps_ms = utils.get_hparams_from_file(f"configs/{config_name}")

# 获取config参数
target_sample = hps_ms.data.sampling_rate

input_filename = 1  # 麦克风采集的语音输入
input_filepath = "record/"  # 输入文件的path

p = pyaudio.PyAudio()
FORMAT = pyaudio.paInt16
CHANNELS = 1  # 声道数
RATE = target_sample  # 采样率
CHUNK = 256
stream = p.open(format=FORMAT,
                channels=CHANNELS,
                rate=RATE,
                input=True,
                frames_per_buffer=CHUNK)


def save_audio(wave_output_filename, channels, format, rate, frames):
    wf = wave.open(wave_output_filename, 'wb')
    wf.setnchannels(channels)
    wf.setsampwidth(p.get_sample_size(format))
    wf.setframerate(rate)
    wf.writeframes(b''.join(frames))
    wf.close()


def get_audio(filepath, stream):
    CHUNK = 256
    RECORD_SECONDS = 2  # 录音时间
    WAVE_OUTPUT_FILENAME = filepath

    frames = []
    for i in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
        data = stream.read(CHUNK)
        frames.append(data)

    threading.Thread(target=save_audio, args=(WAVE_OUTPUT_FILENAME, CHANNELS, FORMAT, RATE, frames,)).start()


# 联合上一篇博客代码使用，就注释掉下面，单独使用就不注释
while True:
    in_path = input_filepath + str(input_filename) + ".wav"
    get_audio(in_path, stream)
    input_filename += 1
