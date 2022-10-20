import logging
import os
import sys
import time

import matplotlib.pyplot as plt
import numpy as np
import torch
import torchaudio

import harmof0
from sovits import hubert_model
from sovits import utils
from sovits.models import SynthesizerTrn
from sovits.preprocess_wave import FeatureInput

sys.setrecursionlimit(1000000)
logging.getLogger('matplotlib').setLevel(logging.WARNING)
dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def timeit(func):
    def run(*args, **kwargs):
        t = time.time()
        res = func(*args, **kwargs)
        print('executing \'%s\' costed %.3fs' % (func.__name__, time.time() - t))
        return res

    return run


def get_end_file(dir_path, end):
    file_lists = []
    for root, dirs, files in os.walk(dir_path):
        files = [f for f in files if f[0] != '.']
        dirs[:] = [d for d in dirs if d[0] != '.']
        for f_file in files:
            if f_file.endswith(end):
                file_lists.append(os.path.join(root, f_file).replace("\\", "/"))
    return file_lists


def load_model(model_path, config_path):
    # 获取模型配置
    hps_ms = utils.get_hparams_from_file(config_path)
    n_g_ms = SynthesizerTrn(
        178,
        hps_ms.data.filter_length // 2 + 1,
        hps_ms.train.segment_size // hps_ms.data.hop_length,
        n_speakers=hps_ms.data.n_speakers,
        **hps_ms.model)
    _ = utils.load_checkpoint(model_path, n_g_ms, None)
    _ = n_g_ms.eval().to(dev)
    # 加载hubert
    hubert_soft = hubert_model.hubert_soft(get_end_file("./pth", "pt")[0])
    feature_input = FeatureInput(hps_ms.data.sampling_rate, hps_ms.data.hop_length)
    return n_g_ms, hubert_soft, feature_input, hps_ms


def resize2d_f0(x, target_len):
    source = np.array(x)
    source[source < 0.001] = np.nan
    target = np.interp(np.arange(0, len(source) * target_len, len(source)) / target_len, np.arange(0, len(source)),
                       source)
    res = np.nan_to_num(target)
    return res


def get_units(in_path, hubert_soft):
    source, sr = torchaudio.load(in_path)
    source = torchaudio.functional.resample(source, sr, 16000)
    if len(source.shape) == 2 and source.shape[1] >= 2:
        source = torch.mean(source, dim=0).unsqueeze(0)
    source = source.unsqueeze(0).to(dev)
    with torch.inference_mode():
        units = hubert_soft.units(source)
        return units


@timeit
def transcribe(source_path, length, transform, feature_input):
    # feature_pit = feature_input.compute_f0(source_path)
    feature_pit = harmof0.extract_file_f0(source_path)
    feature_pit = feature_pit * 2 ** (transform / 12)
    feature_pit = resize2d_f0(feature_pit, length)
    coarse_pit = feature_input.coarse_f0(feature_pit)
    return coarse_pit


def get_unit_pitch(in_path, tran, hubert_soft, feature_input):
    soft = get_units(in_path, hubert_soft).squeeze(0).cpu().numpy()
    input_pitch = transcribe(in_path, soft.shape[0], tran, feature_input)
    return soft, input_pitch


def clean_pitch(input_pitch):
    num_nan = np.sum(input_pitch == 1)
    if num_nan / len(input_pitch) > 0.9:
        input_pitch[input_pitch != 1] = 1
    return input_pitch


def plt_pitch(input_pitch):
    input_pitch = input_pitch.astype(float)
    input_pitch[input_pitch == 1] = np.nan
    return input_pitch


def f0_to_pitch(ff):
    f0_pitch = 69 + 12 * np.log2(ff / 440)
    return f0_pitch


def f0_plt(in_path, out_path, tran, hubert_soft, feature_input):
    s1, input_pitch = get_unit_pitch(in_path, tran, hubert_soft, feature_input)
    s2, output_pitch = get_unit_pitch(out_path, 0, hubert_soft, feature_input)
    plt.clf()
    plt.plot(plt_pitch(input_pitch), color="#66ccff")
    plt.plot(plt_pitch(output_pitch), color="orange")
    plt.savefig("temp.jpg")


def calc_error(in_path, out_path, tran, feature_input):
    input_pitch = feature_input.compute_f0(in_path)
    output_pitch = feature_input.compute_f0(out_path)
    sum_y = []
    if np.sum(input_pitch == 0) / len(input_pitch) > 0.9:
        mistake, var_take = 0, 0
    else:
        for i in range(min(len(input_pitch), len(output_pitch))):
            if input_pitch[i] > 0 and output_pitch[i] > 0:
                sum_y.append(abs(f0_to_pitch(output_pitch[i]) - (f0_to_pitch(input_pitch[i]) + tran)))
        num_y = 0
        for x in sum_y:
            num_y += x
        len_y = len(sum_y) if len(sum_y) else 1
        mistake = round(float(num_y / len_y), 2)
        var_take = round(float(np.std(sum_y, ddof=1)), 2)
    return mistake, var_take


def infer(source_path, speaker_id, tran, net_g_ms, hubert_soft, feature_input):
    sid = torch.LongTensor([int(speaker_id)]).to(dev)
    soft, pitch = get_unit_pitch(source_path, tran, hubert_soft, feature_input)
    pitch = torch.LongTensor(clean_pitch(pitch)).unsqueeze(0).to(dev)
    stn_tst = torch.FloatTensor(soft)
    with torch.no_grad():
        x_tst = stn_tst.unsqueeze(0).to(dev)
        x_tst_lengths = torch.LongTensor([stn_tst.size(0)]).to(dev)
        audio = \
            net_g_ms.infer(x_tst, x_tst_lengths, pitch, sid=sid, noise_scale=0.3, noise_scale_w=0.5,
                           length_scale=1)[0][
                0, 0].data.float().cpu().numpy()
    return audio, audio.shape[-1]


def del_temp_wav(path_data):
    for i in get_end_file(path_data, "wav"):  # os.listdir(path_data)#返回一个列表，里面是当前目录下面的所有东西的相对路径
        os.remove(i)


def format_wav(audio_path, tar_sample):
    raw_audio, raw_sample_rate = torchaudio.load(audio_path)
    if len(raw_audio.shape) == 2 and raw_audio.shape[1] >= 2:
        raw_audio = torch.mean(raw_audio, dim=0).unsqueeze(0)
    tar_audio = torchaudio.functional.resample(raw_audio, raw_sample_rate, tar_sample)
    torchaudio.save(audio_path[:-4] + ".wav", tar_audio, tar_sample)
    return tar_audio, tar_sample


def fill_a_to_b(a, b):
    if len(a) < len(b):
        for _ in range(0, len(b) - len(a)):
            a.append(a[0])


def mkdir(paths: list):
    for path in paths:
        if not os.path.exists(path):
            os.mkdir(path)
