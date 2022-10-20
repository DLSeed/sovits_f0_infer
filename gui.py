# coding=utf-8
import logging

import gradio as gr
import soundfile
import torch

import infer_tool

logging.getLogger('numba').setLevel(logging.WARNING)
dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")

spk_dict = {}

net_g_ms, hubert_soft, feature_input, hps_ms = None, None, None, None


def load_model(md_name, cfg_json):
    global net_g_ms, hubert_soft, feature_input, hps_ms
    net_g_ms, hubert_soft, feature_input, hps_ms = infer_tool.load_model(md_name, cfg_json)
    global spk_dict
    if "speakers" in hps_ms.keys():
        for i, spk in enumerate(hps_ms.speakers):
            spk_dict[spk] = i
        spk_list = list(spk_dict.keys())
    else:
        spk_list = ["0"]
    return "模型加载成功", gr.Dropdown.update(choices=spk_list)


def infer(sid, audio_record, audio_upload, tran):
    if audio_upload is not None:
        audio_path = audio_upload
    elif audio_record is not None:
        audio_path = audio_record
    else:
        return "你需要上传wav文件或使用网页内置的录音！", None
    target_sample = hps_ms.data.sampling_rate
    o_audio, out_sr = infer_tool.infer(audio_path, spk_dict[sid], tran, net_g_ms, hubert_soft, feature_input)
    out_path = f"./out_temp.wav"
    soundfile.write(out_path, o_audio, target_sample)
    infer_tool.f0_plt(audio_path, out_path, tran, hubert_soft, feature_input)
    mistake, var = infer_tool.calc_error(audio_path, out_path, tran, feature_input)
    return f"分段误差参考：0.3优秀，0.5左右合理，少量0.8-1可以接受\n若偏差过大，请调整升降半音数；多次调整均过大、说明超出歌手音域\n半音偏差：{mistake}\n半音方差：{var}", (
        target_sample, o_audio), gr.Image.update("temp.jpg")


app = gr.Blocks()
with app:
    with gr.Tabs():
        with gr.TabItem("合成"):
            gr.Markdown(value="""
            本模型为sovits_f0，支持**45s以内**的**无伴奏**wav、mp3、aac、m4a格式，或使用**网页内置**的录音（二选一）
            
            转换效果取决于源音频语气、节奏是否与目标音色相近。
            
            **即霜**转换女声效果较好，**女声常用音域误差均在1%-3%。**
            
            转换女声干声时，若目标音色为男声，**建议降6-9key**，**误差越接近0，音准越高**
            
            """)
            model_name = gr.Dropdown(label="模型", choices=infer_tool.get_end_file("./pth", "pth"))
            config_json = gr.Dropdown(label="配置", choices=infer_tool.get_end_file("./configs", "json"))
            vc_config = gr.Button("加载模型", variant="primary")
            model_mess = gr.Textbox(label="Output Message")

            with gr.Box():
                speaker_id = gr.Dropdown(label="目标音色")
                record_input = gr.Audio(source="microphone", label="录制你的声音", type="filepath",
                                        elem_id="audio_inputs")
                upload_input = gr.Audio(source="upload", label="上传音频（长度小于45秒）", type="filepath",
                                        elem_id="audio_inputs")
                vc_transform = gr.Number(label="变调（整数，可以正负，半音数量，升高八度就是12）", value=0)
                vc_submit = gr.Button("转换", variant="primary")
                out_message = gr.Textbox(label="Output Message")
                out_audio = gr.Audio(label="Output Audio")
                f0_image = gr.Image(label="f0曲线")
            vc_config.click(load_model, [model_name, config_json], [model_mess, speaker_id])
            vc_submit.click(infer, [speaker_id, record_input, upload_input, vc_transform],
                            [out_message, out_audio, f0_image])
        with gr.TabItem("使用说明"):
            gr.Markdown(value="""
                        0、合集：https://github.com/IceKyrin/sovits_guide/blob/main/README.md
                        
                        1、仅支持sovit_f0（sovits2.0）模型
                        
                        2、自行下载hubert-soft-0d54a1f4.pt改名为hubert.pt放置于pth文件夹下（已经下好了）
                            https://github.com/bshall/hubert/releases/tag/v0.1

                        3、pth文件夹下放置sovits2.0的模型
                        
                        4、与模型配套的xxx.json，需有speaker项——人物列表
                        
                        5、放无伴奏的音频、或网页内置录音，不要放奇奇怪怪的格式
                        
                        6、仅供交流使用，不对用户行为负责

                        """)
    app.launch()
