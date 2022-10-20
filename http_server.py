import soundfile
import uvicorn
from fastapi import FastAPI, File
from starlette.responses import FileResponse

from sovits import infer_tool

app = FastAPI()
http_temp_path = "./wav_temp/http"
infer_tool.mkdir(["./wav_temp", http_temp_path])
net_g_ms, hubert_soft, feature_input, hps_ms = None, None, None, None


# 加载模型、配套json
@app.get("/model")
async def load_model(model_name: str = "152_epochs.pth", config_name: str = "nyarumul.json"):
    global net_g_ms, hubert_soft, feature_input, hps_ms
    net_g_ms, hubert_soft, feature_input, hps_ms = infer_tool.load_model(f"pth/{model_name}", f"configs/{config_name}")


# 上传示例参考http_cli.py
# 以bytes形式上传文件名为audio的音频，speaker为音色序号
@app.get("/svc", response_class=FileResponse)
async def infer(speaker: str, tran: str, audio: bytes = File(..., max_length=2097152)):
    target_sample = hps_ms.data.sampling_rate
    with open(f"{http_temp_path}/http_in.wav", "wb") as f:
        f.write(audio)
    out_audio, out_sr = infer_tool.infer(f"{http_temp_path}/http_in.wav", int(speaker), int(tran), net_g_ms,
                                         hubert_soft,feature_input)
    soundfile.write(f"{http_temp_path}/http_out.wav", out_audio, target_sample)
    return f"{http_temp_path}/http_out.wav"


uvicorn.run(app, host="127.0.0.1", port=8000)
