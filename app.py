# app.py
from flask import Flask, request, send_file
import torch
import torchaudio
from einops import rearrange
from stable_audio_tools import get_pretrained_model
from stable_audio_tools.inference.generation import generate_diffusion_cond
import os

app = Flask(__name__)

device = "cuda" if torch.cuda.is_available() else "cpu"

# モデルのロード（アプリケーション起動時に1回だけ実行）
model, model_config = get_pretrained_model("stabilityai/stable-audio-open-1.0")
sample_rate = model_config["sample_rate"]
sample_size = model_config["sample_size"]
model = model.to(device)

# リクエストの書き方
# curl -X POST http://xxxxxxx:7860/generate_audio -H "Content-Type: application/json" -d '{"prompt": "birds singing in forest", "duration": 5}' --output generated_music.wav
@app.route('/generate_audio', methods=['POST'])
def generate_audio():
    prompt = request.json.get('prompt')
    duration = request.json.get('duration', 10)  # デフォルトは10秒

    conditioning = [{
        "prompt": prompt,
        "seconds_start": 0,
        "seconds_total": duration
    }]

    output = generate_diffusion_cond(
        model,
        steps=100,
        cfg_scale=7,
        conditioning=conditioning,
        sample_size=sample_size,
        sigma_min=0.3,
        sigma_max=500,
        sampler_type="dpmpp-3m-sde",
        device=device
    )

    output = rearrange(output, "b d n -> d (b n)")
    output = output.to(torch.float32).div(torch.max(torch.abs(output))).clamp(-1, 1).mul(32767).to(torch.int16).cpu()

    output_path = "output.wav"
    torchaudio.save(output_path, output, sample_rate)

    return send_file(output_path, as_attachment=True)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=7860)