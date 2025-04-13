# Disfluency Detection From Audio as an API

This repo includes an api for running audio through the language, acoustic, and multimodal disfluency detection models. 
It also includes extra data such as loudness and pitch. This is a fork of https://github.com/amritkromana/disfluency_detection_from_audio. 

# Disfluency Detection Demo

## Dependencies 

Use uv to install dependencies.

Use gdown to download the pretrained model weights and save to demo_models: 
```
mkdir demo_models && cd demo_models
mkdir asr && cd asr
gdown --id 1BeT7m_5qv19Sb5yrZ2zhKu6fEprUoB9N -O config.json
gdown --id 13n8VrTFVq4jGouCDamkReHlHm_1yz20U -O pytorch_model.bin
cd ..
gdown --id 1GQIXgCSF3Usiuy5hkxgOl483RPX3f_SX -O language.pt
gdown --id 1wWrmopvvdhlBw-cL7EDyih9zn_IJu5Wr -O acoustic.pt
gdown --id 1LPchbScA_cuFx1XoNxpFCYZfGoJCfWao -O multimodal.pt
```

```
uv run main.py
```
