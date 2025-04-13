# Disfluency Detection From Audio as an API

This repo includes an api for running audio through the language, acoustic, and multimodal disfluency detection models. 
It also includes extra data such as loudness and pitch. This is a fork of https://github.com/amritkromana/disfluency_detection_from_audio. 

# Disfluency Detection Demo

## Dependencies 

Use uv to install dependencies.

Use gdown to download the pretrained model weights and save to demo_models: 
```bash
mkdir demo_models && cd demo_models
mkdir asr && cd asr
gdown --id 1BeT7m_5qv19Sb5yrZ2zhKu6fEprUoB9N -O config.json
gdown --id 13n8VrTFVq4jGouCDamkReHlHm_1yz20U -O pytorch_model.bin
cd ..
gdown --id 1GQIXgCSF3Usiuy5hkxgOl483RPX3f_SX -O language.pt
gdown --id 1wWrmopvvdhlBw-cL7EDyih9zn_IJu5Wr -O acoustic.pt
gdown --id 1LPchbScA_cuFx1XoNxpFCYZfGoJCfWao -O multimodal.pt
```

## Running
```bash
uv run main.py
```
By default it is available on port 3876 on /upload. On upload of a wav file it analyzes the audio.
It generates a report that looks something like this:
```json
{
  "disfluency": {
    "FP": [
      {
        "start_ms": 18900.0,
        "end_ms": 19120.0
      },
      {
        "start_ms": 19580.0,
        "end_ms": 19900.0
      }
    ],
    "RP": [],
    "RV": [
      {
        "start_ms": 6400.0,
        "end_ms": 6920.0
      }
    ],
    "RS": [],
    "PW": [
      {
        "start_ms": 7200.0,
        "end_ms": 7340.0
      }
    ]
  },
  "transcription": [
    {
      "text": "like",
      "start_ms": 0,
      "end_ms": 120,
      "confidence": 0.484
    },
    {
      "text": "a",
      "start_ms": 120,
      "end_ms": 240,
      "confidence": 0.978
    },
    {
      "text": "formula",
      "start_ms": 240,
      "end_ms": 580,
      "confidence": 0.998
    },
    {
      "text": "to",
      "start_ms": 580,
      "end_ms": 900,
      "confidence": 0.999
    },
...
```
Disfluencies correspond to:
- Filled pause (FP): And um I think one thing...
- Partial word (PW): H- how do you feel about that?
- Repetition (RP): well with my with my grandmother...
- Revision (RV): And uh we were I was fortunate...
- Restart (RS): If you how long do you want to stay?


