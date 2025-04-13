import os, sys
import warnings
import argparse
import logging
import numpy as np
import pandas as pd

import torch, torchaudio

warnings.filterwarnings("ignore")
from transformers import BertTokenizerFast, BertForTokenClassification, Wav2Vec2FeatureExtractor
import whisper_timestamped as whisper

from models import AcousticModel, MultimodalModel

labels = ['FP', 'RP', 'RV', 'RS', 'PW']
device='cuda'

model_whisper = whisper.load_model('demo_models/asr', device='cuda')
model_whisper.to(device)
def run_asr(audio, orgnl_sr):

    # Load audio file and resample to 16 kHz
    audio_rs = torchaudio.functional.resample(audio, orgnl_sr, 16000)[0, :]
    audio_rs.to(device)



    # Get Whisper output
    result = whisper.transcribe(model_whisper, audio_rs, language='en', beam_size=5, temperature=(0.0, 0.2, 0.4, 0.6, 0.8, 1.0))

    # Convert output dictionary to a dataframe
    words = []
    for segment in result['segments']:
        words += segment['words']
    text_df = pd.DataFrame(words)
    text_df['text'] = text_df['text'].str.lower()

    return text_df

tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')
model_bert = BertForTokenClassification.from_pretrained('bert-base-uncased', num_labels=5)
model_dict = torch.load('demo_models/language.pt', map_location='cuda')
model_dict.pop('bert.embeddings.position_ids')
model_bert.load_state_dict(model_dict)
model_bert.config.output_hidden_states = True
model_bert.to(device)

def run_language_based(audio, orgnl_sr, text_df):

    # Tokenize the text
    text = ' '.join(text_df['text'])
    tokens = tokenizer(text, return_tensors="pt")
    input_ids = tokens['input_ids'].to(device)

    # Initialize Bert model and load in pre-trained weights

    # Get Bert output at the word-level
    output = model_bert.forward(input_ids=input_ids)
    probs = torch.sigmoid(output.logits)
    preds = (probs > 0.5).int()[0][1:-1]
    emb = output.hidden_states[-1][0][1:-1]

    # Convert Bert word-level output to a dataframe with word timestamps
    pred_columns = [f"pred{i}" for i in range(preds.shape[1])]
    pred_df = pd.DataFrame(preds.cpu(), columns=pred_columns)
    emb_columns = [f"emb{i}" for i in range(emb.shape[1])]
    emb_df = pd.DataFrame(emb.detach().cpu(), columns=emb_columns)
    df = pd.concat([text_df, pred_df, emb_df], axis=1)

    # Convert dataframe to frame-level output
    frame_emb, frame_pred = convert_word_to_framelevel(audio, orgnl_sr, df.fillna(0))

    return frame_emb, frame_pred

def convert_word_to_framelevel(audio, orgnl_sr, df):

    # How long does the frame-level output need to be?
    df['end'] = df['end'] + 0.01
    end = audio.shape[-1] / orgnl_sr

    # Initialize lists for frame-level predictions and embeddings (every 10 ms)
    frame_time = np.arange(0, end, 0.01).tolist()
    num_labels = len(labels)
    frame_pred = [[0] * num_labels] * len(frame_time)
    frame_emb = [[0] * 768] * len(frame_time)

    # Loop through text to convert each word's predictions and embeddings to the frame-level (every 10 ms)
    for idx, row in df.iterrows():
        start_idx = round(row['start'] * 100)
        end_idx = round(row['end'] * 100)
        end_idx = min(end_idx, len(frame_time))
        frame_pred[start_idx:end_idx] = [[row['pred' + str(pidx)] for pidx in range(num_labels)]] * (end_idx - start_idx)
        frame_emb[start_idx:end_idx] = [[row['emb' + str(eidx)] for eidx in range(768)]] * (end_idx - start_idx)

    # Convert these frame-level predictions and embeddings from every 10 ms to every 20 ms (consistent with WavLM output)
    frame_emb = torch.Tensor(np.array(frame_emb)[::2])
    frame_pred = torch.Tensor(np.array(frame_pred)[::2])

    return frame_emb, frame_pred

model_acu = AcousticModel()
model_acu.load_state_dict(torch.load('demo_models/acoustic.pt', map_location='cuda'))
model_acu.to(device)
def run_acoustic_based(audio, orgnl_sr):

    # Load audio file and resample to 16 kHz
    audio_rs = torchaudio.functional.resample(audio, orgnl_sr, 16000)[0, :]
    feature_extractor = Wav2Vec2FeatureExtractor(feature_size=1,
                                                 sampling_rate=16000,
                                                 padding_value=0.0,
                                                 do_normalize=True,
                                                 return_attention_mask=False)
    audio_feats = feature_extractor(audio_rs, sampling_rate=16000).input_values[0]
    audio_feats = torch.Tensor(audio_feats).unsqueeze(0)
    audio_feats = audio_feats.to(device)

    # Initialize WavLM model and load in pre-trained weights

    # Get WavLM output
    emb, output = model_acu(audio_feats)
    probs = torch.sigmoid(output)
    preds = (probs > 0.5).int()[0]
    emb = emb[0]

    return emb, preds

model_multi = MultimodalModel()
model_multi.load_state_dict(torch.load('demo_models/multimodal.pt', map_location='cuda'))
model_multi.to(device)
def run_multimodal(language, acoustic):

    # Rounding differences may result in slightly different embedding sizes
    # Adjust so they're both the same size
    min_size = min(language.size(0), acoustic.size(0))
    language = language[:min_size].unsqueeze(0)
    acoustic = acoustic[:min_size].unsqueeze(0)

    language = language.to(device)
    acoustic = acoustic.to(device)

    # Initialize multimodal model and load in pre-trained weights

    # Get multimodal output
    output = model_multi(language, acoustic)
    probs = torch.sigmoid(output)
    preds = (probs > 0.5).int()[0]

    return preds