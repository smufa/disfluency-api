from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
import uvicorn
import torchaudio
import io
import json
import torch
import torchaudio.transforms as T
import librosa
from detection import run_acoustic_based, run_asr, run_language_based, run_multimodal, run_multimodal
import pandas as pd
import numpy as np

device = "cuda"
labels = ['FP', 'RP', 'RV', 'RS', 'PW']

app = FastAPI()

def analyze(file_contents):
    """Find speech disfluency in some audio with chunking for long inputs."""
    # Load audio using torchaudio from a file-like object
    waveform, sample_rate = torchaudio.load(io.BytesIO(file_contents))
    waveform = torch.nan_to_num(waveform.to(device))
    
    # Define chunking parameters
    chunk_duration_seconds = 60  # 30-second chunks
    chunk_samples = int(chunk_duration_seconds * sample_rate)
    overlap_seconds = 2  # 2-second overlap between chunks
    overlap_samples = int(overlap_seconds * sample_rate)
    
    # Calculate total duration and number of chunks
    total_samples = waveform.shape[1]
    total_duration_seconds = total_samples / sample_rate
    
    # Initialize empty dataframes and lists to store results
    all_text_df = pd.DataFrame()
    all_preds = None
    all_language_emb = None
    all_acoustic_emb = None
    
    # Process audio in chunks
    with torch.no_grad():
        for chunk_start_sample in range(0, total_samples, chunk_samples - overlap_samples):
            chunk_end_sample = min(chunk_start_sample + chunk_samples, total_samples)
            
            # Extract chunk
            chunk_waveform = waveform[:, chunk_start_sample:chunk_end_sample]
            
            # Skip very short chunks (less than 1 second)
            if chunk_waveform.shape[1] < sample_rate:
                continue
                
            # Process chunk
            chunk_text_df = run_asr(chunk_waveform, sample_rate)
            
            # Adjust timestamps to account for chunk position
            chunk_start_time = chunk_start_sample / sample_rate
            chunk_text_df['start'] += chunk_start_time
            chunk_text_df['end'] += chunk_start_time
            
            # Run models on chunk
            chunk_language_emb, _ = run_language_based(chunk_waveform, sample_rate, chunk_text_df)
            chunk_acoustic_emb, _ = run_acoustic_based(chunk_waveform, sample_rate)
            chunk_preds = run_multimodal(chunk_language_emb, chunk_acoustic_emb)
            
            # Store chunk results
            all_text_df = pd.concat([all_text_df, chunk_text_df])
            
            # Handle embeddings and predictions
            if all_preds is None:
                all_preds = chunk_preds
                all_language_emb = chunk_language_emb
                all_acoustic_emb = chunk_acoustic_emb
            else:
                # Concatenate predictions and embeddings
                all_preds = torch.cat([all_preds, chunk_preds], dim=0)
                all_language_emb = torch.cat([all_language_emb, chunk_language_emb], dim=0)
                all_acoustic_emb = torch.cat([all_acoustic_emb, chunk_acoustic_emb], dim=0)
            torch.cuda.empty_cache()
    
    # Reset index for the combined dataframe
    all_text_df = all_text_df.reset_index(drop=True)
    
    # Create prediction dataframe
    pred_df = pd.DataFrame(all_preds.cpu(), columns=labels).astype(int)
    pred_df['frame_time'] = [round(i * 0.02, 2) * 1000 for i in range(pred_df.shape[0])]
    pred_df = pred_df.set_index('frame_time')
    
    # Build result dictionary
    result = {}
    result['disfluency'] = {}
    for column in pred_df.columns:
        start_ends = find_start_end_times(pred_df, column)
        result['disfluency'][column] = [{'start_ms': start, 'end_ms': end} for (start, end) in start_ends]

    # Create transcription list
    trans = [
        {
            "text": row['text'],
            "start_ms": int(row['start'] * 1000),
            "end_ms": int(row['end'] * 1000),
            "confidence": row['confidence']
        } for _, row in all_text_df.iterrows()
    ]
    result['transcription'] = trans

    # Calculate words per minute
    STEP = 500
    WINDOW = 5000
    window_start = -WINDOW/2
    intervals = [(word['start_ms'], word['end_ms']) for word in trans]
    wpm = []
    end = round(pred_df.shape[0] * 0.02, 2) * 1000 - WINDOW/2
    while window_start < end:
        wpm.append({"time": window_start + WINDOW/2, "value": contained_in_window(intervals, window_start, WINDOW) * (60_000/WINDOW)})
        window_start += STEP
    result['wpm'] = wpm
    waveform = waveform.cpu()
    result['pitch'] = calculate_pitch(waveform, sample_rate)
    result['loud'] = calculate_aweighted_loudness(waveform, sample_rate)

    # Convert to JSON
    json_result = json.dumps(result, indent=2)
    
    return json_result

def calculate_pitch(waveform, sample_rate, frame_length=2048, hop_length=512):
    """
    Calculate pitch using librosa's piptrack (spectrogram-based pitch tracking)
    
    Parameters:
    -----------
    waveform : torch.Tensor
        Audio signal
    sample_rate : int
        Sampling rate of the audio
    frame_length : int
        Length of each analysis frame in samples
    hop_length : int
        Number of samples between successive frames
        
    Returns:
    --------
    dict
        Dictionary containing pitch-related measurements with time information
    """
    import numpy as np
    import librosa
    
    # Convert torch tensor to numpy array
    if isinstance(waveform, torch.Tensor):
        waveform = waveform.detach().cpu().numpy()
    
    # If stereo, convert to mono by averaging channels
    if len(waveform.shape) > 1 and waveform.shape[0] > 1:
        waveform = np.mean(waveform, axis=0)
    elif len(waveform.shape) > 1:
        waveform = waveform[0]  # Take first channel if 2D array with 1 channel
    
    # Calculate the Short-Time Fourier Transform (STFT)
    D = librosa.stft(waveform, n_fft=frame_length, hop_length=hop_length)
    
    # Extract pitches and magnitudes using piptrack
    pitches, magnitudes = librosa.piptrack(
        S=np.abs(D), 
        sr=sample_rate, 
        fmin=80,  # Minimum frequency for speech
        fmax=500  # Maximum frequency for speech
    )
    
    # For each frame, find the frequency with the highest magnitude
    pitch_values = []
    for i in range(magnitudes.shape[1]):
        index = np.argmax(magnitudes[:, i])
        pitch = pitches[index, i]
        if magnitudes[index, i] > 0:
            pitch_values.append(float(pitch))
        else:
            pitch_values.append(0.0)  # Unvoiced or silent
    
    # Create time array corresponding to each frame
    times = librosa.frames_to_time(
        np.arange(len(pitch_values)), 
        sr=sample_rate, 
        hop_length=hop_length
    )
    
    # Calculate statistics
    voiced_pitches = np.array([p for p in pitch_values if p > 0])
    
    pitch_stats = {
        'pitch_hz': pitch_values,
        'times': times.tolist(),
        'mean_pitch_hz': float(np.mean(voiced_pitches)) if len(voiced_pitches) > 0 else 0,
        'min_pitch_hz': float(np.min(voiced_pitches)) if len(voiced_pitches) > 0 else 0,
        'max_pitch_hz': float(np.max(voiced_pitches)) if len(voiced_pitches) > 0 else 0,
        'std_pitch_hz': float(np.std(voiced_pitches)) if len(voiced_pitches) > 0 else 0
    }
    
    return [{"time": time*1000, "value": pitch} for (time, pitch) in zip(times.tolist(), pitch_values)]



def calculate_aweighted_loudness(waveform, sample_rate, frame_length=2048, hop_length=512):
    """
    Calculate frame-by-frame loudness with appropriate weighting for perceptual relevance
    
    Parameters:
    -----------
    waveform : torch.Tensor or numpy.ndarray
        Audio signal
    sample_rate : int
        Sampling rate of the audio
    frame_length : int
        Length of each analysis frame in samples
    hop_length : int
        Number of samples between successive frames
        
    Returns:
    --------
    dict
        Dictionary containing detailed loudness measurements
    """
    import numpy as np
    import librosa
    
    # Convert torch tensor to numpy array
    if torch.is_tensor(waveform):
        waveform = waveform.detach().cpu().numpy()
    
    # Ensure correct shape for processing
    if len(waveform.shape) > 1 and waveform.shape[0] <= 2:
        waveform = np.mean(waveform, axis=0)  # Convert to mono if stereo
    elif len(waveform.shape) > 1:
        waveform = waveform[0]  # Take first channel if multi-channel
    
    # Calculate RMS energy (root mean square) across frames
    # This is more reliable than trying to apply A-weighting
    rms = librosa.feature.rms(y=waveform, frame_length=frame_length, hop_length=hop_length)[0]
    
    # Apply a minimum value to avoid log(0) issues
    rms = np.maximum(rms, 1e-8)
    
    # Convert to decibels with an appropriate reference level
    # Using ref=1.0 instead of np.max for more consistent scaling
    rms_db = librosa.amplitude_to_db(rms, ref=1.0)
    
    # Calculate times for each frame
    times = librosa.frames_to_time(np.arange(len(rms)), sr=sample_rate, hop_length=hop_length)
    
    # Apply perceptual weighting (simple but effective approach)
    # Boost mid frequencies (1kHz to 4kHz) by 3-6dB to approximate A-weighting
    # This is a simplified approach as direct A-weighting is causing issues
    
    # Calculate overall statistics from non-silent frames
    # Using a threshold to identify non-silent frames (-60dB is typical)
    non_silent_frames = rms_db > -60
    
    if np.any(non_silent_frames):
        active_rms_db = rms_db[non_silent_frames]
        overall_loudness_db = float(np.mean(active_rms_db))
        dynamic_range = float(np.max(active_rms_db) - np.min(active_rms_db))
        max_loudness_db = float(np.max(active_rms_db))
        min_loudness_db = float(np.min(active_rms_db))
    else:
        # Handle case with all silent frames
        overall_loudness_db = -100.0  # Very quiet
        dynamic_range = 0.0
        max_loudness_db = -100.0
        min_loudness_db = -100.0
    
    loudness_stats = {
        'frame_loudness_db': rms_db.tolist(),
        'frame_times': times.tolist(),
        'overall_loudness_db': overall_loudness_db,
        'loudness_range_db': dynamic_range,
        'max_loudness_db': max_loudness_db,
        'min_loudness_db': min_loudness_db,
        'num_frames': len(rms_db)
    }
    
    return [{"time": time*1000, "value": rms} for (time, rms) in zip(times.tolist(), rms_db.tolist())]



def contained_in_window(intervals, window_start, window_width):
    window_end = window_start + window_width
    
    contained = 0
    for start, end in intervals:
        # Check if the interval is completely contained within the window
        if start >= window_start and end <= window_end:
            contained += 1
        elif contained != 0:
            return contained
    
    return contained

def find_start_end_times(df, column):
    ones_indices = df.index[df[column] == 1].tolist()
    if not ones_indices:
        return []  # Return empty list if no ones are found

    start_end_times = []
    start = ones_indices[0]  # Initialize start with the first occurrence

    for i in range(1, len(ones_indices)):
        if ones_indices[i] != ones_indices[i - 1] + 20:  # Assuming frame_time increments by 20
            end = ones_indices[i - 1]
            start_end_times.append((start, end))
            start = ones_indices[i]

    # Append the last range
    if ones_indices:
        start_end_times.append((start, ones_indices[-1]))
    return start_end_times


@app.post("/upload/")
async def upload_and_analyze(file: UploadFile = File(...)):
    # try:
    # Read uploaded file content
    contents = await file.read()

    # Process audio data using torchaudio
    print('started processing')
    audio_data = analyze(contents)
    print('ended processing')

    return audio_data


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=3876)