import os
import json
import wave
import pandas as pd
import numpy as np
from scipy.io import wavfile
from vosk import Model, KaldiRecognizer

# --- CONFIGURATION ---
MODEL_PATH = "models/vosk-model-small-en-us-0.15" 
DATA_DIR = "data"
OUTPUT_CSV = "data/dataset.csv"

# --- 1. LOAD MODEL ---
if not os.path.exists(MODEL_PATH):
    print(f"❌ CRITICAL ERROR: Model not found at {MODEL_PATH}")
    exit(1)

print("Loading Vosk Model... (This takes a few seconds)")
model = Model(MODEL_PATH)

def analyze_audio(file_path):
    try:
        # METHOD A: Use built-in Wave library (Standard, Safe)
        wf = wave.open(file_path, "rb")
    except Exception as e:
        print(f"   ⚠️ Error reading {file_path}: {e}")
        return None

    # Check format (Vosk needs Mono PCM)
    if wf.getnchannels() != 1 or wf.getsampwidth() != 2 or wf.getcomptype() != "NONE":
        print(f"   ⚠️ Skipping {file_path}: Must be WAV Mono PCM 16-bit")
        wf.close()
        return None

    rec = KaldiRecognizer(model, wf.getframerate())
    rec.SetWords(True)
    
    results = []
    while True:
        data = wf.readframes(4000)
        if len(data) == 0: break
        if rec.AcceptWaveform(data):
            part = json.loads(rec.Result())
            if 'result' in part: results.extend(part['result'])
    
    final_part = json.loads(rec.FinalResult())
    if 'result' in final_part: results.extend(final_part['result'])
    
    wf.close()
    
    if not results: return None

    # --- FEATURE EXTRACTION ---
    # 1. Pause Rate (Time spent silent / Total time)
    total_duration = results[-1]['end'] - results[0]['start']
    pause_time = 0.0
    
    # Calculate gaps between words
    for i in range(len(results) - 1):
        gap = results[i+1]['start'] - results[i]['end']
        if gap > 0.25: # Any gap > 0.25s counts as hesitation
            pause_time += gap
    
    pause_rate = pause_time / total_duration if total_duration > 0 else 0

    # 2. Vocabulary Richness (Type-Token Ratio)
    words = [w['word'] for w in results]
    unique_words = set(words)
    ttr = len(unique_words) / len(words) if words else 0
    
    return {
        "pause_rate": round(pause_rate, 3),
        "vocab_richness": round(ttr, 3),
        "word_count": len(words)
    }

# --- 2. PROCESSING LOOP ---
data_records = []
labels_map = {"healthy": 0, "mild": 1, "moderate": 2, "severe": 3}

print(f"\n📂 Scanning '{DATA_DIR}'...")

for folder_name, label_code in labels_map.items():
    folder_path = os.path.join(DATA_DIR, folder_name)
    if not os.path.exists(folder_path): continue
        
    files = [f for f in os.listdir(folder_path) if f.lower().endswith('.wav')]
    
    for filename in files:
        file_path = os.path.join(folder_path, filename)
        print(f"     -> Analyzing {filename}...")
        
        features = analyze_audio(file_path)
        if features:
            features['label'] = label_code
            features['filename'] = filename
            data_records.append(features)

# --- 3. SAVE RESULTS ---
if data_records:
    df = pd.DataFrame(data_records)
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"\n✅ SUCCESS! Processed {len(df)} files.")
else:
    print("\n❌ NO DATA EXTRACTED. Are your files .wav?")