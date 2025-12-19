import os
import numpy as np
import librosa
from scipy.io.wavfile import write

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR, "data")
TARGET_RMS = 0.07

def normalize_audio(audio_path, target_rms=TARGET_RMS):
    """Normalize audio ke target RMS"""
    y, sr = librosa.load(audio_path, sr=16000)
    
    current_rms = np.sqrt(np.mean(y**2))
    
    if current_rms < 0.001:
        print("Audio terlalu lemah, skip")
        return False
    
    gain = target_rms / current_rms
    y_normalized = y * gain
    y_normalized = np.clip(y_normalized, -1.0, 1.0)
    
    # Save
    write(audio_path, sr, (y_normalized * 32767).astype(np.int16))
    
    new_rms = np.sqrt(np.mean(y_normalized**2))
    print(f"✓ {os.path.basename(audio_path)}: {current_rms:.4f} -> {new_rms:.4f}")
    
    return True

def normalize_dataset():
    print("--- NORMALISASI DATASET AUDIO ---")
    print(f"Target RMS: {TARGET_RMS}\n")
    
    labels = ['down', 'left', 'right', 'up']
    
    for label in labels:
        print(f"{'-'*50}")
        print(f"{label.upper()}")
        print(f"{'-'*50}")
        
        folder = os.path.join(DATA_DIR, label)
        
        files = [f for f in os.listdir(folder) if f.endswith('.wav')]
        
        for file in files:
            original_path = os.path.join(folder, file)
            
            normalize_audio(original_path, TARGET_RMS)
        
        print(f"JUMLAH FILE : {len(files)} files\n")
    
    print("-"*50)
    print("SELESAI! Train model:")
    print("   python model/train_model.py")
    print("-"*50)


if __name__ == "__main__":
    print("--- NORMALISASI AUDIO ---\n")
    response = input("Normalisasi semua audio? (ya/tidak): ").lower()
    
    if response in ['ya', 'yes', 'y']:
        normalize_dataset()
    else:
        print("DIBATALKAN")
