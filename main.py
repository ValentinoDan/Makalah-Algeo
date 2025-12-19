import sounddevice as sd
import numpy as np
import pickle
import time
import librosa
from scipy.io.wavfile import write
import os

# Directory
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(SCRIPT_DIR, "model", "voice_model.pkl")

SAMPLE_RATE = 16000
DURATION = 1.5   # durasi rekaman
N_MELS = 64
SVD_COMPONENTS = 40   # jumlah komponen 40


# Feature extraction 
def extract_svd_features_audio(audio):
    """HARUS sama dengan extract_svd_features di svd_features.py"""
    y = audio.flatten().astype(float)
    sr = SAMPLE_RATE

    # Mel-spectrogram + SVD
    S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=N_MELS)
    S_db = librosa.power_to_db(S, ref=np.max)
    U, s, Vt = np.linalg.svd(S_db, full_matrices=False)
    svd_features = s[:SVD_COMPONENTS]
    
    # MFCC features
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    mfcc_mean = np.mean(mfcc, axis=1)
    mfcc_std = np.std(mfcc, axis=1)
    
    # Zero Crossing Rate
    zcr = np.mean(librosa.feature.zero_crossing_rate(y))
    
    # Spectral Centroid
    spectral_centroid = np.mean(librosa.feature.spectral_centroid(y=y, sr=sr))
    
    # Combine semua fitur
    feature = np.concatenate([
        svd_features,
        mfcc_mean,
        mfcc_std,
        [zcr],
        [spectral_centroid]
    ])
    
    # Normalisasi fitur
    feature = feature / (np.linalg.norm(feature) + 1e-8)

    return feature

# Rekam audio (dataset)
def record_audio(save_path=None):
    print("\nBicara sekarang...")
    time.sleep(0.5)

    audio = sd.rec(int(SAMPLE_RATE * DURATION),
                   samplerate=SAMPLE_RATE,
                   channels=1,
                   dtype='float32',
                   device=1) 
    sd.wait()
    
    # Normalize audio ke RMS 0.07 
    audio = audio.flatten()
    rms = np.sqrt(np.mean(audio**2))
    if rms > 0.0001:
        target_rms = 0.07
        gain = target_rms / rms
        audio = audio * gain
        audio = np.clip(audio, -1.0, 1.0)
    
    # untuk menyimpan file (belum digunakan)
    if save_path:
        write(save_path, SAMPLE_RATE, (audio * 32767).astype(np.int16))

    return audio

# Load model
def load_model():
    if not os.path.exists(MODEL_PATH):
        print(f"Model tidak ditemukan di: {MODEL_PATH}")
        raise FileNotFoundError("Model belum ditrain. Jalankan train_model.py dulu!")

    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)

    return model

# Prediksi suara
def predict_voice(return_confidence=False):
    # Rekam audio
    audio = record_audio()

    # Ekstrak fitur
    features = extract_svd_features_audio(audio)

    # Sesuaiakan ukuran
    features = features.reshape(1, -1)

    # Load model 
    model_data = load_model()
    
    # Handle old and new model format
    if isinstance(model_data, dict):
        scaler = model_data['scaler']
        classifier = model_data['classifier']
        features = scaler.transform(features)
    else:
        # Old format: model is the classifier directly
        classifier = model_data

    # Prediksi
    pred = classifier.predict(features)[0]
    max_prob = 0.0
    
    if hasattr(classifier, 'predict_proba'):
        proba = classifier.predict_proba(features)[0]
        classes = classifier.classes_
        max_prob = np.max(proba)
        
        # Output probability
        print("\nProbabilitas:")
        for cls, prob in zip(classes, proba):
            marker = " <--" if cls == pred else ""
            print(f"  {cls}: {prob:.1%}{marker}")
    
    print(f"\n>> {pred.upper()}")
    
    if return_confidence:
        return pred, max_prob
    return pred

# Game sederhana
GRID_SIZE = 10

class GameSimulator:
    def __init__(self):
        self.x = 5
        self.y = 5
        self.last_move = ""

    def print_grid(self):
        # Header angka
        header = "      "
        for i in range(1, GRID_SIZE + 1):
            header += f" {i}  "
        print(header)
        
        # Garis
        print("    +" + "---+" * GRID_SIZE)
        
        for row in range(1, GRID_SIZE + 1):
            if row == GRID_SIZE:
                line = f" {row} |"
            else:
                line = f" {row}  |"
            for col in range(1, GRID_SIZE + 1):
                if col == self.x and row == self.y:
                    line += " O |"  # Player
                else:
                    line += "   |"
            print(line)
            print("    +" + "---+" * GRID_SIZE)
        
        print(f"\nPosisi: ({self.x}, {self.y})")
        if self.last_move:
            print(f"Move: {self.last_move}")

    def move(self, direction):
        if direction == "up":
            self.y = max(0, self.y - 1)
        elif direction == "down":
            self.y = min(GRID_SIZE - 1, self.y + 1)
        elif direction == "left":
            self.x = max(0, self.x - 1)
        elif direction == "right":
            self.x = min(GRID_SIZE - 1, self.x + 1)
        self.last_move = direction.upper()

# Main
if __name__ == "__main__":
    print("Game Navigasi 2D Sederhana")
    print("Tekan ENTER untuk rekam, ketik 'q' untuk keluar\n")
    
    robot = GameSimulator()
    robot.print_grid()
    
    while True:
        try:
            user_input = input("\n> Tekan Enter untuk lanjut, ketik 'q' untuk keluar: ")
            if user_input.lower() == 'q':
                print("Selesai!")
                break
            cmd = predict_voice()
            robot.move(cmd)
            robot.print_grid()
        except KeyboardInterrupt:
            print("\nSelesai!")
            break
        except Exception as e:
            print(f"Error: {e}")


