import os
import numpy as np
import librosa
from tqdm import tqdm

DATA_DIR = "data"
OUTPUT_X = "features/X.npy"
OUTPUT_Y = "features/y.npy"

N_MELS = 64          # jumlah filter mel
K_COMPONENTS = 40    # lebih banyak komponen untuk fitur lebih kaya

def extract_svd_features(audio_path):
    """Load audio -> Mel spectrogram -> SVD + MFCC -> fitur."""
    y, sr = librosa.load(audio_path, sr=16000)

    # Mel-spectrogram + SVD
    S = librosa.feature.melspectrogram(
        y=y,
        sr=sr,
        n_mels=N_MELS
    )
    S_db = librosa.power_to_db(S, ref=np.max)
    U, s, Vt = np.linalg.svd(S_db, full_matrices=False)
    svd_features = s[:K_COMPONENTS]
    
    # MFCC features (tambahan untuk meningkatkan akurasi)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    mfcc_mean = np.mean(mfcc, axis=1)
    mfcc_std = np.std(mfcc, axis=1)
    
    # Zero Crossing Rate (karakteristik suara)
    zcr = np.mean(librosa.feature.zero_crossing_rate(y))
    
    # Spectral Centroid (brightness suara)
    spectral_centroid = np.mean(librosa.feature.spectral_centroid(y=y, sr=sr))
    
    # Gabungkan semua fitur
    feature = np.concatenate([
        svd_features,      # 40 features
        mfcc_mean,         # 13 features
        mfcc_std,          # 13 features
        [zcr],             # 1 feature
        [spectral_centroid] # 1 feature
    ])  # Total 68 features
    
    # Normalisasi fitur
    feature = feature / (np.linalg.norm(feature) + 1e-8)

    return feature

def load_dataset():
    X = []
    y = []
    labels = []

    # Ambil label: nama folder dalam /data/
    for label in sorted(os.listdir(DATA_DIR)):
        folder = os.path.join(DATA_DIR, label)
        if not os.path.isdir(folder):
            continue

        labels.append(label)

        print(f"\nProcessing label: {label}")
        for file in tqdm(os.listdir(folder)):
            if not file.endswith(".wav"):
                continue

            path = os.path.join(folder, file)
            features = extract_svd_features(path)

            X.append(features)
            y.append(label)

    return np.array(X), np.array(y), labels


if __name__ == "__main__":
    print("--- Extraksi Fitur SVD dari Dataset Audio ---")

    X, y, labels = load_dataset()

    # Simpan dataset
    np.save(OUTPUT_X, X)
    np.save(OUTPUT_Y, y)

    print("\nSelesai!")
    print(f"Fitur disimpan ke: {OUTPUT_X}")
    print(f"Label disimpan ke: {OUTPUT_Y}")
    print(f"Labels: {list(sorted(set(y)))}")
