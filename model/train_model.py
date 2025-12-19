import os
import glob
import numpy as np
import pickle
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from features.svd_features import extract_svd_features

# Absolute path
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)

DATA_DIR = os.path.join(PROJECT_DIR, "data")
MODEL_PATH = os.path.join(PROJECT_DIR, "model", "voice_model.pkl")

# label
LABELS = sorted(["up", "down", "left", "right"])

def load_dataset():
    X = []
    y = []

    for label in LABELS:
        folder = os.path.join(DATA_DIR, label)
        files = glob.glob(os.path.join(folder, "*.wav"))

        print(f"Memproses {label}: {len(files)} files")

        for f in files:
            try:
                feat = extract_svd_features(f)
                X.append(feat)
                y.append(label)
            except Exception as e:
                print(f"  Error membaca {f}: {e}")

    return np.array(X), np.array(y)

def train():
    print("Loading dataset...")
    X, y = load_dataset()
    print(f"Dataset: {X.shape[0]} sampel, {X.shape[1]} features\n")

    print("Training SVM...")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    model = SVC(kernel='rbf', C=10, gamma='scale', probability=True)
    model.fit(X_scaled, y)
    
    model_data = {'classifier': model, 'scaler': scaler}

    os.makedirs("model", exist_ok=True)
    with open(MODEL_PATH, "wb") as f:
        pickle.dump(model_data, f)

    print("Model telah disimpan")

if __name__ == "__main__":
    train()
