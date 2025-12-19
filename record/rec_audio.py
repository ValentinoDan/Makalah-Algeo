import sounddevice as sd
from scipy.io.wavfile import write
import numpy as np
import os
from pathlib import Path

SAMPLE_RATE = 16000
DURATION = 1.5

def record_dataset(label, count=10):
    base_dir = Path(__file__).resolve().parent.parent / "data"
    label_dir = base_dir / label
    label_dir.mkdir(parents=True, exist_ok=True)

    for i in range(count):
        input(f"[{label}] Sample {i+1}/{count} - tekan ENTER untuk rekam...")
        print("Merekam...")

        audio = sd.rec(int(SAMPLE_RATE * DURATION), samplerate=SAMPLE_RATE, channels=1, dtype='float32')
        sd.wait()

        audio_int16 = np.int16(audio * 32767)
        filename = label_dir / f"{label}_{i+11}.wav"
        write(filename, SAMPLE_RATE, audio_int16)

        print(f"Disimpan ke {filename}\n")

if __name__ == "__main__":
    label = input("Label perintah (up, down, left, right): ").strip().lower()
    record_dataset(label, count=10)
