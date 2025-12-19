# Sistem Pengenalan Suara pada Game Navigasi 2D Sederhana

Sistem Pengenalan Suara berbasis **Singular Value Decomposition (SVD)** dan **Machine Learning**.

## Konsep Utama

### Singular Value Decomposition (SVD)

SVD mendekomposisi mel-spectrogram audio menjadi komponen-komponen penting:
```
S = U Σ V^T
```

**Aplikasi:**
- Ekstraksi Fitur Utama Audio
- Reduksi dimensi
- Menghilangkan noise & mempertahankan informasi penting

### Support Vector Machine (SVM)

- Klasifikasi Suara kedalam beberapa perintah yang tersedia (*up*, *down*, *left*, dan *right*)

## Struktur Project

```
├── data/                    
│   ├── up/
|   ├── down/
|   ├── left/
|   └── right/
├── features/
│   └── svd_features.py     
├── model/
│   └── train_model.py      
├── record/
│   └── rec_audio.py        
├── normalize_audio.py      
└── main.py                 
```

## Cara Menggunakan

### 1. Record Audio
Rekam data training untuk setiap perintah (minimal 10 samples per kelas):
```bash
python record/rec_audio.py
```
- Pilih kelas (up/down/left/right)
- Bicara saat recording dimulai
- Audio akan tersimpan secara otomatis di `data/[kelas]/`

### 2. Normalisasi Audio
Normalisasi semua audio ke volume konsisten (RMS ~0.07):
```bash
python normalize_audio.py
```
- Memproses semua audio yang perlu dinormalisasi

### 3. Training Model
Train SVM classifier dengan audio yang sudah dinormalisasi:
```bash
python model/train_model.py
```
- Ekstraksi fitur SVD + MFCC dari dataset
- Output: `model/voice_model.pkl`

### 4. Main Program
Menjalankan program utama:
```bash
python main.py
```
- Tekan ENTER untuk berbicara, kemudian sebutkan perintah yang ingin dilakukan
- Sistem akan melakukan navigasi berdasarkan perintah yang diberikan

## Alur Kerja Program

```
Recording → Normalisasi → Feature Extraction → Classification
    ↓            ↓              ↓                    ↓
1.5s audio   RMS ~0.07    SVD (30) + MFCC (13)   SVM Model
16kHz mono                    43 features        4 classes
```

## Detail

### Feature Extraction
1. **Mel-Spectrogram:** Audio → frequency domain (64 mel bands)
2. **SVD:** Ekstraksi 30 singular values terbesar
3. **MFCC:** Mean dari 13 koefisien untuk timbre karakteristik
4. **Total:** 43 fitur per audio sample

### Model
- **Classifier:** SVM dengan RBF kernel
- **Input:** 43-dimensional feature vector
- **Output:** Probabilitas 4 kelas + label prediksi

### Dataset
- **Training:** 20 sampel per kelas (up, down, left, right)
- **Format:** WAV, 16kHz mono, 1.5 detik
- **Preprocessing:** Normalisasi audio ke RMS ~0.07