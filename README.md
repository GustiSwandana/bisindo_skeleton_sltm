# Aplikasi Pendeteksian Alfabet BISINDO

Aplikasi ini mengenali alfabet BISINDO dari gambar statis menggunakan pipeline hibrida **Skeleton + LSTM**.

## 1. Arsitektur Sistem

1. Frontend React menerima upload gambar tangan dan menampilkan preview.
2. Backend FastAPI menerima file gambar melalui endpoint `/predict`.
3. MediaPipe Hands mendeteksi 21 landmark tangan.
4. Landmark dinormalisasi terhadap wrist dan diskalakan agar robust terhadap posisi dan ukuran tangan.
5. Setiap landmark diperlakukan sebagai satu timestep pseudo-sequence berukuran 5 fitur: `x`, `y`, `z`, jarak ke wrist, dan jarak ke centroid.
6. Sequence `21 x 5` diproses oleh model Bidirectional LSTM untuk mengklasifikasikan alfabet.
7. API mengembalikan label prediksi, confidence, top-3 class, dan gambar skeleton overlay dalam JSON.

## 2. Kenapa LSTM Dipakai untuk Gambar Statis

LSTM biasanya dipakai untuk data berurutan waktu. Karena dataset Anda berupa gambar statis, pendekatan yang digunakan di proyek ini adalah **pseudo-sequence landmark**:

- Urutan 21 landmark MediaPipe dianggap sebagai sequence anatomis dari tangan.
- Setiap titik landmark menjadi satu timestep.
- LSTM belajar hubungan spasial antartitik secara berurutan, bukan hubungan temporal antar frame.

Pendekatan ini tetap memenuhi syarat hybrid skeleton + LSTM, walaupun secara praktis untuk citra statis model MLP, 1D CNN, atau GCN bisa menjadi alternatif yang lebih natural. Versi di sini tetap dibuat sesuai kebutuhan Anda.

## 3. Struktur Folder Project

```text
Bisindo_Project/
├── backend/
│   ├── artifacts/
│   ├── model/
│   │   ├── inference.py
│   │   ├── network.py
│   │   └── train.py
│   ├── utils/
│   │   ├── dataset.py
│   │   ├── hand_skeleton.py
│   │   ├── preprocessing.py
│   │   └── serialization.py
│   ├── app.py
│   ├── config.py
│   ├── predict.py
│   ├── requirements.txt
│   └── train.py
├── frontend/
│   ├── public/
│   ├── src/
│   │   ├── components/
│   │   │   └── PredictionCard.jsx
│   │   ├── App.jsx
│   │   ├── main.jsx
│   │   └── styles.css
│   ├── .env.example
│   ├── index.html
│   ├── package.json
│   └── vite.config.js
├── dataset/
│   └── README.md
├── dataset_bisindo/
│   ├── train/
│   └── val/
├── .gitignore
└── README.md
```

Catatan: kode backend otomatis membaca `dataset_bisindo/` karena folder data itu sudah ada di workspace Anda. Jika nanti Anda ingin menggunakan `dataset/`, tinggal pindahkan atau ubah argumen training.

## 4. Alur Preprocessing Dataset

1. Loader membaca gambar dari `train/images/<label>` dan `val/images/<label>`.
2. Untuk setiap gambar, MediaPipe Hands mengekstraksi 21 landmark tangan.
3. Landmark ditranslasi dengan acuan wrist (`landmark[0]`).
4. Semua titik dinormalisasi dengan skala jarak maksimum dari wrist.
5. Dari tiap landmark dibentuk fitur:
   - koordinat `x`, `y`, `z`
   - `distance_to_wrist`
   - `distance_to_centroid`
6. Hasil akhir per gambar adalah tensor `21 x 5`.
7. Label alfabet diubah menjadi indeks integer sesuai urutan nama folder kelas.
8. Sampel yang gagal terdeteksi tangan akan di-skip dan dicatat pada `backend/artifacts/training_history.json`.

## 5. Backend API

### Endpoint utama

- `GET /health`
- `GET /labels`
- `POST /predict`

### Contoh response prediksi

```json
{
  "success": true,
  "predicted_label": "A",
  "confidence": 0.94,
  "top_predictions": [
    { "label": "A", "confidence": 0.94 },
    { "label": "S", "confidence": 0.03 },
    { "label": "E", "confidence": 0.01 }
  ],
  "skeleton_image_base64": "..."
}
```

## 6. Cara Menjalankan Project

### Backend

```bash
cd backend
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
python train.py
uvicorn app:app --reload --host 0.0.0.0 --port 8000
```

Jika dataset Anda bukan di `dataset_bisindo/`, jalankan training dengan argumen:

```bash
python train.py --train-dir ..\dataset\train --val-dir ..\dataset\val
```

### Frontend

```bash
cd frontend
npm install
copy .env.example .env
npm run dev
```

Buka `http://localhost:5173`.

## 7. File Penting

- `backend/app.py`: API FastAPI untuk health check, daftar label, dan prediksi.
- `backend/model/train.py`: pipeline training, validasi, callback, dan penyimpanan model.
- `backend/model/inference.py`: loader model dan prediksi satu gambar.
- `backend/utils/hand_skeleton.py`: ekstraksi landmark dan visualisasi skeleton.
- `frontend/src/App.jsx`: halaman utama React.
- `frontend/src/components/PredictionCard.jsx`: tampilan hasil prediksi.

## 8. Saran Pengembangan Lanjutan

1. Tambahkan webcam live inference untuk prediksi real-time.
2. Simpan confusion matrix dan classification report setelah training.
3. Tambahkan augmentasi landmark atau image preprocessing sebelum ekstraksi skeleton.
4. Tambahkan fallback model MLP atau 1D CNN untuk membandingkan performa dengan LSTM.
5. Simpan model versioning dan endpoint `/metrics` untuk monitoring performa.
6. Tambahkan autentikasi dan penyimpanan riwayat prediksi bila aplikasi dipakai multi-user.

## 9. Catatan Pengembangan

- Model belum otomatis dilatih pada proyek ini. Anda tetap perlu menjalankan `python train.py` setelah dependency terpasang.
- Karena memakai MediaPipe, performa training sangat dipengaruhi kualitas framing tangan pada dataset.
- Jika beberapa kelas sulit dibedakan, pertimbangkan dataset video agar LSTM bisa memanfaatkan dinamika temporal yang sesungguhnya.
