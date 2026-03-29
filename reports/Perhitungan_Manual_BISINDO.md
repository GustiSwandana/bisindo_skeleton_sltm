# Perhitungan Manual Aplikasi Pendeteksian Alfabet BISINDO

Tanggal ekspor: 2026-03-27 12:23:31

## 1. Ringkasan Model dan Dataset
- Folder train: D:\Kerjaan\Client\Brians\Bisindo_Project\dataset_bisindo\train
- Folder validasi: D:\Kerjaan\Client\Brians\Bisindo_Project\dataset_bisindo\val
- Jumlah kelas alfabet: 26
- Sequence length model: 42
- Feature dimension per timestep: 8
- Total fitur per sampel: 42 x 8 = 336
- Sampel train terpakai: 8257
- Sampel train di-skip: 1328
- Total train awal: 9585
- Persentase skip train: 13.85%
- Sampel validasi terpakai: 2067
- Sampel validasi di-skip: 338
- Total validasi awal: 2405
- Persentase skip validasi: 14.05%
- Akurasi validasi akhir: 92.65%
- Loss validasi akhir: 0.272903


## 2. Uraian Metode dalam Bahasa Laporan
Penelitian ini mengimplementasikan sistem pengenalan alfabet Bahasa Isyarat BISINDO berbasis citra statis dengan pendekatan hibrida skeleton dan Long Short-Term Memory (LSTM). Tahap awal sistem adalah ekstraksi ciri tangan menggunakan MediaPipe Hands untuk memperoleh titik-titik landmark anatomi tangan. Setiap tangan direpresentasikan oleh 21 landmark tiga dimensi, sehingga ketika sistem mendeteksi dua tangan, total landmark yang digunakan menjadi 42 titik.
Landmark yang diperoleh tidak langsung digunakan sebagai masukan model. Tahap praproses dilakukan melalui translasi terhadap titik wrist, rotasi orientasi terhadap sumbu acuan wrist ke middle metacarpophalangeal joint, dan normalisasi skala berdasarkan geometri telapak tangan. Tahap ini bertujuan untuk mengurangi variasi yang disebabkan oleh perbedaan posisi tangan, kemiringan, serta jarak tangan terhadap kamera.
Setelah normalisasi, setiap landmark diubah menjadi vektor fitur berdimensi 8, yang terdiri atas koordinat ternormalisasi, vektor relatif terhadap parent joint, jarak terhadap wrist, serta jarak terhadap centroid. Dengan demikian, satu sampel citra direpresentasikan sebagai matriks berukuran 42 x 8. Representasi ini selanjutnya diperlakukan sebagai pseudo-sequence agar hubungan antarlandmark dapat dipelajari secara berurutan oleh model LSTM meskipun data yang digunakan berupa gambar statis, bukan video.
Pemilihan arsitektur LSTM pada penelitian ini didasarkan pada kemampuannya dalam mempelajari dependensi antar elemen urutan. Pada implementasinya, model menggunakan Bidirectional LSTM sehingga proses pembelajaran dilakukan dari arah awal ke akhir dan dari arah akhir ke awal sequence. Pendekatan ini memungkinkan model menangkap keterkaitan spasial antarlandmark dengan konteks yang lebih lengkap dibandingkan LSTM satu arah.
Keluaran dari lapisan Bidirectional LSTM kemudian diteruskan ke lapisan dense untuk melakukan pemetaan fitur menuju 26 kelas alfabet BISINDO. Lapisan output menggunakan fungsi aktivasi softmax untuk menghasilkan probabilitas setiap kelas. Kelas dengan probabilitas tertinggi ditetapkan sebagai hasil prediksi akhir sistem.
Berdasarkan hasil pelatihan terakhir, model memperoleh akurasi validasi sebesar 92.65%. Selain itu, evaluasi multi-kelas menunjukkan nilai macro F1-score sebesar 0.9247 dan weighted F1-score sebesar 0.9264. Nilai tersebut menunjukkan bahwa model tidak hanya memiliki ketepatan prediksi yang tinggi secara umum, tetapi juga cukup seimbang dalam mengenali berbagai kelas alfabet BISINDO.
Secara keseluruhan, alur kerja sistem terdiri atas empat tahap utama, yaitu akuisisi citra, ekstraksi skeleton tangan, pembentukan pseudo-sequence fitur, dan klasifikasi dengan model Bidirectional LSTM. Rangkaian proses ini dirancang agar sistem mampu mengenali alfabet BISINDO secara otomatis baik dari citra unggahan maupun frame webcam secara real-time.

## 3. Perhitungan Manual Preprocessing Skeleton
Pada aplikasi ini, setiap gambar dapat dibaca sampai 2 tangan. Tiap tangan memiliki 21 landmark MediaPipe, sehingga sequence length = 2 x 21 = 42 timestep.

### 3.1 Translasi terhadap wrist
Rumus: translated = landmark - wrist
- Wrist = (0.5, 0.6, 0.0)
- Index MCP = (0.45, 0.4, -0.01)
- Hasil translasi Index MCP = (-0.04999999999999999, -0.19999999999999996, -0.01)
- Middle MCP = (0.55, 0.35, -0.02)
- Hasil translasi Middle MCP = (0.050000000000000044, -0.25, -0.02)

### 3.2 Rotasi orientasi tangan
Aplikasi memutar sumbu x-y agar arah wrist ke middle MCP menjadi tegak. Ini membuat model lebih stabil terhadap tangan yang miring.
- Sudut awal = atan2(-0.2500, 0.0500) = -1.373401 rad
- Target sudut = -pi/2 = -1.570796 rad
- Rotasi yang dipakai = target - sudut awal = -0.197396 rad
- cos(theta) = 0.980581
- sin(theta) = -0.196116
- Index MCP setelah rotasi = (-0.088252, -0.186310, -0.010000)
- Middle MCP setelah rotasi = (0.000000, -0.254951, -0.020000)

### 3.3 Penentuan skala normalisasi
Aplikasi memilih skala terbesar dari tinggi telapak, lebar telapak, dan radius maksimum.
- Palm height = ||middle_mcp|| = 0.255734
- Palm width = ||index_mcp - pinky_mcp|| = 0.251992
- Max radius = 0.263249
- Skala akhir = max(palm_height, palm_width, max_radius) = 0.263249

### 3.4 Normalisasi satu landmark contoh
Rumus: normalized = translated_rotated / scale
- Normalized Index MCP = (-0.335243, -0.707734, -0.037987)
- Normalized Middle MCP = (0.000000, -0.968479, -0.075974)

### 3.5 Penyusunan feature vector per landmark
Untuk model terbaru, satu landmark menghasilkan 8 fitur:
- normalized_x
- normalized_y
- normalized_z
- parent_vector_x
- parent_vector_y
- parent_vector_z
- distance_to_wrist
- distance_to_centroid

Contoh perhitungan untuk landmark Index MCP (parent = wrist):
- Parent vector = normalized_index - wrist_normalized = (-0.335243, -0.707734, -0.037987)
- Centroid contoh (3 titik ilustrasi) = (0.094365, -0.819482, -0.063311)
- Distance to wrist = 0.784040
- Distance to centroid = 0.444625

Maka feature vector ilustratif untuk Index MCP adalah:
[-0.335243, -0.707734, -0.037987, -0.335243, -0.707734, -0.037987, 0.784040, 0.444625]

## 4. Perhitungan Ukuran Input Model
- Maksimum tangan terbaca = 2
- Landmark per tangan = 21
- Sequence length = 2 x 21 = 42
- Feature dim = 8
- Total nilai input per sampel = 42 x 8 = 336

## 5. Perhitungan Manual Parameter LSTM
Rumus parameter satu layer LSTM: 4 x ((input_dim + units) x units + units)

### 5.1 Bidirectional LSTM pertama
- input_dim = 8
- units = 128
- Parameter satu arah = 4 x ((8 + 128) x 128 + 128) = 70144
- Karena bidirectional, total = 2 x 70144 = 140288

### 5.2 Bidirectional LSTM kedua
- input_dim = 2 x 128 = 256
- units = 64
- Parameter satu arah = 4 x ((256 + 64) x 64 + 64) = 82176
- Karena bidirectional, total = 2 x 82176 = 164352

### 5.3 Dense layer
- Dense(64): (128 x 64) + 64 = 8256
- Dense output(26): (64 x 26) + 26 = 1690

### 5.4 Total parameter model
- Total = 140288 + 164352 + 8256 + 1690 = 314586

## 6. Perhitungan Prediksi Softmax
Output model adalah 26 nilai logit yang diubah menjadi probabilitas dengan softmax:
softmax(z_i) = exp(z_i) / sum(exp(z_j))

Contoh sederhana jika tiga logit terbesar adalah [2.80, 1.20, 0.50]:
- exp(2.80) = 16.4446
- exp(1.20) = 3.3201
- exp(0.50) = 1.6487
- Total = 21.4134
- Probabilitas kelas-1 = 16.4446 / 21.4134 = 0.7680 = 76.80%

## 7. Confusion Matrix
- Total sampel validasi pada confusion matrix: 2067
- Prediksi benar pada diagonal utama: 1915
- Gambar confusion matrix: D:\Kerjaan\Client\Brians\Bisindo_Project\backend\artifacts\confusion_matrix.png
- Baris menunjukkan label sebenarnya, kolom menunjukkan label hasil prediksi.
- Nilai diagonal yang tinggi menandakan model mampu mengklasifikasikan kelas tersebut dengan benar.

## 8. Classification Report per Kelas
| Kelas | Precision | Recall | F1-score | Support |
|---|---:|---:|---:|---:|
| A | 0.8814 | 0.8667 | 0.8739 | 60 |
| B | 0.9333 | 0.8974 | 0.9150 | 78 |
| C | 0.8875 | 0.9103 | 0.8987 | 78 |
| D | 0.9506 | 0.9059 | 0.9277 | 85 |
| E | 0.9383 | 0.9744 | 0.9560 | 78 |
| F | 0.9189 | 0.8718 | 0.8947 | 78 |
| G | 0.9351 | 0.9600 | 0.9474 | 75 |
| H | 0.9296 | 0.8919 | 0.9103 | 74 |
| I | 0.9889 | 1.0000 | 0.9944 | 89 |
| J | 0.9474 | 1.0000 | 0.9730 | 90 |
| K | 0.9841 | 0.9538 | 0.9688 | 65 |
| L | 0.9022 | 0.9432 | 0.9222 | 88 |
| M | 0.9870 | 0.8172 | 0.8941 | 93 |
| N | 0.8411 | 0.9783 | 0.9045 | 92 |
| O | 1.0000 | 0.9610 | 0.9801 | 77 |
| P | 0.8627 | 0.9462 | 0.9026 | 93 |
| Q | 0.9706 | 0.9041 | 0.9362 | 73 |
| R | 0.9529 | 0.9529 | 0.9529 | 85 |
| S | 0.9342 | 0.8987 | 0.9161 | 79 |
| T | 0.8642 | 0.9091 | 0.8861 | 77 |
| U | 0.9487 | 0.9136 | 0.9308 | 81 |
| V | 0.9792 | 1.0000 | 0.9895 | 94 |
| W | 0.8986 | 0.8611 | 0.8794 | 72 |
| X | 0.8846 | 0.8961 | 0.8903 | 77 |
| Y | 0.8500 | 0.8500 | 0.8500 | 60 |
| Z | 0.9359 | 0.9605 | 0.9481 | 76 |
| accuracy | - | - | 0.9265 | 2067 |
| macro avg | 0.9272 | 0.9240 | 0.9247 | 2067 |
| weighted avg | 0.9282 | 0.9265 | 0.9264 | 2067 |

## 9. Kesimpulan untuk Laporan
Aplikasi membaca maksimum 2 tangan, menyusun sequence sepanjang 42 timestep, dan menghasilkan 8 fitur per timestep sehingga total input per sampel adalah 336 nilai.
Dari metadata training terbaru, model mencapai akurasi validasi 92.65% dengan total parameter jaringan sekitar 314,586 parameter trainable.
Perhitungan manual ini dapat langsung dimasukkan ke bab metode, implementasi, dan analisis hasil pada laporan skripsi atau proyek.