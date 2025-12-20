# SentriSevil V2.0 - Sistem Deteksi Dini Stunting & Wasting Berbasis AI

**SentriSevil** adalah aplikasi desktop berbasis Python yang mengintegrasikan _Computer Vision_ dan _Sensor Hardware_ untuk melakukan pengukuran antropometri (Berat Badan & Panjang Badan) pada balita secara digital. Aplikasi ini secara otomatis mengklasifikasikan status gizi (Stunting/Wasting) menggunakan standar Z-Score (Permenkes No. 2 Tahun 2020).

## 📋 Fitur Utama

- **Pengukuran Panjang Badan (Non-Kontak):** Menggunakan kamera webcam dengan algoritma AI (_YOLOv8 + MediaPipe Pose_) untuk mendeteksi titik kerangka tubuh (skeleton) balita.
- **Pengukuran Berat Badan (Real-time):** Terintegrasi dengan sensor _Load Cell_ (via Arduino/Wemos) melalui koneksi Serial (USB).
- **Klasifikasi Otomatis:** Menghitung Z-Score (PB/U dan BB/PB) secara instan berdasarkan jenis kelamin dan umur.
- **Monitoring Sistem:** Memantau suhu laptop, penggunaan CPU, dan RAM untuk mencegah _overheating_ saat menjalankan AI.
- **Manajemen Data:** Menyimpan riwayat pengukuran ke dalam database (SQLite/CSV).

## 🛠️ Teknologi yang Digunakan

## 🔬 Detail Teknis & Pustaka (Library)

Aplikasi ini dibangun menggunakan **Python** (Kompatibel v3.10 & v3.11) dengan pustaka utama sebagai berikut:

| Kategori              | Library (Modul)         | Fungsi dalam Aplikasi                                                                                        |
| :-------------------- | :---------------------- | :----------------------------------------------------------------------------------------------------------- |
| **GUI (Tampilan)**    | `PyQt5`                 | Membangun antarmuka modern (Jendela, Tombol, Tabel) yang responsif dan user-friendly.                        |
| **Kamera & Citra**    | `opencv-python` (`cv2`) | Mengakses webcam, memproses frame video, dan menggambar visualisasi kerangka (_skeleton_) pada layar.        |
| **Kecerdasan Buatan** | `ultralytics` (YOLO)    | Memuat model Deep Learning (`best.pt`) untuk mendeteksi keberadaan balita dalam video secara akurat.         |
| **Pose Estimation**   | `mediapipe`             | Mendeteksi titik koordinat sendi tubuh (hidung, bahu, pinggul, lutut, kaki) untuk perhitungan panjang badan. |
| **Hardware Serial**   | `pyserial`              | Membuka komunikasi data dengan mikrokontroler (Arduino/Wemos) untuk membaca sensor berat badan.              |
| **Data Processing**   | `pandas`                | Membaca dan mengolah data tabel standar antropometri (file CSV) untuk penentuan Z-Score.                     |
| **System Monitor**    | `psutil`                | Memantau kesehatan sistem (Suhu CPU, RAM) secara _real-time_ untuk mencegah kendala teknis.                  |

---

### Perangkat Lunak (Software)

- **Bahasa:** Python 3.10 atau 3.11
- **GUI Framework:** PyQt5 (Modern UI)
- **Artificial Intelligence:**
  - `ultralytics` (YOLOv8) - Deteksi objek (Manusia).
  - `mediapipe` - Estimasi pose/skeleton presisi.
- **Computer Vision:** OpenCV (`cv2`)
- **Data Processing:** Pandas (Untuk data tabel antropometri).
- **Hardware Interface:** `pyserial` (Komunikasi data timbangan).

### Perangkat Keras (Hardware)

1.  **Laptop/PC/Raspberry Pi 5** (Min. RAM 4GB).
2.  **Webcam Eksternal** (Disarankan resolusi HD 720p/1080p dengan lensa _wide_ jika memungkinkan).
3.  **Modul Timbangan Digital:** Load Cell + HX711 + Mikrokontroler (Arduino/Wemos/ESP32) yang mengirim data via Serial USB.
4.  **Tiang Penyangga Kamera (Rig):** Kamera harus diletakkan pada ketinggian tetap (misal: 100 cm atau 135 cm) tegak lurus menghadap matras ukur.

---

## ⚙️ Cara Instalasi

1.  **Persiapan Python**
    Pastikan Python 3.10 atau 3.11 sudah terinstall. Cek dengan perintah:

    ```bash
    python --version
    ```

2.  **Install Library Dependensi**
    Buka terminal/CMD di folder proyek ini dan jalankan:

    ```bash
    pip install PyQt5 opencv-python ultralytics mediapipe pandas pyserial psutil
    ```

3.  **Persiapan Model AI**
    Pastikan file model YOLO (`best.pt` atau `best1.pt`) sudah berada di dalam folder `data/` atau sesuai dengan konfigurasi di `src/config.py`.

---

## 🚀 Cara Penggunaan Aplikasi

1.  **Jalankan Aplikasi**

    ```bash
    python main.py
    ```

    _(Atau `StuntingMain.py` untuk versi legacy)_

2.  **Setup Pengukuran (Menu Ukur Bayi)**

    - **Kamera:** Pilih kamera dari _dropdown_ lalu klik **"Buka"**. Pastikan video tampil.
    - **Aktifkan AI:** Klik **"Mulai AI"**. Garis kerangka (skeleton) akan muncul pada tubuh balita di layar.
    - **Timbangan:** Hubungkan alat timbang ke USB. Pilih _Port COM_ yang sesuai, lalu klik **"Hubungkan"**. Pastikan angka berat badan muncul.

3.  **Proses Pengukuran**

    - Baringkan balita di bawah kamera.
    - Pastikan seluruh tubuh (kepala hingga kaki) terlihat kamera.
    - Tunggu hingga deteksi skeleton stabil.
    - Lihat hasil **Panjang Badan (PB)** dan **Berat Badan (BB)** di layar.
    - Status Gizi (Normal/Stunting/Wasting) akan muncul otomatis.

4.  **Simpan Data**
    Klik tombol **"Simpan"** untuk menyimpan hasil pengukuran ke database.

---

## ⚠️ Kalibrasi (PENTING!)

Agar pengukuran panjang badan akurat, Anda **WAJIB** melakukan kalibrasi, terutama jika ketinggian kamera berubah.

1.  Buka menu **"⚙️ Kalibrasi"** di dalam aplikasi.
2.  **Jarak Kamera:** Masukkan tinggi fisik kamera dari lantai/matras (contoh: `135` cm).
3.  **Nilai Regresi (a & b):**
    - Nilai ini didapat dengan membandingkan hasil ukur aplikasi vs meteran manual.
    - Rumus: `Tinggi_Asli = a + (b * Tinggi_Pixel_Ratio)`.
    - Jika belum dikalibrasi, gunakan nilai default atau sesuaikan hingga hasil ukur mendekati penggaris manual.

---

## 🔧 Troubleshooting (Kendala Umum)

- **Aplikasi Berat / Laptop Panas:**
  Aplikasi ini menggunakan 2 model AI sekaligus. Jika laptop _lag_ atau _restart_ sendiri:

  - Gunakan mode "Hemat Daya" pada kode (sudah dioptimasi di `src/ui/dialogs.py`).
  - Pastikan ventilasi laptop tidak tertutup.
  - Gunakan resolusi kamera 640x480 jika spesifikasi laptop rendah.

- **Kamera Tidak Terbuka:**

  - Pastikan tidak ada aplikasi lain (Zoom/Meet) yang menggunakan kamera.
  - Coba cabut dan colok ulang USB kamera.

- **Status Gizi Tidak Muncul:**
  - Pastikan file CSV data antropometri (`PBU_Laki...csv`, dll) ada di folder `data/`.
  - Pastikan data Umur dan Jenis Kelamin bayi sudah terisi dengan benar.

---

**Copyright © 2025 SentriSevil Team**
