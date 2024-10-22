# Receipt Scanner Application

Aplikasi **Receipt Scanner** adalah aplikasi web berbasis Flask yang memungkinkan pengguna untuk memindai atau mengunggah nota, mengekstrak total jumlah, dan mengisi detail tambahan sebelum mengirimkan data tersebut ke Google Sheet dan API eksternal. Aplikasi ini dirancang untuk memudahkan proses pengumpulan dan pencatatan data nota secara efisien.

## Daftar Isi

- [Fitur](#fitur)
- [Persyaratan Sistem](#persyaratan-sistem)
- [Instalasi](#instalasi)
- [Konfigurasi](#konfigurasi)
- [Menjalankan Aplikasi](#menjalankan-aplikasi)
- [Struktur Proyek](#struktur-proyek)
- [Penggunaan](#penggunaan)
- [Catatan Penting](#catatan-penting)
- [Lisensi](#lisensi)

## Fitur

- **Pindai atau Unggah Nota**: Pengguna dapat menggunakan kamera perangkat untuk memindai nota atau mengunggah gambar nota dari perangkat mereka.
- **Ekstraksi Total Jumlah**: Aplikasi menggunakan model YOLOv8 dan Google Cloud Vision OCR untuk mengekstrak total jumlah dari gambar nota.
- **Formulir Multi-Langkah**: Proses input data dibagi menjadi tiga langkah untuk memandu pengguna melalui proses dengan lebih efisien.
- **Validasi dan Umpan Balik**: Validasi input dilakukan di sisi klien dan server, dengan pesan kesalahan yang jelas untuk membantu pengguna.
- **Integrasi Google Sheets**: Data yang dikumpulkan dikirim dan disimpan ke Google Sheet yang telah ditentukan.
- **API Eksternal**: Aplikasi terhubung dengan API eksternal untuk mengunggah file nota dan bukti lainnya.
- **Antarmuka Responsif**: Desain antarmuka menggunakan Tailwind CSS dan dioptimalkan untuk berbagai ukuran layar.

## Persyaratan Sistem

- Python 3.11 atau lebih baru
- Paket dan library Python yang diperlukan (lihat bagian Instalasi)
- Akun Google Cloud dengan API Vision diaktifkan
- Kredensial layanan (Service Account) untuk Google Cloud dan Google Sheets
- Model YOLOv8 yang telah dilatih
- Koneksi internet untuk mengakses API eksternal dan layanan Google

## Instalasi

Ikuti langkah-langkah berikut untuk menginstal dan menjalankan aplikasi:

1. **Clone Repository**

   ```bash
   git clone https://github.com/username/receipt-scanner.git
   cd receipt-scanner
   ```

2. **Buat Virtual Environment**

   Disarankan untuk menggunakan virtual environment untuk mengelola dependensi.

   ```bash
   python3 -m venv venv
   source venv/bin/activate  # Untuk Linux/Mac
   venv\Scripts\activate  # Untuk Windows
   ```

3. **Instal Dependensi**

   Instal semua paket yang diperlukan menggunakan `pip`.

   ```bash
   pip install -r requirements.txt
   ```

   **Catatan**: Pastikan file `requirements.txt` berisi semua paket yang diperlukan seperti Flask, ultralytics, google-cloud-vision, gspread, oauth2client, dll.

4. **Pasang Model YOLOv8**

   Pastikan Anda memiliki model YOLOv8 yang telah dilatih dan simpan di path yang akan digunakan dalam konfigurasi.

## Konfigurasi

1. **Buat File `.env`**

   Buat file `.env` di direktori proyek dan isi dengan variabel lingkungan yang diperlukan:

   ```env
   RECEIPT_API_ENDPOINT=your_receipt_api_endpoint
   EVIDENCE_API_ENDPOINT=your_evidence_api_endpoint
   YOLO_MODEL_PATH=path/to/your/yolo_model.pt
   GOOGLE_CREDENTIALS_PATH=path/to/your/google_credentials.json
   GOOGLE_SHEET_ID=your_google_sheet_id
   FLASK_SECRET_KEY=your_flask_secret_key
   ```

   **Penjelasan Variabel:**

   - `RECEIPT_API_ENDPOINT`: URL endpoint untuk API penerimaan nota.
   - `EVIDENCE_API_ENDPOINT`: URL endpoint untuk API bukti lainnya.
   - `YOLO_MODEL_PATH`: Path ke file model YOLOv8 yang telah dilatih.
   - `GOOGLE_CREDENTIALS_PATH`: Path ke file kredensial layanan Google (JSON).
   - `GOOGLE_SHEET_ID`: ID dari Google Sheet tempat data akan disimpan.
   - `FLASK_SECRET_KEY`: Kunci rahasia Flask untuk sesi dan keamanan.

2. **Atur Kredensial Google**

   - Pastikan API Vision dan Google Sheets API diaktifkan di proyek Google Cloud Anda.
   - Unduh file kredensial layanan (JSON) dan simpan di path yang ditentukan dalam `GOOGLE_CREDENTIALS_PATH`.

3. **Atur Izin pada Google Sheet**

   Berikan akses edit ke akun layanan Anda pada Google Sheet yang akan digunakan.

## Menjalankan Aplikasi

Setelah instalasi dan konfigurasi selesai, jalankan aplikasi dengan perintah berikut:

```bash
python app.py
```

Aplikasi akan berjalan pada `http://0.0.0.0:5151` secara default. Buka browser dan akses aplikasi melalui URL tersebut.

## Struktur Proyek

```
receipt-scanner/
├── app.py
├── index.html
├── requirements.txt
├── .env
├── uploads/
│   └── (folder untuk menyimpan file yang diunggah sementara)
└── README.md
```

- `app.py`: File utama aplikasi Flask.
- `index.html`: Template HTML utama untuk antarmuka pengguna.
- `requirements.txt`: Daftar paket Python yang diperlukan.
- `.env`: File konfigurasi lingkungan.
- `uploads/`: Direktori untuk menyimpan file yang diunggah sementara.

## Penggunaan

Berikut adalah panduan langkah demi langkah untuk menggunakan aplikasi:

### Langkah 1: Pindai atau Unggah Nota dan Konfirmasi Jumlah

- **Pindai Nota**: Klik tombol **"Scan Receipt"** untuk menggunakan kamera perangkat Anda dan memindai nota.
- **Unggah Nota**: Klik tombol **"Upload Receipt"** untuk mengunggah gambar nota dari perangkat Anda.
- **Konfirmasi Jumlah**:
  - Setelah gambar diproses, aplikasi akan mengekstrak total jumlah dari nota.
  - Jumlah yang diekstrak akan ditampilkan di bidang **"Amount"**.
  - Periksa dan edit jumlah tersebut jika diperlukan.
- **Mata Uang**: Pilih mata uang yang sesuai dari menu dropdown **"Currency"**.
- **Lanjut ke Langkah Berikutnya**: Klik tombol **"Next"** untuk melanjutkan.

### Langkah 2: Isi Detail Tambahan

- **Unggah Bukti Tambahan (Opsional)**:
  - Anda dapat mengunggah gambar bukti tambahan dengan mengklik **"Upload Evidence Files"**.
- **Pilih Account SKKO**:
  - Pilih Account SKKO yang sesuai dari menu dropdown **"Account SKKO"**.
- **Pilih ID Rencana**:
  - Pilih ID Rencana dari menu dropdown **"ID Rencana"**.
  - Setelah memilih, detail rencana akan ditampilkan dalam modal untuk konfirmasi.
- **Isi Uraian dan Judul Laporan**:
  - **Uraian**: Masukkan deskripsi transaksi.
  - **Judul Laporan**: Masukkan judul laporan yang relevan.
- **Lanjut ke Langkah Berikutnya**: Klik tombol **"Next"** untuk melanjutkan.

### Langkah 3: Tinjau dan Kirim

- **Tinjau Informasi**:
  - Periksa semua informasi yang telah Anda masukkan.
- **Kembali jika Perlu**:
  - Jika ada kesalahan, Anda dapat kembali ke langkah sebelumnya dengan klik tombol **"Back"**.
- **Kirim Data**:
  - Jika semua informasi sudah benar, klik tombol **"Submit"** untuk mengirim data.
- **Konfirmasi Pengiriman**:
  - Setelah pengiriman berhasil, Anda akan menerima pesan konfirmasi.

## Catatan Penting

- **Validasi Input**:
  - Pastikan semua bidang yang ditandai dengan tanda bintang (*) diisi dengan benar.
  - Pesan kesalahan akan ditampilkan jika ada input yang tidak valid.
- **Keamanan Data**:
  - Jangan bagikan informasi sensitif atau kredensial API dalam file konfigurasi atau kode sumber.
- **Batas Ukuran File**:
  - Ukuran maksimum file yang diunggah adalah 100MB.
  - Pastikan file yang diunggah berformat gambar yang valid.
- **Koneksi Internet**:
  - Aplikasi memerlukan koneksi internet yang stabil untuk mengakses API eksternal dan layanan Google.

## Lisensi

Aplikasi ini dilisensikan di bawah [MIT License](LICENSE). Anda bebas untuk menggunakan, memodifikasi, dan mendistribusikan aplikasi ini sesuai dengan ketentuan lisensi.

---

Jika Anda memiliki pertanyaan atau membutuhkan bantuan lebih lanjut, silakan hubungi pengembang melalui [email@example.com](mailto:email@example.com).