# Panduan Deployment ke Cloud (Railway) & Monitoring (Prometheus)

Panduan ini dibuat khusus karena kamu belum punya akun Railway. Ikuti langkah-langkah di bawah **secara berurutan**. Semua berkas yang dibutuhkan (`Dockerfile`, `serving_model_deploy/`, `monitoring/`) sudah disiapkan di dalam folder submission ini.

---

## Bagian A — Deploy Model ke Railway (Kriteria Wajib #3)

### A.1 Buat Akun Railway
1. Buka [https://railway.app](https://railway.app).
2. Klik **Login**, lalu pilih **Login with GitHub** (paling mudah, tidak perlu isi form manual).
3. Jika kamu belum punya akun GitHub, buat dulu di [https://github.com/signup](https://github.com/signup).
4. Setelah login, Railway biasanya meminta verifikasi email/nomor telepon untuk mengaktifkan free trial/credit. Ikuti saja instruksinya.

### A.2 Push Folder Proyek ke GitHub
Railway men-deploy dari repository GitHub, jadi proyek ini perlu diunggah dulu ke GitHub:

```bash
# di dalam folder hasil ekstrak submission ini
git init
git add .
git commit -m "Initial commit - submission 2 ML pipeline"
# buat repo baru di github.com (misal: yusriiii-diabetes-mlops), lalu:
git remote add origin https://github.com/<username-github>/yusriiii-diabetes-mlops.git
git branch -M main
git push -u origin main
```

> **Catatan:** Folder `yusriiii-pipeline/` cukup besar karena berisi seluruh artefak TFX. Kamu boleh menambahkan folder tersebut ke `.gitignore` agar repo GitHub tetap ringan — yang benar-benar dibutuhkan Railway untuk deployment hanyalah: `Dockerfile`, `serving_model_deploy/`, dan `config/`.

### A.3 Deploy di Railway
1. Di dashboard Railway, klik **New Project** → **Deploy from GitHub repo**.
2. Pilih repository yang baru saja kamu push (`yusriiii-diabetes-mlops`).
3. Railway akan otomatis mendeteksi `Dockerfile` di root folder dan mulai proses build. Tunggu sampai status berubah menjadi **Active/Success** (biasanya 2-5 menit).
4. Buka tab **Settings** pada service tersebut → bagian **Networking** → klik **Generate Domain**. Railway akan memberi domain publik, contoh:
   `https://yusriiii-diabetes-mlops-production.up.railway.app`
5. Railway otomatis menyediakan environment variable `PORT` — Dockerfile pada proyek ini sudah didesain untuk membaca `$PORT` secara otomatis, jadi tidak perlu konfigurasi tambahan.

### A.4 Uji Coba Model Serving
Buka browser atau gunakan `curl` untuk memastikan model sudah bisa diakses:

```bash
curl https://<domain-railway-kamu>/v1/models/diabetes-model/metadata
```

Jika berhasil, akan muncul respons JSON berisi signature model. **Screenshot halaman/response ini** dan simpan dengan nama `yusriiii-deployment.png` — ini adalah bukti keberhasilan deployment yang wajib dilampirkan.

---

## Bagian B — Menjalankan Prometheus untuk Monitoring (Kriteria Wajib #4)

Endpoint metrik Prometheus dari TF Serving sudah otomatis aktif di:
`https://<domain-railway-kamu>/monitoring/prometheus/metrics`
(karena Dockerfile sudah mengaktifkan `--monitoring_config_file=/model_config/prometheus.config`).

### B.1 Update Target di `monitoring/prometheus.yml`
Edit file `monitoring/prometheus.yml`, ganti baris berikut:
```yaml
targets: ["GANTI_DENGAN_DOMAIN_RAILWAY_ANDA"]
```
menjadi domain Railway kamu (tanpa `https://`), contoh:
```yaml
targets: ["yusriiii-diabetes-mlops-production.up.railway.app:443"]
```

### B.2 Jalankan Prometheus Secara Lokal via Docker
Pastikan Docker Desktop sudah terinstall di komputer kamu ([https://www.docker.com/products/docker-desktop/](https://www.docker.com/products/docker-desktop/)), lalu jalankan:

```bash
cd monitoring
docker build -t yusriiii-prometheus .
docker run -p 9090:9090 yusriiii-prometheus
```

### B.3 Cek Dashboard Prometheus
1. Buka browser ke `http://localhost:9090/targets`.
2. Pastikan target `diabetes-model-serving` berstatus **UP** (berwarna hijau) — ini menandakan Prometheus berhasil melakukan scrape metrik dari model yang sudah di-deploy ke Railway.
3. Coba juga query sederhana di tab **Graph**, misalnya ketik `:tensorflow:serving:request_count` lalu klik **Execute**.
4. **Screenshot halaman ini** (targets berstatus UP, atau hasil query) dan simpan dengan nama `yusriiii-monitoring.png`.

---

## Ringkasan Berkas yang Perlu Kamu Lengkapi Sendiri

Karena proses deployment membutuhkan akun cloud milik kamu sendiri, ada 3 hal yang **tidak bisa disiapkan otomatis** dan perlu kamu lakukan manual mengikuti panduan di atas:

| Berkas | Cara Mendapatkan |
|---|---|
| `yusriiii-deployment.png` | Screenshot response `/v1/models/diabetes-model/metadata` setelah deploy berhasil (Bagian A.4) |
| `yusriiii-monitoring.png` | Screenshot dashboard Prometheus `localhost:9090/targets` (Bagian B.3) |
| Tautan web app di README.md | Ganti placeholder link di README.md dengan domain Railway asli kamu setelah Bagian A.3 |

Setelah 3 hal di atas selesai, letakkan kedua file screenshot di root folder submission, lalu submission siap di-ZIP ulang dan dikirim.
