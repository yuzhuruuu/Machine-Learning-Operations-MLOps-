# Submission 2: Sistem Machine Learning Prediksi Risiko Diabetes

Nama: yusriiii
Username dicoding: yusriiii

| | Deskripsi |
| ----------- | ----------- |
| Dataset | [Pima Indians Diabetes Database](https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database) — dataset klasik untuk klasifikasi biner risiko diabetes, berisi 768 data pasien wanita keturunan Pima Indian (usia ≥ 21 tahun) dengan 8 fitur klinis numerik (jumlah kehamilan, kadar glukosa, tekanan darah, ketebalan kulit, insulin, BMI, diabetes pedigree function, usia) dan 1 label target biner (`Outcome`). Data yang dipakai pada proyek ini adalah data asli (bukan sintetis), diunduh dari mirror publik dataset tersebut. |
| Masalah | Fasilitas kesehatan membutuhkan cara untuk mengidentifikasi pasien yang berisiko mengidap diabetes sedini mungkin berdasarkan data klinis dasar dari pemeriksaan rutin, sekaligus membutuhkan sistem yang dapat diakses secara *real-time* melalui API dan dapat dipantau (monitoring) kesehatannya secara berkelanjutan setelah di-deploy ke produksi. |
| Solusi machine learning | Membangun model klasifikasi biner (berisiko / tidak berisiko diabetes) menggunakan Neural Network, dilatih melalui *machine learning pipeline* end-to-end berbasis **TensorFlow Extended (TFX)** yang dijalankan dengan **Apache Beam** sebagai pipeline orchestrator. Pipeline dilengkapi komponen **Tuner** untuk pencarian hyperparameter otomatis. Model yang dihasilkan kemudian di-containerize dengan **Docker** dan di-deploy ke platform cloud **Railway** menggunakan **TensorFlow Serving**, serta dipantau secara *real-time* menggunakan **Prometheus**. |
| Metode pengolahan | Data diproses melalui komponen **Transform** TFX dengan `preprocessing_fn`: seluruh 8 fitur numerik dinormalisasi menggunakan **z-score scaling** (`tft.scale_to_z_score`) agar memiliki skala yang seragam. Data juga melewati validasi otomatis (StatisticsGen, SchemaGen, ExampleValidator) sebelum masuk ke tahap tuning dan training. |
| Arsitektur model | Model dibangun dengan **Keras Functional API**: 8 fitur numerik yang telah dinormalisasi masuk sebagai input dense, digabungkan (`concatenate`), lalu diproses melalui 2 hidden layer `Dense` (jumlah unit ditentukan otomatis oleh komponen Tuner) dengan aktivasi ReLU yang diselingi `Dropout`; output layer 1 unit dengan aktivasi *sigmoid*. Hyperparameter (`units_1`, `units_2`, `dropout_rate`, `learning_rate`) dicari secara otomatis oleh komponen **Tuner** (KerasTuner `RandomSearch`, 8 trial, objective `val_binary_accuracy`) sebelum training model final oleh Trainer. Hasil tuning terbaik pada run ini: `units_1=80, units_2=24, dropout_rate=0.3, learning_rate=0.01`. |
| Metrik evaluasi | Evaluasi dilakukan menggunakan **TensorFlow Model Analysis (TFMA)** pada komponen Evaluator, dengan metrik: **AUC**, **Precision**, **Recall**, **Binary Accuracy** (threshold kelulusan minimal 0.6, dengan *change threshold* agar model baru tidak boleh lebih buruk dari model sebelumnya), dan **Example Count**. Model hanya di-*push* oleh Pusher apabila dinyatakan *blessed* oleh Evaluator. |
| Performa model | Model dinyatakan **BLESSED**. Hasil evaluasi pada data eval (182 sampel): **AUC 0.8526**, **Binary Accuracy 0.7802**, **Precision 0.7885**, **Recall 0.5857**. Model final otomatis disimpan oleh Pusher ke `yusriiii-pipeline/serving_model/`. |
| Opsi deployment | Model di-deploy menggunakan **Docker** yang menjalankan **TensorFlow Serving**, di-hosting pada platform cloud **Railway** (alternatif dari Heroku). Docker image membaca variabel environment `$PORT` yang disediakan otomatis oleh Railway, serta mengaktifkan `--monitoring_config_file` bawaan TF Serving untuk mengekspos metrik dalam format Prometheus di endpoint `/monitoring/prometheus/metrics`. |
| Web app | `https://GANTI_DENGAN_DOMAIN_RAILWAY_ANDA/v1/models/diabetes-model/metadata` — *(ganti dengan tautan domain Railway asli setelah proses deployment selesai, lihat `DEPLOYMENT.md`)* |
| Monitoring | Monitoring dilakukan menggunakan **Prometheus**, yang melakukan *scrape* metrik secara berkala dari endpoint `/monitoring/prometheus/metrics` milik TF Serving (konfigurasi pada `monitoring/prometheus.yml`). Metrik yang dipantau antara lain jumlah request (`:tensorflow:serving:request_count`), latensi, dan status kesehatan model. Melalui dashboard Prometheus (`localhost:9090/targets`), status target model serving dapat dipantau apakah berstatus **UP** (sehat, model merespons dengan baik) atau **DOWN** (model bermasalah/tidak dapat diakses). Detail langkah menjalankan Prometheus ada pada `DEPLOYMENT.md`. |

## Saran Tambahan yang Diterapkan

- ✅ **Komponen Tuner** — hyperparameter tuning otomatis menggunakan KerasTuner `RandomSearch` (lihat `modules/tuner.py` dan bagian "Hasil Hyperparameter Tuning" pada notebook).
- ✅ **Clean code** — seluruh logika pipeline dipisah menjadi modul-modul kecil dan reusable (`transform.py`, `tuner.py`, `trainer.py`, `utils.py`, `components.py`), didokumentasikan dengan docstring, dan bebas duplikasi kode. Hasil pengujian **pylint: 10.00/10** (lihat `pylint_report.txt` dan screenshot `yusriiii-pylint.png`).

## Struktur Proyek

```
.
├── yusriiii-pipeline/               # Direktori seluruh artefak komponen TFX (dijalankan via Apache Beam)
│   └── serving_model/               # Model hasil Pusher (SavedModel)
├── data/
│   └── diabetes.csv                 # Dataset Pima Indians Diabetes (asli)
├── modules/                          # Clean code: seluruh modul pipeline
│   ├── transform.py                  # preprocessing_fn untuk komponen Transform
│   ├── tuner.py                      # tuner_fn untuk komponen Tuner
│   ├── trainer.py                    # run_fn untuk komponen Trainer
│   ├── utils.py                      # fungsi bersama (menghindari duplikasi kode)
│   └── components.py                 # perakit seluruh komponen pipeline (clean code)
├── pipeline.py                       # Skrip utama, menjalankan pipeline via BeamDagRunner
├── yusriiii_submission.ipynb         # Notebook dokumentasi (sudah dijalankan)
├── requirements.txt
├── pylint_report.txt                 # Hasil analisis pylint pada folder modules/
├── .pylintrc
├── Dockerfile                        # Deployment model serving (TF Serving) ke cloud
├── config/
│   └── prometheus.config             # Konfigurasi monitoring bawaan TF Serving
├── serving_model_deploy/              # Model final, siap di-COPY oleh Dockerfile
├── monitoring/                        # Semua kebutuhan menjalankan Prometheus
│   ├── Dockerfile
│   ├── prometheus.yml
│   └── prometheus.config
├── DEPLOYMENT.md                      # Panduan lengkap deploy ke Railway & menjalankan Prometheus
├── yusriiii-deployment.png            # (dilengkapi sendiri, lihat DEPLOYMENT.md)
└── yusriiii-monitoring.png            # (dilengkapi sendiri, lihat DEPLOYMENT.md)
```

## Cara Menjalankan Pipeline

```bash
python3 -m venv tfx-env
source tfx-env/bin/activate
pip install -r requirements.txt

# Menjalankan pipeline via Apache Beam (BeamDagRunner)
python pipeline.py

# atau menjalankan notebook dokumentasi lengkap
jupyter notebook yusriiii_submission.ipynb
```

## Cara Deploy ke Cloud & Menjalankan Monitoring

Lihat panduan lengkap langkah-demi-langkah pada berkas **`DEPLOYMENT.md`**.
