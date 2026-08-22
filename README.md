# Credit Default Scoring

Model prediksi **credit default** untuk proses approval kredit, dibangun dalam dua versi: **Logistic Regression** (baseline interpretable) dan **LightGBM** (boosting).

Dataset: Kaggle [Home Credit Default Risk](https://www.kaggle.com/competitions/home-credit-default-risk).

## Hasil

Metrik out-of-fold dari Stratified 5-Fold pada 307.511 nasabah (default rate ~8,1%):

| Model | AUC | Gini | KS | CV std AUC | Overfit gap |
|---|---|---|---|---|---|
| LightGBM | 0.783 | 0.567 | 0.426 | 0.003 | 0.100 |
| Logistic Regression | 0.743 | 0.486 | 0.363 | 0.004 | 0.000 |

LightGBM unggul pada AUC/KS; Logistic Regression memberi transparansi arah pengaruh tiap fitur. Std AUC antar fold yang kecil menunjukkan performa stabil, bukan kebetulan bagus di satu split. Metrik utama AUC-ROC.

## Struktur proyek

```
Home Credit Default Risk.ipynb   # Notebook: narasi, EDA, feature engineering, model, evaluasi, reason codes
credit_risk_model.py             # Python Script: pipeline end-to-end (train 2 model + evaluasi + SHAP + submission)
src/
  features.py                    # Feature engineering (application + bureau + previous + installments)
  modeling.py                    # LogReg + LightGBM, Stratified 5-Fold, metrik (AUC/Gini/KS/PR-AUC)
  reason_codes.py                # SHAP -> reason codes, opsional dipoles LLM via Ollama
outputs/
  metrics.json                   # Metrik hasil run
  submission.csv                 # Prediksi untuk application_test
  figures/                       # Grafik untuk PPT
requirements.txt
Home Credit Default Risk.pptx    # PPT
```

Prinsip: logika inti ada di `src/` dan dipakai bersama oleh notebook dan `credit_risk_model.py`, jadi hasil keduanya konsisten dan tidak ada kode yang diduplikasi.

## Cara Menggunakan

### 1. Siapkan environment

```powershell
python -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements.txt
```

### 2. Sediakan dataset

Dataset tidak disertakan dalam paket (besar, berlisensi Kaggle). Unduh dari halaman kompetisi di atas, lalu taruh semua CSV di folder `data/` pada root proyek:

```
data/
  application_train.csv
  application_test.csv
  bureau.csv
  previous_application.csv
  installments_payments.csv
  ...
```

Proyek ini hanya memakai `application_*`, `bureau`, `previous_application`, dan `installments_payments`.

### 3. Jalankan pipeline (Python Script)

```powershell
.venv\Scripts\python.exe credit_risk_model.py
```

Melatih 2 model, menghitung metrik, membuat SHAP + reason codes, dan menulis `outputs/metrics.json`, `outputs/submission.csv`, serta grafik di `outputs/figures/`. Berjalan ~7 menit (LightGBM ~1200 iterasi per fold, 2x 5-fold CV).

### 4. Jalankan notebook

Buka `Home Credit Default Risk.ipynb` di Jupyter/VS Code lalu Run All, atau eksekusi headless:

```powershell
.venv\Scripts\python.exe -m ipykernel install --user --name hci-venv
.venv\Scripts\python.exe -m nbconvert --to notebook --execute --inplace "Home Credit Default Risk.ipynb" --ExecutePreprocessor.kernel_name=hci-venv --ExecutePreprocessor.timeout=1200
```

## Deliverable

Tiga output utama: **Python Notebook** (`Home Credit Default Risk.ipynb`), **Python Script** (`credit_risk_model.py` + `src/`), dan **PPT 5 slide** (`Home Credit Default Risk.pptx`).

## Catatan Gen AI

Reason codes (alasan risiko per nasabah dalam Bahasa Indonesia) dihasilkan dari SHAP secara deterministik, jadi selalu bisa direproduksi. Opsional dipoles LLM open-source `qwen2.5:3b` via Ollama lokal (`http://localhost:11434`); kalau Ollama tidak aktif, otomatis fallback ke template. Jadi pipeline tetap jalan tanpa Ollama.
