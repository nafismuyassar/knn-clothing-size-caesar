# Prediksi Ukuran Pakaian — KNN + CAESAR

Aplikasi web prediksi ukuran pakaian berdasarkan data antropometri tubuh menggunakan algoritma **K-Nearest Neighbor (KNN)** dengan pembobotan fitur lingkar dada, dilatih pada dataset **CAESAR**.

> Skripsi: *Prediksi Ukuran Pakaian Menggunakan Algoritma K-Nearest Neighbor dengan Pembobotan Fitur Lingkar Dada pada Dataset CAESAR*  
> Nafis Muyassar · NIM 20.11.4082 · Program Studi Informatika · Universitas Amikom Yogyakarta · 2024

---

## Fitur Aplikasi

- Input data tubuh: jenis kelamin, tinggi, berat, lingkar dada/bust, pinggang, pinggul
- Prediksi ukuran pakaian (XS–XXL) berdasarkan standar **EN 13402**
- Distribusi kepercayaan per ukuran
- Tabel rentang lingkar dada per ukuran per gender

## Performa Model

| Metrik | Nilai |
|--------|-------|
| Hold-out Accuracy | **95.26%** |
| Hold-out F1-macro | **94.79%** |
| Akurasi ±1 ukuran | **100%** |
| Nested CV Accuracy | 95.48% ± 0.87% |
| Nested CV F1-macro | 94.84% ± 1.04% |

Hyperparameter final: `k=15`, bobot chest `×3.0`, SMOTE=False, jarak Euclidean, MinMaxScaler.

## Cara Menjalankan Lokal

```bash
# 1. Clone repo
git clone https://github.com/<username>/knn-clothing-size-caesar.git
cd knn-clothing-size-caesar

# 2. Install dependensi
pip install -r requirements.txt

# 3. Siapkan dataset (tidak ada di repo karena ukuran besar)
#    Letakkan caesar.csv di folder data/
mkdir data
# salin file caesar.csv ke data/

# 4. Train model (sekali saja)
python train_model.py --data data/caesar.csv

# 5. Jalankan aplikasi
streamlit run app.py
```

## Struktur Repo

```
knn-clothing-size-caesar/
├── app.py              # Aplikasi Streamlit
├── train_model.py      # Script training model
├── requirements.txt    # Dependensi Python
├── model/
│   └── model.pkl       # Model terlatih (di-commit ke repo)
└── data/               # Dataset (tidak di-commit, ada di .gitignore)
```

## Dataset

Dataset **CAESAR** (*Civilian American and European Surface Anthropometry Resource*) tersedia di [Kaggle](https://www.kaggle.com). Dataset berisi data antropometri 4.465 subjek dewasa Amerika dan Eropa.

## Lisensi

Kode sumber: MIT License.  
Dataset CAESAR: lihat ketentuan penggunaan dari sumber aslinya.
