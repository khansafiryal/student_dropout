# 🎓 Early Warning System: Student Dropout Risk Prediction

> Prediksi Risiko Dropout Mahasiswa menggunakan Machine Learning untuk Deteksi Dini dan Intervensi Akademik

Projek ini merupakan bagian dari program **Internship Basic Computing Community — Divisi Data Science**. Kami menerapkan alur kerja data science end-to-end pada studi kasus nyata prediksi dropout mahasiswa, mulai dari pemahaman bisnis, preprocessing, pemodelan, hingga evaluasi model.

**Mata Kuliah:** Penambangan Data — Universitas Brawijaya 2025/2026  
**Dataset:** [UCI ML Repository — Predict Students' Dropout and Academic Success](https://archive.uci.edu/dataset/697/predict+students+dropout+and+academic+success)  
**Implementasi:** [Google Colab Notebook](https://colab.research.google.com/drive/137GY_gHl-E91SkH3kSesTXDRVNsrz9ww?usp=sharing)

---

## 📌 Business Understanding

Dropout mahasiswa berdampak besar — bagi mahasiswa (kerugian waktu & finansial) dan institusi (reputasi, akreditasi). Sistem ini dirancang sebagai **early warning system** yang aktif setelah semester 2 selesai, mengklasifikasikan status akademik mahasiswa ke dalam tiga kelas:

| Kelas | Proporsi | Karakteristik |
|-------|----------|---------------|
| Graduate | 49.1% | Sudah lulus |
| Dropout | 32.7% | Berhenti kuliah |
| **Enrolled** | 18.2% | **Masih aktif — pola fiturnya ambigu, berada di antara Dropout dan Graduate** |

Kelas **Enrolled** menjadi tantangan utama karena secara akademis profilnya tumpang tindih dengan kedua kelas lain — mahasiswa yang masih aktif bisa saja sedang menuju kelulusan *atau* menuju dropout. Hal ini mendorong strategi preprocessing yang dioptimalkan per kelompok algoritma.

**Metrik utama: F2-Score kelas Dropout** — recall diberi bobot 4× lebih besar dari precision, karena melewatkan mahasiswa yang akan dropout (false negative) jauh lebih merugikan daripada false alarm.

---

## 📊 Dataset

**4.424 baris × 35 fitur** dari Polytechnic Institute of Portalegre, Portugal (Realinho et al., 2022).

| Kelompok Fitur | Contoh |
|----------------|--------|
| Demografi | Gender, Age at Enrollment, Nationality, Marital Status |
| Jalur Akademik | Application Mode, Course, Admission Grade, Previous Qualification |
| Sosial-Ekonomi | Tuition Fees Paid, Debtor, Scholarship, Parent Education & Occupation |
| Makroekonomi | Unemployment Rate, Inflation Rate, GDP |
| Performa Akademik | Credited / Enrolled / Evaluations / Approved / Grade / Without Evaluations (Sem 1 & 2) |

---

## ⚙️ Pipeline

### Stage 1 — Early Preprocessing *(semua model)*

Dilakukan sebelum train-test split agar tidak ada data leakage dari koreksi berbasis domain knowledge.

| Langkah | Keterangan |
|---------|------------|
| Perbaikan tipe data | Kode nominal dikembalikan ke `object`; desimal Eropa (koma → titik) dikonversi |
| Grade format fix | Format ganda titik desimal seperti `"13.927.272"` diperbaiki menjadi `"13.92"` |
| Domain quality check | Hapus grade di luar 0–20; koreksi `Approved > 8/semester`; fix `Without Evaluations > Enrolled` |
| Feature engineering | `pass_rate_diff` = selisih pass rate Sem 2 − Sem 1, menangkap tren performa antar semester |
| Train-test split | Stratified 80:20 → 3.457 train / 864 test |

Dataset bersih: **4.321 baris** (103 baris dihapus).

---

### Stage 2 — Advanced Preprocessing: Mengapa Dipisah Dua Pipeline?

Algoritma berbeda memiliki sensitivitas berbeda terhadap skala data, distribusi, dan outlier:

- **Tree-based models** (Decision Tree, Random Forest, XGBoost, LightGBM) belajar via *threshold splitting* — tidak terpengaruh skala, tidak perlu distribusi normal, dan lebih toleran terhadap outlier
- **Non-tree-based models** (KNN, Naive Bayes) bergantung pada *jarak Euclidean* atau *asumsi distribusi Gaussian* — sangat sensitif terhadap skala, outlier, dan bentuk distribusi

Memaksakan preprocessing yang sama ke semua model justru menurunkan performa, terutama pada kelas Enrolled yang pola fiturnya sudah ambigu. Memisahkan pipeline memastikan setiap model mendapat data dalam format paling optimal sesuai cara kerjanya.

---

#### 🌲 Pipeline Tree-Based

> Model: **Decision Tree · Random Forest · XGBoost · LightGBM**
> 
> Karakteristik: tidak sensitif terhadap skala, outlier, dan redundansi fitur — keputusan berbasis threshold splitting

| Langkah | Metode | Alasan |
|---------|--------|--------|
| Redundancy check numerik | Spearman Correlation (threshold > 0.8) | Robust terhadap outlier & distribusi tidak normal; fitur redundan **dipertahankan** karena model masih dapat mengekstrak info temporal antar semester |
| Redundancy check kategorikal | Bias-Corrected Cramér's V | Hapus `attendance_time` (turunan `major`) & `international` (turunan `nationality`) |
| Encoding | Label Encoding semua fitur kategorikal | Aman untuk tree-based — nilai numerik hanya digunakan sebagai titik split, bukan sebagai jarak |
| Feature selection | Mutual Information + Sequential Forward Selection | Dua metode dibandingkan; SFS mempertimbangkan interaksi antar fitur |
| Class imbalance | SMOTE di dalam setiap fold CV | Mencegah data leakage dari oversampling |

---

#### 🔵 Pipeline Non-Tree-Based

> Model: **KNN · Naive Bayes**
>
> Karakteristik: KNN sensitif terhadap skala & outlier (jarak Euclidean); Naive Bayes mengasumsikan distribusi Gaussian per fitur

| Langkah | Metode | Alasan |
|---------|--------|--------|
| Outlier handling | IQR detection → nilai non-natural outlier diubah ke `NaN` → KNN Imputer (k=3) | Natural outlier (nilai ekstrem valid) dipertahankan; hanya outlier noise yang diimputasi |
| Normalisasi distribusi | Yeo-Johnson Power Transform | Mereduksi skewness agar mendekati distribusi Gaussian yang diasumsikan Naive Bayes; fit pada train only |
| Redundancy check numerik | Pearson Correlation (threshold ≥ 0.89) | Setelah transformasi, multikolinearitas tinggi perlu dihilangkan |
| Reduksi dimensi | PCA 1 komponen per pasang fitur redundan | Pasang `Enrolled Sem1&2` dan `Approved Sem1&2` dikompresi tanpa kehilangan informasi |
| Redundancy check kategorikal | Bias-Corrected Cramér's V | Sama seperti tree-based: hapus `attendance_time` & `international` |
| Encoding | Label Encoding (biner & ordinal) + Frequency Encoding (nominal) | Frequency Encoding menghindari urutan palsu pada fitur nominal di model berbasis jarak |
| Scaling | RobustScaler (median & IQR) | Lebih stabil terhadap natural outlier yang masih tersisa dibanding StandardScaler |
| Feature selection | Mutual Information + Sequential Forward Selection | Sama seperti tree-based, untuk perbandingan yang adil |
| Class imbalance | SMOTE di dalam setiap fold CV | Mencegah data leakage |

---

## 🛠️ Tech Stack

`Python` · `scikit-learn` · `XGBoost` · `LightGBM` · `Optuna` · `imbalanced-learn` · `pandas` · `numpy` · `matplotlib` · `seaborn`

---

