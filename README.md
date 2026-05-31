# NetAnomAI

Multi-model network anomaly detection system — Autoencoder, Isolation Forest, One-Class SVM, PCA.

## 🚀 Projeyi Başlatma (Hızlı)

```powershell
# 1. Bağımlılıkları kur
C:\Users\Ufuk\AppData\Local\Programs\Python\Python311\python.exe -m pip install -r requirements.txt

# 2. Backend'i başlat
C:\Users\Ufuk\AppData\Local\Programs\Python\Python311\python.exe -m uvicorn backend.main:app --reload

# 3. Tarayıcıda aç
# http://127.0.0.1:8000
```

---

## 📋 Tam Pipeline (Sıfırdan Eğitim)

```powershell
# Proje klasörüne geç
cd C:\Users\Ufuk\Desktop\NetAnomAI

# Kısayol için değişken tanımla
$PY = "C:\Users\Ufuk\AppData\Local\Programs\Python\Python311\python.exe"

# Adım 1: Ham PCAP dosyalarından feature çıkar
&$PY -m ml.01_extract_features

# Adım 2: Veri setini hazırla (train/val/test split)
&$PY -m ml.02_prepare_dataset

# Adım 3a: Autoencoder eğit
&$PY -m ml.03_train_autoencoder

# Adım 3b: Diğer modelleri eğit (IF, OC-SVM, PCA)
&$PY -m ml.03b_train_classic_models

# Adım 4a: Autoencoder'ı değerlendir
&$PY -m ml.04_evaluate_model

# Adım 4b: Tüm modelleri karşılaştır (Compare Models sayfası)
&$PY -m ml.04b_evaluate_all_models

# Adım 4c: Isolation Forest raporu üret
&$PY -m ml.04c_evaluate_isolation_forest

# Adım 4d: One-Class SVM raporu üret
&$PY -m ml.04d_evaluate_ocsvm

# Adım 4e: PCA Reconstruction raporu üret
&$PY -m ml.04e_evaluate_pca

# Adım 5: Backend'i başlat
&$PY -m uvicorn backend.main:app --reload
```

---

## 🌐 Web Arayüzü Sayfaları

| Sayfa | Açıklama |
|-------|----------|
| **Analyze** | PCAP dosyası yükle, model seç, anomali analizi yap |
| **Realistic Test** | `data/raw/realistic_test/` içindeki hazır dosyaları analiz et |
| **History** | Tüm geçmiş analizler (SQLite) |
| **Model Info** | Autoencoder konfigürasyonu ve metrikleri |
| **Compare Models** | 4 modelin yan yana karşılaştırması |
| **Isolation Forest** | IF değerlendirme raporu + görseller |
| **One-Class SVM** | OC-SVM değerlendirme raporu + görseller |
| **PCA Reconstruction** | PCA değerlendirme raporu + görseller |

---

## 📁 Klasör Yapısı

```
NetAnomAI/
├── data/
│   ├── raw/
│   │   ├── normal/          ← normal trafik PCAP'ları buraya
│   │   ├── anomaly/         ← anomali PCAP'ları buraya
│   │   └── realistic_test/  ← test için karışık PCAP'lar
│   ├── splits/              ← otomatik oluşturulur
│   └── uploads/             ← web'den yüklenen dosyalar
├── ml/
│   ├── 01_extract_features.py
│   ├── 02_prepare_dataset.py
│   ├── 03_train_autoencoder.py
│   ├── 03b_train_classic_models.py
│   ├── 04_evaluate_model.py
│   ├── 04b_evaluate_all_models.py
│   ├── 04c_evaluate_isolation_forest.py
│   ├── 04d_evaluate_ocsvm.py
│   └── 04e_evaluate_pca.py
├── models/                  ← eğitilmiş modeller (.keras, .pkl)
├── outputs/
│   ├── isolation_forest/    ← IF metrikleri + grafikler
│   ├── ocsvm/               ← OC-SVM metrikleri + grafikler
│   └── pca/                 ← PCA metrikleri + grafikler
├── backend/
│   ├── main.py              ← FastAPI uygulama
│   ├── predictor.py         ← model yükleme + tahmin
│   └── database.py          ← SQLite yönetimi
├── frontend/
│   ├── index.html
│   └── assets/
│       ├── app.js
│       └── styles.css
└── requirements.txt
```

---

## ⚡ Sadece Web'i Başlatmak (Modeller Zaten Eğitilmişse)

```powershell
cd C:\Users\Ufuk\Desktop\NetAnomAI
C:\Users\Ufuk\AppData\Local\Programs\Python\Python311\python.exe -m uvicorn backend.main:app --reload
```

Tarayıcıda: **http://127.0.0.1:8000**
