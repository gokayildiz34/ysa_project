

import json
import os

import joblib
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import OneClassSVM

# ── Yollar ──────────────────────────────────────────────────────────────────
SPLIT_DIR = "data/splits"
MODEL_DIR  = "models"

TRAIN_CSV = os.path.join(SPLIT_DIR, "train", "train_normal.csv")
VAL_CSV   = os.path.join(SPLIT_DIR, "validation", "validation_normal.csv")

SCALER_PATH         = os.path.join(MODEL_DIR, "scaler.pkl")
FEATURE_COLUMNS_PATH = os.path.join(MODEL_DIR, "feature_columns.pkl")

DROP_COLUMNS = ["file_name", "window_id", "label"]
RANDOM_STATE = 42

# Threshold: validation normal verisinin bu percentile'i esik olarak kullanilir.
# IF/OCSVM icin: decision_function skoru bu degerden DUSUK olanlar anomali.
# PCA icin      : reconstruction error bu degerden YUKSEK olanlar anomali.
THRESHOLD_PERCENTILE = 95.0


# ── Yardimci ────────────────────────────────────────────────────────────────
def load_data():
    """Train ve validation veri setlerini yukler, scaler ve feature_columns'u dondurur."""
    if not os.path.exists(TRAIN_CSV):
        raise FileNotFoundError(f"[HATA] Train dosyasi bulunamadi: {TRAIN_CSV}\n"
                                "Once 02_prepare_dataset.py calistir.")
    if not os.path.exists(VAL_CSV):
        raise FileNotFoundError(f"[HATA] Validation dosyasi bulunamadi: {VAL_CSV}\n"
                                "Once 02_prepare_dataset.py calistir.")
    if not os.path.exists(FEATURE_COLUMNS_PATH):
        raise FileNotFoundError(f"[HATA] feature_columns.pkl bulunamadi: {FEATURE_COLUMNS_PATH}\n"
                                "Once 03_train_autoencoder.py calistir.")

    feature_columns = joblib.load(FEATURE_COLUMNS_PATH)

    train_df = pd.read_csv(TRAIN_CSV)
    val_df   = pd.read_csv(VAL_CSV)

    X_train = train_df[feature_columns].values
    X_val   = val_df[feature_columns].values

    # Mevcut scaler varsa kullan, yoksa yeni MinMaxScaler olustur
    if os.path.exists(SCALER_PATH):
        print("[INFO] Mevcut scaler.pkl kullaniliyor.")
        scaler = joblib.load(SCALER_PATH)
        X_train_s = scaler.transform(X_train)
        X_val_s   = scaler.transform(X_val)
    else:
        print("[INFO] Yeni MinMaxScaler olusturuluyor.")
        scaler = MinMaxScaler()
        X_train_s = scaler.fit_transform(X_train)
        X_val_s   = scaler.transform(X_val)
        joblib.dump(scaler, SCALER_PATH)

    return X_train_s, X_val_s, feature_columns


def save_config(name, config_dict):
    path = os.path.join(MODEL_DIR, f"{name}_config.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(config_dict, f, indent=4)
    print(f"[KAYIT] {path}")


# ── Isolation Forest ─────────────────────────────────────────────────────────
def train_isolation_forest(X_train, X_val):
    print("\n" + "="*55)
    print("[IF] Isolation Forest egitimi basliyor...")

    model = IsolationForest(
        n_estimators=200,
        contamination=0.05,   # Egitim verisinde ~%5 anomali tahmini
        max_samples="auto",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    model.fit(X_train)

    # Validation: decision_function → negatif = daha anormal
    val_scores = model.decision_function(X_val)          # buyuk = normal
    # Esik: normal skorlarin alt percentile'i
    threshold = float(np.percentile(val_scores, 100.0 - THRESHOLD_PERCENTILE))

    model_path     = os.path.join(MODEL_DIR, "isolation_forest.pkl")
    threshold_path = os.path.join(MODEL_DIR, "if_threshold.txt")

    joblib.dump(model, model_path)
    with open(threshold_path, "w") as f:
        f.write(str(threshold))

    print(f"[IF] Threshold ({100-THRESHOLD_PERCENTILE:.0f}th pct of val scores): {threshold:.6f}")
    print(f"[KAYIT] {model_path}")
    print(f"[KAYIT] {threshold_path}")

    val_preds = (val_scores < threshold).astype(int)
    print(f"[IF] Validation anomaly rate: {val_preds.mean()*100:.1f}%")

    save_config("if", {
        "model_type": "isolation_forest",
        "n_estimators": 200,
        "contamination": 0.05,
        "threshold_percentile": THRESHOLD_PERCENTILE,
        "threshold": threshold,
        "train_rows": int(len(X_train)),
        "val_rows":   int(len(X_val)),
        "val_anomaly_rate": float(val_preds.mean()),
    })
    return model, threshold


# ── One-Class SVM ────────────────────────────────────────────────────────────
def train_ocsvm(X_train, X_val):
    print("\n" + "="*55)
    print("[OCSVM] One-Class SVM egitimi basliyor...")
    print(f"[OCSVM] Train boyutu: {X_train.shape} — bu biraz surebilir...")

    # Buyuk veri setleri icin subsample
    max_samples = 5000
    if len(X_train) > max_samples:
        rng = np.random.default_rng(RANDOM_STATE)
        idx = rng.choice(len(X_train), max_samples, replace=False)
        X_fit = X_train[idx]
        print(f"[OCSVM] Train verisi {max_samples} ornek ile sinirlandirildi.")
    else:
        X_fit = X_train

    model = OneClassSVM(
        kernel="rbf",
        nu=0.05,       # beklenen anomali orani
        gamma="scale",
    )
    model.fit(X_fit)

    val_scores = model.decision_function(X_val)
    threshold  = float(np.percentile(val_scores, 100.0 - THRESHOLD_PERCENTILE))

    model_path     = os.path.join(MODEL_DIR, "ocsvm.pkl")
    threshold_path = os.path.join(MODEL_DIR, "ocsvm_threshold.txt")

    joblib.dump(model, model_path)
    with open(threshold_path, "w") as f:
        f.write(str(threshold))

    print(f"[OCSVM] Threshold: {threshold:.6f}")
    print(f"[KAYIT] {model_path}")
    print(f"[KAYIT] {threshold_path}")

    val_preds = (val_scores < threshold).astype(int)
    print(f"[OCSVM] Validation anomaly rate: {val_preds.mean()*100:.1f}%")

    save_config("ocsvm", {
        "model_type": "ocsvm",
        "kernel": "rbf",
        "nu": 0.05,
        "gamma": "scale",
        "threshold_percentile": THRESHOLD_PERCENTILE,
        "threshold": threshold,
        "train_rows_used": int(len(X_fit)),
        "train_rows_total": int(len(X_train)),
        "val_rows": int(len(X_val)),
        "val_anomaly_rate": float(val_preds.mean()),
    })
    return model, threshold


# ── PCA Reconstruction ────────────────────────────────────────────────────────
def train_pca(X_train, X_val):
    print("\n" + "="*55)
    print("[PCA] PCA Reconstruction egitimi basliyor...")

    # Variance ile otomatik component secimi
    model = PCA(n_components=0.95, random_state=RANDOM_STATE)
    model.fit(X_train)

    n_comp = model.n_components_
    explained = float(model.explained_variance_ratio_.sum())
    print(f"[PCA] n_components: {n_comp}  (explained variance: {explained*100:.1f}%)")

    # Validation reconstruction error
    X_val_reduced     = model.transform(X_val)
    X_val_reconstructed = model.inverse_transform(X_val_reduced)
    val_errors = np.mean(np.square(X_val - X_val_reconstructed), axis=1)

    threshold = float(np.percentile(val_errors, THRESHOLD_PERCENTILE))

    model_path     = os.path.join(MODEL_DIR, "pca_model.pkl")
    threshold_path = os.path.join(MODEL_DIR, "pca_threshold.txt")

    joblib.dump(model, model_path)
    with open(threshold_path, "w") as f:
        f.write(str(threshold))

    print(f"[PCA] Threshold ({THRESHOLD_PERCENTILE:.0f}th pct val error): {threshold:.8f}")
    print(f"[KAYIT] {model_path}")
    print(f"[KAYIT] {threshold_path}")

    val_preds = (val_errors > threshold).astype(int)
    print(f"[PCA] Validation anomaly rate: {val_preds.mean()*100:.1f}%")

    save_config("pca", {
        "model_type": "pca",
        "n_components_requested": 0.95,
        "n_components_fitted": int(n_comp),
        "explained_variance": explained,
        "threshold_percentile": THRESHOLD_PERCENTILE,
        "threshold": threshold,
        "train_rows": int(len(X_train)),
        "val_rows":   int(len(X_val)),
        "val_anomaly_rate": float(val_preds.mean()),
    })
    return model, threshold


# ── Ana Fonksiyon ─────────────────────────────────────────────────────────────
def main():
    os.makedirs(MODEL_DIR, exist_ok=True)
    print("[INFO] Klasik model egitimi basliyor.")
    print("[INFO] Egitim ve validation SADECE normal veri ile yapiliyor.")

    try:
        X_train, X_val, feature_columns = load_data()
    except FileNotFoundError as e:
        print(e)
        return

    print(f"\n[INFO] Train: {X_train.shape[0]} satir, {X_train.shape[1]} ozellik")
    print(f"[INFO] Val  : {X_val.shape[0]}  satir")

    train_isolation_forest(X_train, X_val)
    train_ocsvm(X_train, X_val)
    train_pca(X_train, X_val)

    print("\n" + "="*55)
    print("[TAMAM] 3 model basariyla egitildi ve kaydedildi.")
    print(f"  models/isolation_forest.pkl  +  if_threshold.txt")
    print(f"  models/ocsvm.pkl             +  ocsvm_threshold.txt")
    print(f"  models/pca_model.pkl         +  pca_threshold.txt")
    print("\nSonraki adim: python -m ml.04b_evaluate_all_models")


if __name__ == "__main__":
    main()
