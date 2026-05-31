

import json
import os

import joblib
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

# ── Yollar ──────────────────────────────────────────────────────────────────
SPLIT_DIR  = "data/splits"
MODEL_DIR  = "models"
OUTPUT_DIR = "outputs"

TEST_CSV = os.path.join(SPLIT_DIR, "test", "test_all.csv")

FEATURE_COLUMNS_PATH = os.path.join(MODEL_DIR, "feature_columns.pkl")
SCALER_PATH          = os.path.join(MODEL_DIR, "scaler.pkl")

COMPARISON_JSON = os.path.join(OUTPUT_DIR, "model_comparison.json")
COMPARISON_CSV  = os.path.join(OUTPUT_DIR, "model_comparison.csv")
DROP_COLUMNS    = ["file_name", "window_id", "label"]


# ── Skor Hesapla ─────────────────────────────────────────────────────────────
def compute_metrics(y_true, y_pred, scores=None):
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    acc  = float(accuracy_score(y_true, y_pred))
    prec = float(precision_score(y_true, y_pred, zero_division=0))
    rec  = float(recall_score(y_true, y_pred, zero_division=0))
    f1   = float(f1_score(y_true, y_pred, zero_division=0))
    try:
        roc = float(roc_auc_score(y_true, scores)) if scores is not None else None
    except Exception:
        roc = None
    return {
        "accuracy":         acc,
        "precision":        prec,
        "recall":           rec,
        "f1_score":         f1,
        "roc_auc":          roc,
        "confusion_matrix": cm.tolist(),
        "classification_report": classification_report(
            y_true, y_pred, labels=[0, 1],
            target_names=["Normal", "Anomaly"],
            zero_division=0,
        ),
    }


# ── Autoencoder ───────────────────────────────────────────────────────────────
def evaluate_autoencoder(X_test, y_true):
    model_path     = os.path.join(MODEL_DIR, "autoencoder.keras")
    threshold_path = os.path.join(MODEL_DIR, "threshold.txt")
    if not os.path.exists(model_path):
        print("[SKIP] autoencoder.keras bulunamadi.")
        return None
    model = tf.keras.models.load_model(model_path)
    threshold = float(open(threshold_path).read().strip())
    reconstructed = model.predict(X_test, verbose=0)
    errors  = np.mean(np.square(X_test - reconstructed), axis=1)
    y_pred  = (errors > threshold).astype(int)
    metrics = compute_metrics(y_true, y_pred, scores=errors)
    metrics.update({"threshold": threshold, "score_type": "reconstruction_error"})
    print(f"[AE]    F1={metrics['f1_score']:.4f}  ROC-AUC={metrics['roc_auc']:.4f}")
    return metrics


# ── Isolation Forest ──────────────────────────────────────────────────────────
def evaluate_isolation_forest(X_test, y_true):
    model_path     = os.path.join(MODEL_DIR, "isolation_forest.pkl")
    threshold_path = os.path.join(MODEL_DIR, "if_threshold.txt")
    if not os.path.exists(model_path):
        print("[SKIP] isolation_forest.pkl bulunamadi.")
        return None
    model     = joblib.load(model_path)
    threshold = float(open(threshold_path).read().strip())
    scores    = model.decision_function(X_test)   # yuksek = normal
    y_pred    = (scores < threshold).astype(int)
    # ROC-AUC icin skoru tersine cevir (yuksek = anomaly)
    metrics   = compute_metrics(y_true, y_pred, scores=-scores)
    metrics.update({"threshold": threshold, "score_type": "decision_function"})
    print(f"[IF]    F1={metrics['f1_score']:.4f}  ROC-AUC={metrics['roc_auc']:.4f}")
    return metrics


# ── One-Class SVM ─────────────────────────────────────────────────────────────
def evaluate_ocsvm(X_test, y_true):
    model_path     = os.path.join(MODEL_DIR, "ocsvm.pkl")
    threshold_path = os.path.join(MODEL_DIR, "ocsvm_threshold.txt")
    if not os.path.exists(model_path):
        print("[SKIP] ocsvm.pkl bulunamadi.")
        return None
    model     = joblib.load(model_path)
    threshold = float(open(threshold_path).read().strip())
    scores    = model.decision_function(X_test)
    y_pred    = (scores < threshold).astype(int)
    metrics   = compute_metrics(y_true, y_pred, scores=-scores)
    metrics.update({"threshold": threshold, "score_type": "decision_function"})
    print(f"[SVM]   F1={metrics['f1_score']:.4f}  ROC-AUC={metrics['roc_auc']:.4f}")
    return metrics


# ── PCA Reconstruction ────────────────────────────────────────────────────────
def evaluate_pca(X_test, y_true):
    model_path     = os.path.join(MODEL_DIR, "pca_model.pkl")
    threshold_path = os.path.join(MODEL_DIR, "pca_threshold.txt")
    if not os.path.exists(model_path):
        print("[SKIP] pca_model.pkl bulunamadi.")
        return None
    model     = joblib.load(model_path)
    threshold = float(open(threshold_path).read().strip())
    X_reduced = model.transform(X_test)
    X_recon   = model.inverse_transform(X_reduced)
    errors    = np.mean(np.square(X_test - X_recon), axis=1)
    y_pred    = (errors > threshold).astype(int)
    metrics   = compute_metrics(y_true, y_pred, scores=errors)
    metrics.update({"threshold": threshold, "score_type": "reconstruction_error"})
    print(f"[PCA]   F1={metrics['f1_score']:.4f}  ROC-AUC={metrics['roc_auc']:.4f}")
    return metrics


# ── Ana Fonksiyon ─────────────────────────────────────────────────────────────
def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Kontrol
    for p in [TEST_CSV, SCALER_PATH, FEATURE_COLUMNS_PATH]:
        if not os.path.exists(p):
            print(f"[HATA] Eksik: {p}")
            print("Sirasıyla 01, 02, 03 ve 03b scriptlerini calistir.")
            return

    feature_columns = joblib.load(FEATURE_COLUMNS_PATH)
    scaler          = joblib.load(SCALER_PATH)

    test_df = pd.read_csv(TEST_CSV)
    X_test  = scaler.transform(test_df[feature_columns].values)
    y_true  = test_df["label"].values

    print(f"\n[INFO] Test seti: {len(test_df)} satir")
    print(f"[INFO] Normal   : {(y_true==0).sum()}")
    print(f"[INFO] Anomaly  : {(y_true==1).sum()}\n")

    results = {}

    r = evaluate_autoencoder(X_test, y_true)
    if r: results["autoencoder"] = r

    r = evaluate_isolation_forest(X_test, y_true)
    if r: results["isolation_forest"] = r

    r = evaluate_ocsvm(X_test, y_true)
    if r: results["ocsvm"] = r

    r = evaluate_pca(X_test, y_true)
    if r: results["pca"] = r

    if not results:
        print("[HATA] Hic model degerlendirilemedi.")
        return

    # JSON kaydet (classification_report metni haric)
    json_results = {}
    for k, v in results.items():
        json_results[k] = {kk: vv for kk, vv in v.items() if kk != "classification_report"}

    with open(COMPARISON_JSON, "w", encoding="utf-8") as f:
        json.dump(json_results, f, indent=4)

    # CSV ozet
    rows = []
    for model_name, m in results.items():
        rows.append({
            "model":     model_name,
            "accuracy":  m["accuracy"],
            "precision": m["precision"],
            "recall":    m["recall"],
            "f1_score":  m["f1_score"],
            "roc_auc":   m["roc_auc"],
            "threshold": m["threshold"],
        })
    pd.DataFrame(rows).to_csv(COMPARISON_CSV, index=False)

    print("\n" + "="*55)
    print("[TAMAM] Model karsilastirmasi tamamlandi.")
    print(f"  {COMPARISON_JSON}")
    print(f"  {COMPARISON_CSV}")

    print("\n-- Karsilastirma Ozeti --")
    for row in rows:
        print(f"  {row['model']:20s}  F1={row['f1_score']:.4f}  "
              f"Precision={row['precision']:.4f}  Recall={row['recall']:.4f}  "
              f"ROC-AUC={row['roc_auc']:.4f}" if row['roc_auc'] else
              f"  {row['model']:20s}  F1={row['f1_score']:.4f}")


if __name__ == "__main__":
    main()
