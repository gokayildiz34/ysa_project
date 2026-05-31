
"""
predictor.py — Cok-model destekli anomali tahmin motoru.
Desteklenen model_type degerleri:
    "autoencoder"      → Keras autoencoder (reconstruction error)
    "isolation_forest" → Isolation Forest (decision_function)
    "ocsvm"            → One-Class SVM (decision_function)
    "pca"              → PCA reconstruction error
"""

import json
import os
from datetime import datetime
from importlib.machinery import SourceFileLoader
from pathlib import Path

import joblib
import numpy as np
import tensorflow as tf

BASE_DIR   = Path(__file__).resolve().parents[1]
MODEL_DIR  = BASE_DIR / "models"
OUTPUT_DIR = BASE_DIR / "outputs"
ML_DIR     = BASE_DIR / "ml"

# Autoencoder yollari
AE_MODEL_PATH    = MODEL_DIR / "autoencoder.keras"
AE_THRESHOLD_PATH = MODEL_DIR / "threshold.txt"
BEST_CONFIG_PATH = MODEL_DIR / "best_config.json"
METRICS_PATH     = OUTPUT_DIR / "metrics.json"

# Ortak yollar
SCALER_PATH          = MODEL_DIR / "scaler.pkl"
FEATURE_COLUMNS_PATH = MODEL_DIR / "feature_columns.pkl"
COMPARISON_PATH      = OUTPUT_DIR / "model_comparison.json"

DEFAULT_ANOMALY_RATIO_THRESHOLD = 0.10
DEFAULT_WINDOW_SIZE             = 1.0
DEFAULT_MODEL_TYPE              = "autoencoder"

# Desteklenen modellerin dosya bilgileri
MODEL_REGISTRY = {
    "autoencoder": {
        "model_file":     "autoencoder.keras",
        "threshold_file": "threshold.txt",
        "loader":         "keras",
        "score_mode":     "reconstruction",
    },
    "isolation_forest": {
        "model_file":     "isolation_forest.pkl",
        "threshold_file": "if_threshold.txt",
        "loader":         "joblib",
        "score_mode":     "decision",
    },
    "ocsvm": {
        "model_file":     "ocsvm.pkl",
        "threshold_file": "ocsvm_threshold.txt",
        "loader":         "joblib",
        "score_mode":     "decision",
    },
    "pca": {
        "model_file":     "pca_model.pkl",
        "threshold_file": "pca_threshold.txt",
        "loader":         "joblib",
        "score_mode":     "reconstruction",
    },
}

# Bellek cache — her model_type icin ayri slot
_cache: dict = {}

import types
_loader = SourceFileLoader("extractor", str(ML_DIR / "01_extract_features.py"))
extractor = types.ModuleType("extractor")
_loader.exec_module(extractor)


# ── Yardimci ─────────────────────────────────────────────────────────────────
def read_json(path: Path):
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def validate_model_type(model_type: str):
    if model_type not in MODEL_REGISTRY:
        raise ValueError(
            f"Bilinmeyen model_type: '{model_type}'. "
            f"Gecerli: {list(MODEL_REGISTRY.keys())}"
        )


def validate_model_files(model_type: str):
    validate_model_type(model_type)
    reg = MODEL_REGISTRY[model_type]
    needed = [
        MODEL_DIR / reg["model_file"],
        MODEL_DIR / reg["threshold_file"],
        SCALER_PATH,
        FEATURE_COLUMNS_PATH,
    ]
    missing = [str(p) for p in needed if not p.exists()]
    if missing:
        raise FileNotFoundError(
            f"[{model_type}] Eksik dosyalar: " + ", ".join(missing)
        )


# ── Artifact Yukleyici ────────────────────────────────────────────────────────
def load_artifacts(model_type: str = DEFAULT_MODEL_TYPE):
    validate_model_files(model_type)
    reg = MODEL_REGISTRY[model_type]

    if model_type not in _cache:
        model_path = MODEL_DIR / reg["model_file"]
        if reg["loader"] == "keras":
            model = tf.keras.models.load_model(model_path)
        else:
            model = joblib.load(model_path)

        threshold       = float((MODEL_DIR / reg["threshold_file"]).read_text().strip())
        scaler          = joblib.load(SCALER_PATH)
        feature_columns = joblib.load(FEATURE_COLUMNS_PATH)

        _cache[model_type] = {
            "model":           model,
            "threshold":       threshold,
            "scaler":          scaler,
            "feature_columns": feature_columns,
            "score_mode":      reg["score_mode"],
        }

    return _cache[model_type]


# ── Skor Hesapla ──────────────────────────────────────────────────────────────
def compute_scores(model, X_scaled: np.ndarray, score_mode: str) -> np.ndarray:
    if score_mode == "reconstruction":
        if hasattr(model, "predict"):
            reconstructed = model.predict(X_scaled, verbose=0)
        else:
            X_reduced     = model.transform(X_scaled)
            reconstructed = model.inverse_transform(X_reduced)
        return np.mean(np.square(X_scaled - reconstructed), axis=1)
    return -model.decision_function(X_scaled)


# ── API: Model Bilgisi ────────────────────────────────────────────────────────
def get_model_info():
    """Autoencoder konfigurasyonu ve metriklerini dondurur (geri uyumluluk)."""
    validate_model_files("autoencoder")
    artifacts       = load_artifacts("autoencoder")
    best_config     = read_json(BEST_CONFIG_PATH)
    metrics         = read_json(METRICS_PATH)
    feature_columns = artifacts["feature_columns"]
    return {
        "model_path":     str(AE_MODEL_PATH),
        "threshold":      artifacts["threshold"],
        "feature_count":  len(feature_columns),
        "feature_columns": feature_columns,
        "best_config":    best_config,
        "metrics":        metrics,
    }


def get_model_comparison():
    """outputs/model_comparison.json icerigini dondurur."""
    data = read_json(COMPARISON_PATH)
    if data is None:
        raise FileNotFoundError(
            "model_comparison.json bulunamadi. "
            "Once python -m ml.04b_evaluate_all_models calistirin."
        )
    return data


def get_available_models():
    """Hangi modellerin egitilmis oldugunu kontrol eder."""
    result = {}
    for name, reg in MODEL_REGISTRY.items():
        model_exists     = (MODEL_DIR / reg["model_file"]).exists()
        threshold_exists = (MODEL_DIR / reg["threshold_file"]).exists()
        result[name] = {
            "trained":    model_exists and threshold_exists,
            "model_file": reg["model_file"],
        }
    return result


# ── Anomali Aralik Bulucu ──────────────────────────────────────────────────────
def find_anomaly_ranges(windows):
    ranges, start, previous = [], None, None
    for item in windows:
        if item["is_anomaly"]:
            if start is None:
                start = item["window_id"]
            previous = item["window_id"]
        elif start is not None:
            ranges.append({"start_window": start, "end_window": previous})
            start = previous = None
    if start is not None:
        ranges.append({"start_window": start, "end_window": previous})
    return ranges


# ── Ana Analiz Fonksiyonu ─────────────────────────────────────────────────────
def analyze_pcap(
    pcap_path,
    source="upload",
    anomaly_ratio_threshold=DEFAULT_ANOMALY_RATIO_THRESHOLD,
    window_size=DEFAULT_WINDOW_SIZE,
    model_type=DEFAULT_MODEL_TYPE,
):
    artifacts       = load_artifacts(model_type)
    model           = artifacts["model"]
    scaler          = artifacts["scaler"]
    feature_columns = artifacts["feature_columns"]
    threshold       = artifacts["threshold"]
    score_mode      = artifacts["score_mode"]

    df = extractor.extract_features_from_pcap(
        str(pcap_path),
        label=-1,
        window_size=window_size,
    )
    if df.empty:
        raise ValueError("Bu dosyadan feature cikarilamadi.")

    X        = df[feature_columns].values
    X_scaled = scaler.transform(X)

    scores            = compute_scores(model, X_scaled, score_mode)
    predicted_windows = (scores > threshold).astype(int)

    windows = [
        {
            "window_id":           int(df.iloc[i]["window_id"]),
            "reconstruction_error": float(scores[i]),   # genel skor alanı
            "is_anomaly":          bool(predicted_windows[i]),
        }
        for i in range(len(scores))
    ]

    total_windows  = len(windows)
    anomaly_windows = int(predicted_windows.sum())
    anomaly_ratio   = anomaly_windows / total_windows if total_windows else 0
    avg_error       = float(np.mean(scores))
    max_error       = float(np.max(scores))
    result          = "ANOMALY" if anomaly_ratio >= anomaly_ratio_threshold else "NORMAL"

    anomaly_ranges = find_anomaly_ranges(windows)
    top_windows    = sorted(windows, key=lambda w: w["reconstruction_error"], reverse=True)[:10]

    summary = {
        "file_name":              os.path.basename(str(pcap_path)),
        "stored_path":            str(pcap_path),
        "source":                 source,
        "model_type":             model_type,
        "created_at":             datetime.now().isoformat(timespec="seconds"),
        "result":                 result,
        "total_windows":          total_windows,
        "anomaly_windows":        anomaly_windows,
        "anomaly_ratio":          float(anomaly_ratio),
        "avg_error":              avg_error,
        "max_error":              max_error,
        "threshold":              float(threshold),
        "anomaly_ratio_threshold": float(anomaly_ratio_threshold),
        "estimated_start_window": anomaly_ranges[0]["start_window"] if anomaly_ranges else None,
        "estimated_end_window":   anomaly_ranges[-1]["end_window"]  if anomaly_ranges else None,
        "anomaly_ranges":         anomaly_ranges,
        "top_windows":            top_windows,
    }
    return summary, windows
