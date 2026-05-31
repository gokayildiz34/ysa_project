import argparse
import os

import joblib
import numpy as np
import tensorflow as tf
from importlib.machinery import SourceFileLoader

extractor = SourceFileLoader(
    "extractor",
    os.path.join("ml", "01_extract_features.py"),
).load_module()

MODEL_DIR = "models"

MODEL_PATH = os.path.join(MODEL_DIR, "autoencoder.keras")
SCALER_PATH = os.path.join(MODEL_DIR, "scaler.pkl")
THRESHOLD_PATH = os.path.join(MODEL_DIR, "threshold.txt")
FEATURE_COLUMNS_PATH = os.path.join(MODEL_DIR, "feature_columns.pkl")

DEFAULT_ANOMALY_RATIO_THRESHOLD = 0.10
DEFAULT_WINDOW_SIZE = 1.0


def predict_single_file(
    pcap_path,
    anomaly_ratio_threshold=DEFAULT_ANOMALY_RATIO_THRESHOLD,
    window_size=DEFAULT_WINDOW_SIZE,
):
    if not os.path.exists(pcap_path):
        print(f"[HATA] Dosya bulunamadi: {pcap_path}")
        return

    needed_files = [
        MODEL_PATH,
        SCALER_PATH,
        THRESHOLD_PATH,
        FEATURE_COLUMNS_PATH,
    ]

    for file in needed_files:
        if not os.path.exists(file):
            print(f"[HATA] Eksik model dosyasi: {file}")
            print("Once modeli egitmelisin: python ml/03_train_autoencoder.py")
            return

    model = tf.keras.models.load_model(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    feature_columns = joblib.load(FEATURE_COLUMNS_PATH)

    with open(THRESHOLD_PATH, "r", encoding="utf-8") as f:
        threshold = float(f.read().strip())

    df = extractor.extract_features_from_pcap(
        pcap_path,
        label=-1,
        window_size=window_size,
    )

    if df.empty:
        print("[HATA] Bu dosyadan feature cikarilamadi.")
        return

    X = df[feature_columns].values
    X_scaled = scaler.transform(X)

    reconstructed = model.predict(X_scaled, verbose=0)
    errors = np.mean(np.square(X_scaled - reconstructed), axis=1)

    predicted_windows = (errors > threshold).astype(int)

    total_windows = len(predicted_windows)
    anomaly_windows = int(predicted_windows.sum())

    avg_error = float(np.mean(errors))
    max_error = float(np.max(errors))
    anomaly_ratio = anomaly_windows / total_windows

    result = "ANOMALY" if anomaly_ratio >= anomaly_ratio_threshold else "NORMAL"

    top_indices = np.argsort(errors)[-5:][::-1]

    print("\n===== TEK DOSYA TAHMIN SONUCU =====")
    print(f"Dosya: {os.path.basename(pcap_path)}")
    print(f"Sonuc: {result}")
    print(f"Toplam pencere: {total_windows}")
    print(f"Anomali pencere: {anomaly_windows}")
    print(f"Anomali orani: {anomaly_ratio:.4f}")
    print(f"Dosya karar esigi: {anomaly_ratio_threshold:.4f}")
    print(f"Ortalama reconstruction error: {avg_error}")
    print(f"Maksimum reconstruction error: {max_error}")
    print(f"Threshold: {threshold}")
    print("\nEn yuksek hatali pencereler:")

    for idx in top_indices:
        window_id = int(df.iloc[idx]["window_id"])
        error = float(errors[idx])
        window_label = "ANOMALY" if error > threshold else "NORMAL"
        print(f"- window_id={window_id}, error={error}, pencere_sonucu={window_label}")


def parse_args():
    parser = argparse.ArgumentParser(description="Tek bir PCAP/PCAPNG dosyasi icin anomali tahmini yapar.")
    parser.add_argument("pcap_path", help="Tahmin edilecek .pcap veya .pcapng dosyasi")
    parser.add_argument(
        "--anomaly-ratio-threshold",
        type=float,
        default=DEFAULT_ANOMALY_RATIO_THRESHOLD,
        help="Dosyanin ANOMALY sayilmasi icin gerekli minimum anomalili pencere orani.",
    )
    parser.add_argument(
        "--window-size",
        type=float,
        default=DEFAULT_WINDOW_SIZE,
        help="Feature extraction icin saniye cinsinden pencere boyutu.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    predict_single_file(
        args.pcap_path,
        anomaly_ratio_threshold=args.anomaly_ratio_threshold,
        window_size=args.window_size,
    )


if __name__ == "__main__":
    main()
