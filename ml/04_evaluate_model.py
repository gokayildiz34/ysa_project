import json
import os

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

SPLIT_DIR = "data/splits"
MODEL_DIR = "models"
OUTPUT_DIR = "outputs"

TEST_CSV = os.path.join(SPLIT_DIR, "test", "test_all.csv")

MODEL_PATH = os.path.join(MODEL_DIR, "autoencoder.keras")
SCALER_PATH = os.path.join(MODEL_DIR, "scaler.pkl")
THRESHOLD_PATH = os.path.join(MODEL_DIR, "threshold.txt")
FEATURE_COLUMNS_PATH = os.path.join(MODEL_DIR, "feature_columns.pkl")
BEST_CONFIG_PATH = os.path.join(MODEL_DIR, "best_config.json")

PREDICTIONS_CSV = os.path.join(OUTPUT_DIR, "predictions.csv")
FILE_SUMMARY_CSV = os.path.join(OUTPUT_DIR, "file_prediction_summary.csv")
REPORT_TXT = os.path.join(OUTPUT_DIR, "classification_report.txt")
METRICS_JSON = os.path.join(OUTPUT_DIR, "metrics.json")
CONFUSION_MATRIX_PNG = os.path.join(OUTPUT_DIR, "confusion_matrix.png")
ERROR_HISTOGRAM_PNG = os.path.join(OUTPUT_DIR, "reconstruction_error_histogram.png")


def load_best_config():
    if not os.path.exists(BEST_CONFIG_PATH):
        return None

    with open(BEST_CONFIG_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    needed_files = [
        TEST_CSV,
        MODEL_PATH,
        SCALER_PATH,
        THRESHOLD_PATH,
        FEATURE_COLUMNS_PATH,
    ]

    for file in needed_files:
        if not os.path.exists(file):
            print(f"[HATA] Eksik dosya: {file}")
            print("Once 01, 02 ve 03 dosyalarini sirayla calistir.")
            return

    model = tf.keras.models.load_model(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    feature_columns = joblib.load(FEATURE_COLUMNS_PATH)
    best_config = load_best_config()

    with open(THRESHOLD_PATH, "r", encoding="utf-8") as f:
        threshold = float(f.read().strip())

    test_df = pd.read_csv(TEST_CSV)

    X_test = test_df[feature_columns].values
    y_true = test_df["label"].values

    X_test_scaled = scaler.transform(X_test)

    reconstructed = model.predict(X_test_scaled, verbose=0)
    errors = np.mean(np.square(X_test_scaled - reconstructed), axis=1)

    y_pred = (errors > threshold).astype(int)

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])

    report = classification_report(
        y_true,
        y_pred,
        labels=[0, 1],
        target_names=["Normal", "Anomaly"],
        zero_division=0,
    )

    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)

    try:
        roc_auc = roc_auc_score(y_true, errors)
    except Exception:
        roc_auc = None

    predictions_df = pd.DataFrame(
        {
            "file_name": test_df["file_name"],
            "window_id": test_df["window_id"],
            "true_label": y_true,
            "predicted_label": y_pred,
            "reconstruction_error": errors,
            "threshold": threshold,
        }
    )
    predictions_df.to_csv(PREDICTIONS_CSV, index=False)

    file_summary = (
        predictions_df.groupby(["file_name", "true_label"])
        .agg(
            total_windows=("predicted_label", "size"),
            anomaly_windows=("predicted_label", "sum"),
            avg_error=("reconstruction_error", "mean"),
            max_error=("reconstruction_error", "max"),
        )
        .reset_index()
    )
    file_summary["anomaly_ratio"] = file_summary["anomaly_windows"] / file_summary["total_windows"]
    file_summary.to_csv(FILE_SUMMARY_CSV, index=False)

    with open(REPORT_TXT, "w", encoding="utf-8") as f:
        f.write("NetAnomAI Classification Report\n")
        f.write("=" * 40 + "\n\n")
        if best_config is not None:
            f.write("Best Config:\n")
            f.write(json.dumps(best_config, indent=4))
            f.write("\n\n")
        f.write(report)
        f.write("\n\nConfusion Matrix:\n")
        f.write(str(cm))
        f.write("\n\n")
        f.write(f"Accuracy: {accuracy}\n")
        f.write(f"Precision: {precision}\n")
        f.write(f"Recall: {recall}\n")
        f.write(f"F1-score: {f1}\n")
        f.write(f"ROC-AUC: {roc_auc}\n")
        f.write(f"Threshold: {threshold}\n")

    metrics = {
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1_score": float(f1),
        "roc_auc": None if roc_auc is None else float(roc_auc),
        "threshold": float(threshold),
        "confusion_matrix": cm.tolist(),
        "best_config": best_config,
    }

    with open(METRICS_JSON, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=4)

    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=["Normal", "Anomaly"],
    )

    disp.plot()
    plt.title("NetAnomAI Confusion Matrix")
    plt.savefig(CONFUSION_MATRIX_PNG, dpi=300, bbox_inches="tight")
    plt.close()

    plt.figure()
    for label_value, label_name in [(0, "Normal"), (1, "Anomaly")]:
        label_errors = errors[y_true == label_value]
        if len(label_errors) > 0:
            plt.hist(label_errors, bins=40, alpha=0.6, label=label_name)
    plt.axvline(threshold, color="red", linestyle="--", label="Threshold")
    plt.xlabel("Reconstruction Error")
    plt.ylabel("Window Count")
    plt.title("Reconstruction Error Distribution")
    plt.legend()
    plt.savefig(ERROR_HISTOGRAM_PNG, dpi=300, bbox_inches="tight")
    plt.close()

    print("\n[TAMAM] Model degerlendirildi.")
    print("\nConfusion Matrix:")
    print(cm)
    print("\nClassification Report:")
    print(report)
    print(f"Accuracy: {accuracy}")
    print(f"Precision: {precision}")
    print(f"Recall: {recall}")
    print(f"F1-score: {f1}")
    print(f"ROC-AUC: {roc_auc}")
    print("\nCiktilar outputs klasorune kaydedildi.")
    print(f"- {PREDICTIONS_CSV}")
    print(f"- {FILE_SUMMARY_CSV}")
    print(f"- {REPORT_TXT}")
    print(f"- {METRICS_JSON}")
    print(f"- {CONFUSION_MATRIX_PNG}")
    print(f"- {ERROR_HISTOGRAM_PNG}")


if __name__ == "__main__":
    main()
