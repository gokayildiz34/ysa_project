import json
import os

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

SPLIT_DIR  = "data/splits"
MODEL_DIR  = "models"
OUTPUT_DIR = os.path.join("outputs", "isolation_forest")

TEST_CSV = os.path.join(SPLIT_DIR, "test", "test_all.csv")

FEATURE_COLUMNS_PATH = os.path.join(MODEL_DIR, "feature_columns.pkl")
SCALER_PATH          = os.path.join(MODEL_DIR, "scaler.pkl")
MODEL_PATH           = os.path.join(MODEL_DIR, "isolation_forest.pkl")
THRESHOLD_PATH       = os.path.join(MODEL_DIR, "if_threshold.txt")
CONFIG_PATH          = os.path.join(MODEL_DIR, "if_config.json")

MODEL_NAME  = "Isolation Forest"
MODEL_COLOR = "#a78bfa"


def load_data():
    for p in [TEST_CSV, SCALER_PATH, FEATURE_COLUMNS_PATH, MODEL_PATH, THRESHOLD_PATH]:
        if not os.path.exists(p):
            raise FileNotFoundError(
                f"[HATA] Eksik: {p}\n"
                "Sirasıyla 02_prepare_dataset.py ve 03b_train_classic_models.py calistir."
            )

    feature_columns = joblib.load(FEATURE_COLUMNS_PATH)
    scaler          = joblib.load(SCALER_PATH)
    model           = joblib.load(MODEL_PATH)
    threshold       = float(open(THRESHOLD_PATH).read().strip())

    test_df = pd.read_csv(TEST_CSV)
    X_test  = scaler.transform(test_df[feature_columns].values)
    y_true  = test_df["label"].values

    return model, threshold, X_test, y_true, feature_columns


def plot_confusion_matrix(cm, out_dir):
    fig, ax = plt.subplots(figsize=(5, 4))
    fig.patch.set_facecolor("#0f172a")
    ax.set_facecolor("#1e293b")

    im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
    cbar = fig.colorbar(im, ax=ax)
    cbar.ax.yaxis.set_tick_params(color="white")
    plt.setp(cbar.ax.yaxis.get_ticklabels(), color="white")

    classes = ["Normal", "Anomaly"]
    tick_marks = np.arange(len(classes))
    ax.set_xticks(tick_marks); ax.set_xticklabels(classes, color="white", fontsize=11)
    ax.set_yticks(tick_marks); ax.set_yticklabels(classes, color="white", fontsize=11)
    ax.tick_params(colors="white")

    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(cm[i, j]),
                    ha="center", va="center", fontsize=14, fontweight="bold",
                    color="white" if cm[i, j] < thresh else "#0f172a")

    ax.set_xlabel("Predicted Label", color="white", fontsize=11)
    ax.set_ylabel("True Label", color="white", fontsize=11)
    ax.set_title(f"{MODEL_NAME} — Confusion Matrix", color="white", fontsize=13, pad=12)
    ax.spines[:].set_color("#1a2744")

    path = os.path.join(out_dir, "confusion_matrix.png")
    plt.tight_layout()
    plt.savefig(path, dpi=120, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    print(f"[KAYIT] {path}")


def plot_score_histogram(scores, threshold, y_true, out_dir):
    fig, ax = plt.subplots(figsize=(8, 4))
    fig.patch.set_facecolor("#0f172a")
    ax.set_facecolor("#1e293b")

    normal_scores  = scores[y_true == 0]
    anomaly_scores = scores[y_true == 1]

    ax.hist(normal_scores,  bins=60, alpha=0.7, color="#00d4aa", label="Normal")
    ax.hist(anomaly_scores, bins=60, alpha=0.7, color="#ff4757", label="Anomaly")
    ax.axvline(threshold, color="#fbbf24", linestyle="--", linewidth=2, label=f"Threshold={threshold:.4f}")

    ax.set_xlabel("Decision Function Score", color="white")
    ax.set_ylabel("Count", color="white")
    ax.set_title(f"{MODEL_NAME} — Score Distribution", color="white", fontsize=13)
    ax.tick_params(colors="white")
    ax.spines[:].set_color("#1a2744")
    legend = ax.legend(facecolor="#1e293b", edgecolor="#2a3a5e", labelcolor="white")

    path = os.path.join(out_dir, "score_histogram.png")
    plt.tight_layout()
    plt.savefig(path, dpi=120, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    print(f"[KAYIT] {path}")


def plot_roc_curve(y_true, roc_scores, roc_auc, out_dir):
    fpr, tpr, _ = roc_curve(y_true, roc_scores)
    fig, ax = plt.subplots(figsize=(6, 5))
    fig.patch.set_facecolor("#0f172a")
    ax.set_facecolor("#1e293b")

    ax.plot(fpr, tpr, color=MODEL_COLOR, linewidth=2, label=f"ROC-AUC = {roc_auc:.4f}")
    ax.plot([0, 1], [0, 1], color="gray", linestyle="--", linewidth=1)
    ax.set_xlabel("False Positive Rate", color="white")
    ax.set_ylabel("True Positive Rate", color="white")
    ax.set_title(f"{MODEL_NAME} — ROC Curve", color="white", fontsize=13)
    ax.tick_params(colors="white")
    ax.spines[:].set_color("#1a2744")
    ax.legend(facecolor="#1e293b", edgecolor="#2a3a5e", labelcolor="white")

    path = os.path.join(out_dir, "roc_curve.png")
    plt.tight_layout()
    plt.savefig(path, dpi=120, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    print(f"[KAYIT] {path}")


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"\n{'='*55}")
    print(f"[IF] {MODEL_NAME} Degerlendirmesi")
    print(f"{'='*55}")

    try:
        model, threshold, X_test, y_true, feature_columns = load_data()
    except FileNotFoundError as e:
        print(e); return

    print(f"[INFO] Test seti: {len(y_true)} satir")
    print(f"[INFO] Normal   : {(y_true==0).sum()}")
    print(f"[INFO] Anomaly  : {(y_true==1).sum()}")

    scores    = model.decision_function(X_test)
    roc_scores = -scores
    y_pred    = (scores < threshold).astype(int)

    cm   = confusion_matrix(y_true, y_pred, labels=[0, 1])
    acc  = float(accuracy_score(y_true, y_pred))
    prec = float(precision_score(y_true, y_pred, zero_division=0))
    rec  = float(recall_score(y_true, y_pred, zero_division=0))
    f1   = float(f1_score(y_true, y_pred, zero_division=0))
    try:
        roc = float(roc_auc_score(y_true, roc_scores))
    except Exception:
        roc = None

    print(f"\n[IF] Accuracy  : {acc:.4f}")
    print(f"[IF] Precision : {prec:.4f}")
    print(f"[IF] Recall    : {rec:.4f}")
    print(f"[IF] F1-Score  : {f1:.4f}")
    print(f"[IF] ROC-AUC   : {roc:.4f}" if roc else "[IF] ROC-AUC   : N/A")
    print(f"[IF] Threshold : {threshold:.6f}")

    config = {}
    if os.path.exists(CONFIG_PATH):
        with open(CONFIG_PATH, encoding="utf-8") as f:
            config = json.load(f)

    metrics = {
        "model_name":      MODEL_NAME,
        "model_type":      "isolation_forest",
        "accuracy":        acc,
        "precision":       prec,
        "recall":          rec,
        "f1_score":        f1,
        "roc_auc":         roc,
        "threshold":       threshold,
        "score_type":      "decision_function",
        "confusion_matrix": cm.tolist(),
        "test_rows":       int(len(y_true)),
        "normal_rows":     int((y_true==0).sum()),
        "anomaly_rows":    int((y_true==1).sum()),
        "config":          config,
        "classification_report": classification_report(
            y_true, y_pred, labels=[0, 1],
            target_names=["Normal", "Anomaly"], zero_division=0
        ),
    }

    metrics_path = os.path.join(OUTPUT_DIR, "metrics.json")
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=4)
    print(f"[KAYIT] {metrics_path}")

    report_path = os.path.join(OUTPUT_DIR, "classification_report.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(f"{MODEL_NAME} Classification Report\n")
        f.write("=" * 50 + "\n\n")
        if config:
            f.write("Config:\n")
            f.write(json.dumps(config, indent=4) + "\n\n")
        f.write(classification_report(
            y_true, y_pred, labels=[0, 1],
            target_names=["Normal", "Anomaly"], zero_division=0
        ))
        f.write(f"\nConfusion Matrix:\n{cm}\n\n")
        f.write(f"Accuracy  : {acc:.10f}\n")
        f.write(f"Precision : {prec:.10f}\n")
        f.write(f"Recall    : {rec:.10f}\n")
        f.write(f"F1-Score  : {f1:.10f}\n")
        f.write(f"ROC-AUC   : {roc:.10f}\n" if roc else "ROC-AUC   : N/A\n")
        f.write(f"Threshold : {threshold:.10f}\n")
    print(f"[KAYIT] {report_path}")

    plot_confusion_matrix(cm, OUTPUT_DIR)
    plot_score_histogram(scores, threshold, y_true, OUTPUT_DIR)
    if roc is not None:
        plot_roc_curve(y_true, roc_scores, roc, OUTPUT_DIR)

    print(f"\n{'='*55}")
    print(f"[TAMAM] {MODEL_NAME} degerlendirmesi tamamlandi.")
    print(f"  Ciktilar: {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
