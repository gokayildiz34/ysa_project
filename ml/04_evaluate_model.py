

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

# Dosya yollarını tanımlıyoruz
SPLIT_DIR = "data/splits"
MODEL_DIR = "models"
OUTPUT_DIR = "outputs"

# Test verisi hem normal hem anomali içeriyor
TEST_CSV = os.path.join(SPLIT_DIR, "test", "test_all.csv")

MODEL_PATH = os.path.join(MODEL_DIR, "autoencoder.keras")
SCALER_PATH = os.path.join(MODEL_DIR, "scaler.pkl")
THRESHOLD_PATH = os.path.join(MODEL_DIR, "threshold.txt")
FEATURE_COLUMNS_PATH = os.path.join(MODEL_DIR, "feature_columns.pkl")
BEST_CONFIG_PATH = os.path.join(MODEL_DIR, "best_config.json")

# Çıktı dosyaları
PREDICTIONS_CSV = os.path.join(OUTPUT_DIR, "predictions.csv")
FILE_SUMMARY_CSV = os.path.join(OUTPUT_DIR, "file_prediction_summary.csv")
REPORT_TXT = os.path.join(OUTPUT_DIR, "classification_report.txt")
METRICS_JSON = os.path.join(OUTPUT_DIR, "metrics.json")
CONFUSION_MATRIX_PNG = os.path.join(OUTPUT_DIR, "confusion_matrix.png")
ERROR_HISTOGRAM_PNG = os.path.join(OUTPUT_DIR, "reconstruction_error_histogram.png")


def load_best_config():
    # Eğitim sırasında kaydedilen en iyi konfigürasyonu okuyoruz
    if not os.path.exists(BEST_CONFIG_PATH):
        return None
    with open(BEST_CONFIG_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Gerekli dosyaların hepsinin var olduğunu kontrol ediyoruz
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

    # Eğitilmiş modeli ve yardımcı nesneleri yüklüyoruz
    model = tf.keras.models.load_model(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    feature_columns = joblib.load(FEATURE_COLUMNS_PATH)
    best_config = load_best_config()

    with open(THRESHOLD_PATH, "r", encoding="utf-8") as f:
        threshold = float(f.read().strip())

    test_df = pd.read_csv(TEST_CSV)

    X_test = test_df[feature_columns].values
    y_true = test_df["label"].values  # 0 = normal, 1 = anomali

    # Test verisini eğitimle aynı ölçekleyiciyle dönüştürüyoruz
    X_test_scaled = scaler.transform(X_test)

    # Modelin yeniden oluşturma hatasını hesaplıyoruz
    reconstructed = model.predict(X_test_scaled, verbose=0)
    errors = np.mean(np.square(X_test_scaled - reconstructed), axis=1)

    # Hata eşik değerini aşıyorsa anomali diyoruz
    y_pred = (errors > threshold).astype(int)

    # Karmaşıklık matrisini oluşturuyoruz
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])

    report = classification_report(
        y_true,
        y_pred,
        labels=[0, 1],
        target_names=["Normal", "Anomali"],
        zero_division=0,
    )

    # Temel performans metriklerini hesaplıyoruz
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)

    try:
        # ROC-AUC için ham hata skorlarını kullanıyoruz (sadece tahmin değil)
        roc_auc = roc_auc_score(y_true, errors)
    except Exception:
        roc_auc = None  # İki sınıf yoksa hata verebilir, atlıyoruz

    # Pencere bazlı tahminleri CSV'ye kaydediyoruz
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

    # Dosya bazında özet rapor — hangi dosyada kaç anomali penceresi var
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

    # Detaylı metin raporunu yazıyoruz
    with open(REPORT_TXT, "w", encoding="utf-8") as f:
        f.write("NetAnomAI Siniflandirma Raporu\n")
        f.write("=" * 40 + "\n\n")
        if best_config is not None:
            f.write("En Iyi Konfigurasyon:\n")
            f.write(json.dumps(best_config, indent=4))
            f.write("\n\n")
        f.write(report)
        f.write("\n\nKarmasiklik Matrisi:\n")
        f.write(str(cm))
        f.write("\n\n")
        f.write(f"Dogruluk: {accuracy}\n")
        f.write(f"Kesinlik: {precision}\n")
        f.write(f"Duyarlilik: {recall}\n")
        f.write(f"F1-skoru: {f1}\n")
        f.write(f"ROC-AUC: {roc_auc}\n")
        f.write(f"Esik Degeri: {threshold}\n")

    # Frontend'in okuyabilmesi için metrikleri JSON'a kaydediyoruz
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

    # Karmaşıklık matrisini görsel olarak çiziyoruz — renk şemasını özelleştirdik
    fig, ax = plt.subplots(figsize=(6, 5))
    fig.patch.set_facecolor('#0f172a')  # Koyu arka plan
    ax.set_facecolor('#0f172a')

    import matplotlib.colors as mcolors
    # Koyu maviden turkuaza giden özel renk geçişi
    cmap = mcolors.LinearSegmentedColormap.from_list(
        'netonom', ['#1e293b', '#0ea5e9', '#00d4aa']
    )

    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    cbar = fig.colorbar(im, ax=ax)
    cbar.ax.yaxis.set_tick_params(color='#94a3b8')
    plt.setp(plt.getp(cbar.ax.axes, 'yticklabels'), color='#94a3b8')

    labels = ['Normal', 'Anomali']
    tick_marks = range(len(labels))
    ax.set_xticks(tick_marks)
    ax.set_yticks(tick_marks)
    ax.set_xticklabels(labels, color='#e2e8f0', fontsize=11)
    ax.set_yticklabels(labels, color='#e2e8f0', fontsize=11)

    # Her hücreye değeri yazıyoruz, arka plan rengine göre yazı rengi seçiyoruz
    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            color = '#0f172a' if cm[i, j] > thresh else '#e2e8f0'
            ax.text(j, i, str(cm[i, j]),
                    ha='center', va='center',
                    color=color, fontsize=14, fontweight='bold')

    ax.set_xlabel('Tahmin Edilen', color='#94a3b8', fontsize=11, labelpad=10)
    ax.set_ylabel('Gerçek Değer', color='#94a3b8', fontsize=11, labelpad=10)
    ax.set_title('NetAnomAI — Karmaşıklık Matrisi', color='#e2e8f0', fontsize=13, pad=14)
    ax.tick_params(colors='#94a3b8')
    for spine in ax.spines.values():
        spine.set_edgecolor('#334155')

    plt.tight_layout()
    plt.savefig(CONFUSION_MATRIX_PNG, dpi=300, bbox_inches='tight', facecolor='#0f172a')
    plt.close()

    # Normal ve anomali hatalarının dağılımını histogram olarak çiziyoruz
    # Eşik değeri kırmızı çizgi olarak gösteriliyor
    fig2, ax2 = plt.subplots(figsize=(8, 4))
    fig2.patch.set_facecolor('#0f172a')
    ax2.set_facecolor('#1e293b')

    colors_map = {0: '#00d4aa', 1: '#f43f5e'}  # Normal = turkuaz, Anomali = kırmızı
    for label_value, label_name in [(0, 'Normal'), (1, 'Anomali')]:
        label_errors = errors[y_true == label_value]
        if len(label_errors) > 0:
            ax2.hist(label_errors, bins=40, alpha=0.7,
                     label=label_name, color=colors_map[label_value], edgecolor='none')
    ax2.axvline(threshold, color='#f59e0b', linestyle='--', linewidth=1.8, label='Eşik Değeri')
    ax2.set_xlabel('Yeniden Yapılandırma Hatası', color='#94a3b8', fontsize=11)
    ax2.set_ylabel('Pencere Sayısı', color='#94a3b8', fontsize=11)
    ax2.set_title('Yeniden Yapılandırma Hatası Dağılımı', color='#e2e8f0', fontsize=13)
    ax2.tick_params(colors='#94a3b8')
    ax2.spines['bottom'].set_color('#334155')
    ax2.spines['left'].set_color('#334155')
    ax2.spines['top'].set_color('#334155')
    ax2.spines['right'].set_color('#334155')
    ax2.grid(True, color='#334155', linewidth=0.5, alpha=0.6)
    legend = ax2.legend(facecolor='#1e293b', edgecolor='#334155', labelcolor='#e2e8f0')
    plt.tight_layout()
    plt.savefig(ERROR_HISTOGRAM_PNG, dpi=300, bbox_inches='tight', facecolor='#0f172a')
    plt.close()

    print("\n[TAMAM] Model degerlendirildi.")
    print("\nKarmasiklik Matrisi:")
    print(cm)
    print("\nSiniflandirma Raporu:")
    print(report)
    print(f"Dogruluk: {accuracy}")
    print(f"Kesinlik: {precision}")
    print(f"Duyarlilik: {recall}")
    print(f"F1-skoru: {f1}")
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
