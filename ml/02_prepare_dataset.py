import os

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

PROCESSED_DIR = "data/processed"
SPLIT_DIR = "data/splits"

TRAIN_DIR = os.path.join(SPLIT_DIR, "train")
VAL_DIR = os.path.join(SPLIT_DIR, "validation")
TEST_DIR = os.path.join(SPLIT_DIR, "test")

FEATURES_CSV = os.path.join(PROCESSED_DIR, "features.csv")

TRAIN_CSV = os.path.join(TRAIN_DIR, "train_normal.csv")
VAL_CSV = os.path.join(VAL_DIR, "validation_normal.csv")
TEST_NORMAL_CSV = os.path.join(TEST_DIR, "test_normal.csv")
TEST_ANOMALY_CSV = os.path.join(TEST_DIR, "test_anomaly.csv")
TEST_ALL_CSV = os.path.join(TEST_DIR, "test_all.csv")

SPLIT_SUMMARY_CSV = os.path.join(SPLIT_DIR, "split_summary.csv")

RANDOM_STATE = 42


def save_split_summary(train_normal, validation_normal, test_normal, anomaly_df):
    summaries = []

    for split_name, label_type, split_df in [
        ("train", "normal", train_normal),
        ("validation", "normal", validation_normal),
        ("test", "normal", test_normal),
        ("test", "anomaly", anomaly_df),
    ]:
        if split_df.empty:
            continue

        summary = split_df.groupby("file_name").size().reset_index(name="row_count")
        summary["split"] = split_name
        summary["label_type"] = label_type
        summaries.append(summary)

    if len(summaries) == 0:
        pd.DataFrame(columns=["split", "label_type", "file_name", "row_count"]).to_csv(SPLIT_SUMMARY_CSV, index=False)
        return

    summary_df = pd.concat(summaries, ignore_index=True)
    summary_df = summary_df[["split", "label_type", "file_name", "row_count"]]
    summary_df.to_csv(SPLIT_SUMMARY_CSV, index=False)


def split_normal_by_file(normal_df):
    normal_files = sorted(normal_df["file_name"].unique())

    if len(normal_files) < 3:
        raise ValueError("Dosya bazli train/validation/test ayrimi icin en az 3 normal pcap dosyasi gerekli.")

    train_files, temp_files = train_test_split(
        normal_files,
        test_size=0.30,
        random_state=RANDOM_STATE,
        shuffle=True,
    )

    validation_files, test_files = train_test_split(
        temp_files,
        test_size=0.50,
        random_state=RANDOM_STATE,
        shuffle=True,
    )

    train_normal = normal_df[normal_df["file_name"].isin(train_files)].copy()
    validation_normal = normal_df[normal_df["file_name"].isin(validation_files)].copy()
    test_normal = normal_df[normal_df["file_name"].isin(test_files)].copy()

    return train_normal, validation_normal, test_normal


def main():
    os.makedirs(PROCESSED_DIR, exist_ok=True)
    os.makedirs(TRAIN_DIR, exist_ok=True)
    os.makedirs(VAL_DIR, exist_ok=True)
    os.makedirs(TEST_DIR, exist_ok=True)

    if not os.path.exists(FEATURES_CSV):
        print("[HATA] features.csv bulunamadi. Once 01_extract_features.py calistir.")
        return

    df = pd.read_csv(FEATURES_CSV)

    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna()
    df = df[df["packet_count"] > 0]
    df = df[df["byte_count"] > 0]

    normal_df = df[df["label"] == 0].copy()
    anomaly_df = df[df["label"] == 1].copy()

    if len(normal_df) < 10:
        print("[HATA] Normal veri cok az. Daha fazla normal pcapng ekle.")
        return

    if len(anomaly_df) == 0:
        print("[UYARI] Anomali verisi bulunamadi. Test anomaly dosyasi bos olabilir.")

    try:
        train_normal, validation_normal, test_normal = split_normal_by_file(normal_df)
    except ValueError as e:
        print(f"[HATA] {e}")
        return

    test_all = pd.concat([test_normal, anomaly_df], ignore_index=True)

    train_normal.to_csv(TRAIN_CSV, index=False)
    validation_normal.to_csv(VAL_CSV, index=False)
    test_normal.to_csv(TEST_NORMAL_CSV, index=False)
    anomaly_df.to_csv(TEST_ANOMALY_CSV, index=False)
    test_all.to_csv(TEST_ALL_CSV, index=False)

    save_split_summary(train_normal, validation_normal, test_normal, anomaly_df)

    print("\n[TAMAM] Veri seti dosya bazli ayrildi.")
    print("\nAyirma yapisi:")
    print("Train       : Normal dosyalarin yaklasik %70'i")
    print("Validation  : Normal dosyalarin yaklasik %15'i")
    print("Test Normal : Normal dosyalarin yaklasik %15'i")
    print("Test Anomali: Tum anomali dosyalari")
    print("\nNot: Model egitimi ve validation sadece normal veri kullanir.")

    print(f"\nTrain normal      : {len(train_normal)} satir / {train_normal['file_name'].nunique()} dosya")
    print(f"Validation normal : {len(validation_normal)} satir / {validation_normal['file_name'].nunique()} dosya")
    print(f"Test normal       : {len(test_normal)} satir / {test_normal['file_name'].nunique()} dosya")
    print(f"Test anomaly      : {len(anomaly_df)} satir / {anomaly_df['file_name'].nunique()} dosya")
    print(f"Test toplam       : {len(test_all)} satir")

    print("\nKaydedilen dosyalar:")
    print(f"- {TRAIN_CSV}")
    print(f"- {VAL_CSV}")
    print(f"- {TEST_NORMAL_CSV}")
    print(f"- {TEST_ANOMALY_CSV}")
    print(f"- {TEST_ALL_CSV}")
    print(f"- {SPLIT_SUMMARY_CSV}")


if __name__ == "__main__":
    main()
