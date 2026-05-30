

import argparse
import json
import os
import random

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.layers import Dense, Dropout, Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

# Dosya yollarını burada sabit tanımlıyoruz
SPLIT_DIR = "data/splits"
MODEL_DIR = "models"
OUTPUT_DIR = "outputs"

TRAIN_CSV = os.path.join(SPLIT_DIR, "train", "train_normal.csv")
VAL_CSV = os.path.join(SPLIT_DIR, "validation", "validation_normal.csv")

MODEL_PATH = os.path.join(MODEL_DIR, "autoencoder.keras")
SCALER_PATH = os.path.join(MODEL_DIR, "scaler.pkl")
THRESHOLD_PATH = os.path.join(MODEL_DIR, "threshold.txt")
FEATURE_COLUMNS_PATH = os.path.join(MODEL_DIR, "feature_columns.pkl")
BEST_CONFIG_PATH = os.path.join(MODEL_DIR, "best_config.json")

# Grafik ve çıktı dosyaları
LOSS_CURVE_PATH = os.path.join(OUTPUT_DIR, "loss_curve.png")
TUNING_RESULTS_CSV = os.path.join(OUTPUT_DIR, "tuning_results.csv")
VALIDATION_ERRORS_CSV = os.path.join(OUTPUT_DIR, "validation_errors.csv")

# Eğitimde kullanmayacağımız sütunlar
DROP_COLUMNS = ["file_name", "window_id", "label"]

# Sonuçların tekrar üretilebilir olması için sabit tohum kullanıyoruz
RANDOM_STATE = 42


def set_seed(seed):
    # Hem NumPy hem TensorFlow hem de Python rastgele sayı üreticisini sabitliyoruz
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def build_autoencoder(input_dim, hidden_layers, bottleneck_dim, dropout_rate, learning_rate):
    # Encoder: giriş verisini darboğaza sıkıştırıyor
    input_layer = Input(shape=(input_dim,))
    x = input_layer

    for units in hidden_layers:
        x = Dense(units, activation="relu")(x)
        if dropout_rate > 0:
            x = Dropout(dropout_rate)(x)  # Aşırı öğrenmeyi azaltmak için dropout

    # Darboğaz katmanı — burada en az boyuta iniyoruz
    bottleneck = Dense(bottleneck_dim, activation="relu")(x)

    # Decoder: darboğazdan orijinal boyuta geri açıyoruz
    for units in reversed(hidden_layers):
        x = Dense(units, activation="relu")(bottleneck if units == hidden_layers[-1] else x)

    # Çıkış katmanı — sigmoid ile 0-1 arasına normalize edilmiş değer üretiyoruz
    output_layer = Dense(input_dim, activation="sigmoid")(x)

    model = Model(inputs=input_layer, outputs=output_layer)
    # Kayıp fonksiyonu olarak MSE kullanıyoruz — yeniden yapılandırma hatasını minimize ediyoruz
    model.compile(optimizer=Adam(learning_rate=learning_rate), loss="mse")

    return model


def make_scaler(name):
    # İstenen ölçekleyiciyi ismine göre döndürüyoruz
    if name == "minmax":
        return MinMaxScaler()
    if name == "standard":
        return StandardScaler()
    if name == "robust":
        return RobustScaler()
    raise ValueError(f"Bilinmeyen scaler: {name}")


def get_search_space():
    # Deneyeceğimiz farklı mimari ve hiperparametre kombinasyonları
    return [
        {
            "hidden_layers": [64, 32, 16],
            "bottleneck_dim": 8,
            "dropout_rate": 0.10,
            "learning_rate": 0.001,
            "batch_size": 32,
            "scaler": "minmax",
        },
        {
            "hidden_layers": [128, 64, 32],
            "bottleneck_dim": 16,
            "dropout_rate": 0.10,
            "learning_rate": 0.001,
            "batch_size": 32,
            "scaler": "minmax",
        },
        {
            "hidden_layers": [64, 32],
            "bottleneck_dim": 8,
            "dropout_rate": 0.20,
            "learning_rate": 0.0005,
            "batch_size": 32,
            "scaler": "minmax",
        },
        {
            "hidden_layers": [128, 64, 32],
            "bottleneck_dim": 8,
            "dropout_rate": 0.20,
            "learning_rate": 0.0005,
            "batch_size": 64,
            "scaler": "minmax",
        },
        {
            "hidden_layers": [64, 32, 16],
            "bottleneck_dim": 8,
            "dropout_rate": 0.10,
            "learning_rate": 0.001,
            "batch_size": 32,
            "scaler": "robust",
        },
    ]


def train_one_config(config, X_train, X_val, epochs, patience):
    # Veriyi ölçeklendiriyoruz — sadece train verisine fit edip val'e uyguluyoruz
    scaler = make_scaler(config["scaler"])
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)

    model = build_autoencoder(
        input_dim=X_train_scaled.shape[1],
        hidden_layers=config["hidden_layers"],
        bottleneck_dim=config["bottleneck_dim"],
        dropout_rate=config["dropout_rate"],
        learning_rate=config["learning_rate"],
    )

    callbacks = [
        # Validation kaybı iyileşmiyorsa eğitimi durdurup en iyi ağırlıkları geri yükle
        EarlyStopping(
            monitor="val_loss",
            patience=patience,
            restore_best_weights=True,
            verbose=0,
        ),
        # Öğrenme hızını otomatik düşür — platoda takılıp kalmayı önler
        ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=max(3, patience // 2),
            min_lr=1e-6,
            verbose=0,
        ),
    ]

    # Autoencoder giriş = çıkış olarak eğitiliyor (kendini yeniden oluşturmayı öğreniyor)
    history = model.fit(
        X_train_scaled,
        X_train_scaled,
        epochs=epochs,
        batch_size=config["batch_size"],
        validation_data=(X_val_scaled, X_val_scaled),
        callbacks=callbacks,
        verbose=0,
    )

    best_val_loss = float(min(history.history["val_loss"]))
    final_train_loss = float(history.history["loss"][-1])

    return model, scaler, history, X_val_scaled, best_val_loss, final_train_loss


def parse_args():
    parser = argparse.ArgumentParser(description="Normal veri ile autoencoder egitir ve hiperparametre aramasi yapar.")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--threshold-percentile", type=float, default=95.0)
    return parser.parse_args()


def main():
    args = parse_args()
    set_seed(RANDOM_STATE)

    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    if not os.path.exists(TRAIN_CSV) or not os.path.exists(VAL_CSV):
        print("[HATA] Train/validation dosyasi yok. Once 02_prepare_dataset.py calistir.")
        return

    train_df = pd.read_csv(TRAIN_CSV)
    val_df = pd.read_csv(VAL_CSV)

    # Eğitimde kullanılmayacak sütunları çıkarıyoruz
    feature_columns = [col for col in train_df.columns if col not in DROP_COLUMNS]

    X_train = train_df[feature_columns].values
    X_val = val_df[feature_columns].values

    best = None
    tuning_rows = []

    print("\n[INFO] Hiperparametre aramasi basladi.")
    print("[INFO] Egitim ve validation sadece normal veri ile yapiliyor.")

    # Her konfigürasyonu sırayla deniyoruz, en iyi validation kaybını kayıt altına alıyoruz
    for idx, config in enumerate(get_search_space(), start=1):
        set_seed(RANDOM_STATE + idx)
        print(f"[DENEME {idx}] {config}")

        model, scaler, history, X_val_scaled, best_val_loss, final_train_loss = train_one_config(
            config=config,
            X_train=X_train,
            X_val=X_val,
            epochs=args.epochs,
            patience=args.patience,
        )

        tuning_row = {
            "trial": idx,
            "best_val_loss": best_val_loss,
            "final_train_loss": final_train_loss,
            "epochs_ran": len(history.history["loss"]),
            **config,
        }
        tuning_rows.append(tuning_row)

        if best is None or best_val_loss < best["best_val_loss"]:
            best = {
                "trial": idx,
                "config": config,
                "model": model,
                "scaler": scaler,
                "history": history,
                "X_val_scaled": X_val_scaled,
                "best_val_loss": best_val_loss,
                "final_train_loss": final_train_loss,
            }

    tuning_df = pd.DataFrame(tuning_rows)
    tuning_df.to_csv(TUNING_RESULTS_CSV, index=False)

    # En iyi modelin validation hatasından eşik değerini belirliyoruz
    # 95. yüzdelik dilim — normalin üstüne çıkan hataları anomali sayıyoruz
    val_reconstructed = best["model"].predict(best["X_val_scaled"], verbose=0)
    val_errors = np.mean(np.square(best["X_val_scaled"] - val_reconstructed), axis=1)
    threshold = float(np.percentile(val_errors, args.threshold_percentile))

    # Modeli ve yardımcı nesneleri diske kaydediyoruz
    best["model"].save(MODEL_PATH)
    joblib.dump(best["scaler"], SCALER_PATH)
    joblib.dump(feature_columns, FEATURE_COLUMNS_PATH)

    best_config = {
        "trial": best["trial"],
        "config": best["config"],
        "best_val_loss": best["best_val_loss"],
        "final_train_loss": best["final_train_loss"],
        "threshold_percentile": args.threshold_percentile,
        "threshold": threshold,
        "train_rows": int(len(train_df)),
        "validation_rows": int(len(val_df)),
        "feature_count": int(len(feature_columns)),
    }

    with open(BEST_CONFIG_PATH, "w", encoding="utf-8") as f:
        json.dump(best_config, f, indent=4)

    with open(THRESHOLD_PATH, "w", encoding="utf-8") as f:
        f.write(str(threshold))

    pd.DataFrame({"reconstruction_error": val_errors, "threshold": threshold}).to_csv(
        VALIDATION_ERRORS_CSV,
        index=False,
    )

    # Eğitim sürecini görselleştirmek için kayıp eğrisini çiziyoruz
    plt.figure()
    plt.plot(best["history"].history["loss"], label="Eğitim Kaybı")
    plt.plot(best["history"].history["val_loss"], label="Doğrulama Kaybı")
    plt.xlabel("Epoch")
    plt.ylabel("MSE Kaybı")
    plt.title("En İyi Autoencoder Kayıp Eğrisi")
    plt.legend()
    plt.savefig(LOSS_CURVE_PATH, dpi=300, bbox_inches="tight")
    plt.close()

    print("\n[TAMAM] Model egitildi.")
    print(f"En iyi deneme: {best['trial']}")
    print(f"En iyi config: {best['config']}")
    print(f"Feature sayisi: {len(feature_columns)}")
    print(f"Threshold percentile: {args.threshold_percentile}")
    print(f"Threshold: {threshold}")
    print(f"Model kaydedildi: {MODEL_PATH}")
    print(f"Tuning sonuclari: {TUNING_RESULTS_CSV}")
    print(f"Best config: {BEST_CONFIG_PATH}")
    print(f"Loss grafigi: {LOSS_CURVE_PATH}")


if __name__ == "__main__":
    main()
