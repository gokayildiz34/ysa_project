"""
Autoencoder mimarisi — sadece model tanımı.
"""

from tensorflow.keras.layers import Dense, Dropout, Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam


def build_autoencoder(input_dim, hidden_layers, bottleneck_dim, dropout_rate, learning_rate):
    """Autoencoder mimarisini oluşturur ve derler."""
    # Encoder
    input_layer = Input(shape=(input_dim,))
    x = input_layer

    for units in hidden_layers:
        x = Dense(units, activation="relu")(x)
        if dropout_rate > 0:
            x = Dropout(dropout_rate)(x)

    # Bottleneck
    bottleneck = Dense(bottleneck_dim, activation="relu")(x)

    # Decoder
    for units in reversed(hidden_layers):
        x = Dense(units, activation="relu")(bottleneck if units == hidden_layers[-1] else x)

    # Output
    output_layer = Dense(input_dim, activation="sigmoid")(x)

    model = Model(inputs=input_layer, outputs=output_layer)
    model.compile(optimizer=Adam(learning_rate=learning_rate), loss="mse")
    return model
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
