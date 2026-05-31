

import sqlite3
from pathlib import Path

# Proje kök dizinini ve veritabanı konumunu belirliyoruz
BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data"
DB_PATH = DATA_DIR / "netanomai.sqlite3"


def get_connection():
    # data klasörü yoksa oluşturuyoruz, sonra bağlantıyı açıyoruz
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    # Row nesneleri sözlük gibi erişilebilsin diye bu ayarı yapıyoruz
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    with get_connection() as conn:
        # Ana analiz tablosunu oluşturuyoruz — her pcap analizi buraya kaydedilecek
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS analyses (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                file_name TEXT NOT NULL,
                stored_path TEXT,
                source TEXT NOT NULL,
                model_type TEXT NOT NULL DEFAULT 'autoencoder',
                created_at TEXT NOT NULL,
                result TEXT NOT NULL,
                total_windows INTEGER NOT NULL,
                anomaly_windows INTEGER NOT NULL,
                anomaly_ratio REAL NOT NULL,
                avg_error REAL NOT NULL,
                max_error REAL NOT NULL,
                threshold REAL NOT NULL,
                anomaly_ratio_threshold REAL NOT NULL,
                estimated_start_window INTEGER,
                estimated_end_window INTEGER
            )
            """
        )
        # Her analize ait pencere detaylarını ayrı tabloda saklıyoruz
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS analysis_windows (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                analysis_id INTEGER NOT NULL,
                window_id INTEGER NOT NULL,
                reconstruction_error REAL NOT NULL,
                is_anomaly INTEGER NOT NULL,
                FOREIGN KEY (analysis_id) REFERENCES analyses(id) ON DELETE CASCADE
            )
            """
        )
        # Eski veritabanlarında model_type kolonu olmayabilir, varsa hata vermeden geç
        try:
            conn.execute("ALTER TABLE analyses ADD COLUMN model_type TEXT NOT NULL DEFAULT 'autoencoder'")
        except Exception:
            pass  # Kolon zaten varsa bir şey yapmıyoruz
        conn.commit()


def create_analysis(summary, windows):
    # Yeni bir analiz kaydı oluşturup ID'sini döndürüyoruz
    with get_connection() as conn:
        cursor = conn.execute(
            """
            INSERT INTO analyses (
                file_name,
                stored_path,
                source,
                model_type,
                created_at,
                result,
                total_windows,
                anomaly_windows,
                anomaly_ratio,
                avg_error,
                max_error,
                threshold,
                anomaly_ratio_threshold,
                estimated_start_window,
                estimated_end_window
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                summary["file_name"],
                summary.get("stored_path"),
                summary["source"],
                summary.get("model_type", "autoencoder"),
                summary["created_at"],
                summary["result"],
                summary["total_windows"],
                summary["anomaly_windows"],
                summary["anomaly_ratio"],
                summary["avg_error"],
                summary["max_error"],
                summary["threshold"],
                summary["anomaly_ratio_threshold"],
                summary.get("estimated_start_window"),
                summary.get("estimated_end_window"),
            ),
        )
        analysis_id = cursor.lastrowid

        # Pencere detaylarını tek seferde toplu ekliyoruz (daha hızlı)
        conn.executemany(
            """
            INSERT INTO analysis_windows (
                analysis_id,
                window_id,
                reconstruction_error,
                is_anomaly
            )
            VALUES (?, ?, ?, ?)
            """,
            [
                (
                    analysis_id,
                    item["window_id"],
                    item["reconstruction_error"],
                    1 if item["is_anomaly"] else 0,
                )
                for item in windows
            ],
        )
        conn.commit()

    return analysis_id


def list_analyses(limit=50):
    # En son yapılan analizleri önce göstermek için ID'ye göre azalan sırada çekiyoruz
    with get_connection() as conn:
        rows = conn.execute(
            """
            SELECT *
            FROM analyses
            ORDER BY id DESC
            LIMIT ?
            """,
            (limit,),
        ).fetchall()
    return [dict(row) for row in rows]


def get_analysis(analysis_id):
    # Belirli bir analizin tüm detaylarını ve pencere verilerini getiriyoruz
    with get_connection() as conn:
        analysis = conn.execute(
            "SELECT * FROM analyses WHERE id = ?",
            (analysis_id,),
        ).fetchone()
        if analysis is None:
            return None

        # O analize ait tüm pencereleri sıralı şekilde alıyoruz
        windows = conn.execute(
            """
            SELECT window_id, reconstruction_error, is_anomaly
            FROM analysis_windows
            WHERE analysis_id = ?
            ORDER BY window_id ASC
            """,
            (analysis_id,),
        ).fetchall()

    # Sonucu düzgün bir sözlük yapısına çevirip döndürüyoruz
    result = dict(analysis)
    result["windows"] = [
        {
            "window_id": row["window_id"],
            "reconstruction_error": row["reconstruction_error"],
            "is_anomaly": bool(row["is_anomaly"]),
        }
        for row in windows
    ]
    return result
