
import json
import shutil
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from backend.database import create_analysis, get_analysis, init_db, list_analyses
from backend.predictor import (
    DEFAULT_ANOMALY_RATIO_THRESHOLD,
    DEFAULT_MODEL_TYPE,
    DEFAULT_WINDOW_SIZE,
    MODEL_REGISTRY,
    BASE_DIR,
    analyze_pcap,
    get_available_models,
    get_model_comparison,
    get_model_info,
)

UPLOAD_DIR         = BASE_DIR / "data" / "uploads"
REALISTIC_TEST_DIR = BASE_DIR / "data" / "raw" / "realistic_test"
FRONTEND_DIR       = BASE_DIR / "frontend"


@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db()
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    REALISTIC_TEST_DIR.mkdir(parents=True, exist_ok=True)
    yield


app = FastAPI(title="NetAnomAI API", version="2.0.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)




if FRONTEND_DIR.exists():
    app.mount("/assets", StaticFiles(directory=FRONTEND_DIR / "assets"), name="assets")

OUTPUTS_DIR = BASE_DIR / "outputs"
if OUTPUTS_DIR.exists():
    app.mount("/outputs", StaticFiles(directory=OUTPUTS_DIR), name="outputs")


@app.get("/")
def index():
    index_path = FRONTEND_DIR / "index.html"
    if index_path.exists():
        return FileResponse(index_path)
    return {"message": "NetAnomAI API calisiyor."}


@app.get("/api/health")
def health():
    return {"status": "ok"}


# ── Model endpoints ───────────────────────────────────────────────────────────

@app.get("/api/model-info")
def model_info():
    try:
        return get_model_info()
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@app.get("/api/available-models")
def available_models():
    """Egitilmis modellerin listesini dondurur."""
    try:
        return get_available_models()
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@app.get("/api/model-comparison")
def model_comparison():
    """4 modelin karsilastirma metriklerini dondurur (04b calistirildiktan sonra)."""
    try:
        return get_model_comparison()
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@app.get("/api/model-metrics/{model_name}")
def model_metrics(model_name: str):
    if model_name not in MODEL_REGISTRY:
        raise HTTPException(status_code=404, detail=f"Model bulunamadi: {model_name}")
    metrics_path = BASE_DIR / "outputs" / model_name / "metrics.json"
    if not metrics_path.exists():
        raise HTTPException(
            status_code=404,
            detail=f"outputs/{model_name}/metrics.json bulunamadi."
        )
    with open(metrics_path, encoding="utf-8") as f:
        return json.load(f)


# ── Analiz endpoints ──────────────────────────────────────────────────────────

@app.post("/api/analyze-upload")
def analyze_upload(
    file: UploadFile = File(...),
    anomaly_ratio_threshold: float = Query(DEFAULT_ANOMALY_RATIO_THRESHOLD, ge=0, le=1),
    window_size: float = Query(DEFAULT_WINDOW_SIZE, gt=0),
    model_type: str = Query(DEFAULT_MODEL_TYPE),
):
    if not file.filename.lower().endswith((".pcap", ".pcapng")):
        raise HTTPException(status_code=400, detail="Sadece .pcap veya .pcapng dosyasi yuklenebilir.")

    destination = UPLOAD_DIR / Path(file.filename).name
    with destination.open("wb") as out_file:
        shutil.copyfileobj(file.file, out_file)

    try:
        summary, windows = analyze_pcap(
            destination,
            source="upload",
            anomaly_ratio_threshold=anomaly_ratio_threshold,
            window_size=window_size,
            model_type=model_type,
        )
        analysis_id = create_analysis(summary, windows)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))

    return {"analysis_id": analysis_id, **summary, "windows": windows}


@app.get("/api/realistic-files")
def realistic_files():
    files = []
    for path in sorted(REALISTIC_TEST_DIR.glob("*")):
        if path.is_file() and path.suffix.lower() in [".pcap", ".pcapng"]:
            files.append({
                "file_name":  path.name,
                "path":       str(path),
                "size_bytes": path.stat().st_size,
            })
    return {"files": files}


@app.post("/api/analyze-realistic/{file_name}")
def analyze_realistic(
    file_name: str,
    anomaly_ratio_threshold: float = Query(DEFAULT_ANOMALY_RATIO_THRESHOLD, ge=0, le=1),
    window_size: float = Query(DEFAULT_WINDOW_SIZE, gt=0),
    model_type: str = Query(DEFAULT_MODEL_TYPE),
):
    path = REALISTIC_TEST_DIR / Path(file_name).name
    if not path.exists() or path.suffix.lower() not in [".pcap", ".pcapng"]:
        raise HTTPException(status_code=404, detail="realistic_test dosyasi bulunamadi.")

    try:
        summary, windows = analyze_pcap(
            path,
            source="realistic_test",
            anomaly_ratio_threshold=anomaly_ratio_threshold,
            window_size=window_size,
            model_type=model_type,
        )
        analysis_id = create_analysis(summary, windows)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))

    return {"analysis_id": analysis_id, **summary, "windows": windows}


# ── History endpoints ─────────────────────────────────────────────────────────

@app.get("/api/analyses")
def analyses(limit: int = Query(50, ge=1, le=200)):
    return {"items": list_analyses(limit=limit)}


@app.get("/api/analyses/{analysis_id}")
def analysis_detail(analysis_id: int):
    analysis = get_analysis(analysis_id)
    if analysis is None:
        raise HTTPException(status_code=404, detail="Analiz bulunamadi.")
    return analysis
