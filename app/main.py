from pathlib import Path
from time import perf_counter
from threading import Lock

import joblib
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = ROOT / "artifacts" / "model.joblib"

app = FastAPI(
    title="ML Deployment & Monitoring Platform",
    description="Machine learning prediction API with basic monitoring",
    version="1.0.0"
)

model_bundle = joblib.load(MODEL_PATH)
model = model_bundle["model"]
feature_names = model_bundle["feature_names"]
target_names = model_bundle["target_names"]

lock = Lock()
request_count = 0
error_count = 0
total_latency_ms = 0.0


class PredictionRequest(BaseModel):
    features: dict[str, float] = Field(
        ...,
        description="Feature names and numeric values"
    )


@app.get("/")
def home():
    return {
        "message": "ML Deployment & Monitoring Platform",
        "docs": "/docs"
    }


@app.get("/health")
def health():
    return {
        "status": "healthy",
        "model_loaded": model is not None
    }


@app.post("/predict")
def predict(request: PredictionRequest):
    global request_count, error_count, total_latency_ms

    start = perf_counter()
    success = False

    try:
        missing = [
            name for name in feature_names
            if name not in request.features
        ]
        extra = [
            name for name in request.features
            if name not in feature_names
        ]

        if missing or extra:
            raise HTTPException(
                status_code=422,
                detail={
                    "missing_features": missing,
                    "unexpected_features": extra
                }
            )

        values = [[request.features[name] for name in feature_names]]
        prediction_id = int(model.predict(values)[0])
        probabilities = model.predict_proba(values)[0]

        success = True

        return {
            "prediction": target_names[prediction_id],
            "class_id": prediction_id,
            "confidence": round(float(max(probabilities)), 4)
        }

    finally:
        elapsed = (perf_counter() - start) * 1000

        with lock:
            request_count += 1
            total_latency_ms += elapsed

            if not success:
                error_count += 1


@app.get("/metrics")
def metrics():
    with lock:
        average_latency = (
            total_latency_ms / request_count
            if request_count else 0.0
        )

        return {
            "prediction_requests_total": request_count,
            "prediction_errors_total": error_count,
            "average_prediction_latency_ms": round(
                average_latency, 3
            )
        }