import csv
import os
from datetime import datetime

from pathlib import Path
from time import perf_counter
from threading import Lock
from collections import Counter

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
        "model_loaded": model is not None,
        "model_type": type(model).__name__ if model is not None else None,
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

        prediction = target_names[prediction_id]
        confidence = round(float(max(probabilities)), 4)

        # Save successful predictions to CSV
        history_file = "prediction_history.csv"
        file_exists = os.path.isfile(history_file)

        with open(history_file, "a", newline="", encoding="utf-8") as file:
            writer = csv.writer(file)

            if not file_exists:
                writer.writerow([
                    "timestamp", "prediction", "class_id", "confidence"
                ])

            writer.writerow([
                datetime.now().isoformat(timespec="seconds"),
                prediction,
                prediction_id,
                confidence
            ])

        success = True

        return {
            "prediction": prediction,
            "class_id": prediction_id,
            "confidence": confidence
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


@app.get("/history")
def prediction_history():
    history_file = "prediction_history.csv"

    if not os.path.isfile(history_file):
        return []

    with open(history_file, "r", newline="", encoding="utf-8") as file:
        return list(csv.DictReader(file))


@app.get("/monitoring")
def monitoring():
    history_file = "prediction_history.csv"

    if not os.path.isfile(history_file):
        return {
            "total_predictions": 0,
            "distribution": {},
            "average_confidence": 0.0,
            "confidence_trend": [],
            "drift_detected": False,
            "alert": "Not enough prediction history."
        }

    with open(history_file, "r", newline="", encoding="utf-8") as file:
        records = list(csv.DictReader(file))

    valid_records = []
    for record in records:
        try:
            valid_records.append({
                "timestamp": record["timestamp"],
                "prediction": record["prediction"],
                "confidence": float(record["confidence"])
            })
        except (KeyError, ValueError):
            continue

    total = len(valid_records)
    distribution = dict(
        Counter(record["prediction"] for record in valid_records)
    )

    average_confidence = (
        sum(r["confidence"] for r in valid_records) / total
        if total else 0.0
    )

    # Compare the latest 20 predictions against the preceding 20.
    window_size = 20
    drift_detected = False
    alert = "Collecting prediction history."

    if total >= window_size * 2:
        baseline = valid_records[-2 * window_size:-window_size]
        recent = valid_records[-window_size:]

        baseline_counts = Counter(r["prediction"] for r in baseline)
        recent_counts = Counter(r["prediction"] for r in recent)

        classes = set(baseline_counts) | set(recent_counts)

        # Total variation distance between prediction distributions.
        distance = 0.5 * sum(
            abs(
                baseline_counts[c] / window_size
                - recent_counts[c] / window_size
            )
            for c in classes
        )

        drift_detected = distance >= 0.35

        alert = (
            f"Possible prediction distribution shift detected "
            f"(distance: {distance:.2f})."
            if drift_detected
            else f"No major distribution shift detected (distance: {distance:.2f})."
        )

    elif total >= window_size:
        alert = "Collecting another 20 predictions for comparison."

    confidence_trend = [
        {
            "timestamp": r["timestamp"],
            "confidence": r["confidence"]
        }
        for r in valid_records[-50:]
    ]

    return {
        "total_predictions": total,
        "distribution": distribution,
        "average_confidence": round(average_confidence, 4),
        "confidence_trend": confidence_trend,
        "drift_detected": drift_detected,
        "alert": alert
    }