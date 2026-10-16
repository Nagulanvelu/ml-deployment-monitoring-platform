
from pathlib import Path
import json
import joblib
import mlflow

from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

ARTIFACTS = Path("artifacts")
ARTIFACTS.mkdir(exist_ok=True)

iris = load_iris(as_frame=True)
X = iris.data
y = iris.target

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

predictions = model.predict(X_test)
accuracy = accuracy_score(y_test, predictions)

mlflow.set_experiment("ml-deployment-platform")

with mlflow.start_run():
    mlflow.log_param("model", "RandomForestClassifier")
    mlflow.log_param("n_estimators", 100)
    mlflow.log_param("random_state", 42)
    mlflow.log_metric("accuracy", float(accuracy))
    mlflow.log_metric("test_samples", len(y_test))

joblib.dump(
    {
        "model": model,
        "feature_names": list(X.columns),
        "target_names": list(iris.target_names)
    },
    ARTIFACTS / "model.joblib"
)

metrics = {
    "accuracy": round(float(accuracy), 4),
    "test_samples": len(y_test)
}

(ARTIFACTS / "metrics.json").write_text(
    json.dumps(metrics, indent=2), encoding="utf-8"
)

print("Model training completed!")
print(f"Accuracy: {accuracy:.2%}")
print(f"Test samples: {len(y_test)}")
print("Model saved in artifacts/model.joblib")
