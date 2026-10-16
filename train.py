
from pathlib import Path
import json
import joblib
import mlflow

from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split

ARTIFACTS = Path("artifacts")
ARTIFACTS.mkdir(exist_ok=True)

iris = load_iris(as_frame=True)
X = iris.data
y = iris.target

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

mlflow.set_experiment("ml-deployment-platform")

best_model = None
best_accuracy = -1
best_n_estimators = None

for n_estimators in [50, 100, 200]:
    model = RandomForestClassifier(
        n_estimators=n_estimators,
        random_state=42
    )
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    f1 = f1_score(y_test, predictions, average="weighted")

    with mlflow.start_run(run_name=f"RandomForest-{n_estimators}"):
        mlflow.log_param("model", "RandomForestClassifier")
        mlflow.log_param("n_estimators", n_estimators)
        mlflow.log_param("random_state", 42)
        mlflow.log_metric("accuracy", float(accuracy))
        mlflow.log_metric("weighted_f1", float(f1))

    print(
        f"Trees: {n_estimators} | "
        f"Accuracy: {accuracy:.2%} | "
        f"Weighted F1: {f1:.4f}"
    )

    if accuracy > best_accuracy:
        best_accuracy = accuracy
        best_model = model
        best_n_estimators = n_estimators

joblib.dump(
    {
        "model": best_model,
        "feature_names": list(X.columns),
        "target_names": list(iris.target_names)
    },
    ARTIFACTS / "model.joblib"
)

metrics = {
    "accuracy": round(float(best_accuracy), 4),
    "n_estimators": best_n_estimators,
    "test_samples": len(y_test)
}

(ARTIFACTS / "metrics.json").write_text(
    json.dumps(metrics, indent=2),
    encoding="utf-8"
)

print("\nBest model saved successfully!")
print(f"Best tree count: {best_n_estimators}")
print(f"Best accuracy: {best_accuracy:.2%}")
