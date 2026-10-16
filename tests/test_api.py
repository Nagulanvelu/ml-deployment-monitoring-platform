

from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)


def test_home():
    response = client.get("/")
    assert response.status_code == 200


def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["model_loaded"] is True


def test_prediction():
    response = client.post(
        "/predict",
        json={
            "features": {
                "sepal length (cm)": 5.1,
                "sepal width (cm)": 3.5,
                "petal length (cm)": 1.4,
                "petal width (cm)": 0.2
            }
        }
    )
    assert response.status_code == 200
    assert response.json()["prediction"] in [
        "setosa", "versicolor", "virginica"
    ]


def test_invalid_features():
    response = client.post(
        "/predict",
        json={"features": {"sepal length (cm)": 5.1}}
    )
    assert response.status_code == 422


def test_monitoring_metrics():
    response = client.get("/metrics")
    assert response.status_code == 200
    assert "prediction_requests_total" in response.json()

