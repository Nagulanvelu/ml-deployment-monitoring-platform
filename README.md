# 🚀 End-to-End ML Deployment & Monitoring Platform

<p align="center">
  <strong>Train • Track • Deploy • Monitor</strong>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.13-blue?logo=python" alt="Python">
  <img src="https://img.shields.io/badge/FastAPI-API-009688?logo=fastapi" alt="FastAPI">
  <img src="https://img.shields.io/badge/Docker-Containerized-2496ED?logo=docker" alt="Docker">
  <img src="https://img.shields.io/badge/MLflow-Experiment%20Tracking-0194E2" alt="MLflow">
  <a href="https://github.com/Nagulanvelu/ml-deployment-monitoring-platform/actions/workflows/ci.yml">
    <img src="https://github.com/Nagulanvelu/ml-deployment-monitoring-platform/actions/workflows/ci.yml/badge.svg?branch=main" alt="CI Status">
  </a>
</p>

## 📌 Overview

An end-to-end machine learning project that trains a Random Forest classifier, tracks experiments using MLflow, serves predictions through FastAPI, monitors API performance, and automates testing with GitHub Actions.

## ✨ Features

- 🤖 Machine learning model training with scikit-learn
- 📊 Model evaluation and accuracy reporting
- 🔬 MLflow experiment tracking
- ⚡ REST API for real-time predictions
- ❤️ API health-check endpoint
- 📈 Request counts, error counts, and latency monitoring
- 🐳 Docker containerization
- 🧪 Automated API tests using pytest
- 🔄 Continuous integration using GitHub Actions

## 🛠️ Tech Stack

| Technology | Purpose |
|---|---|
| Python | Core programming language |
| scikit-learn | Machine learning |
| Pandas | Data handling |
| FastAPI | REST API development |
| Uvicorn | API server |
| MLflow | Experiment tracking |
| Docker | Containerization |
| pytest | Automated testing |
| GitHub Actions | CI pipeline |

## 🏗️ Architecture

```text
Iris Dataset
     |
     v
Model Training (train.py)
     |
     +----> Model Evaluation
     |
     +----> MLflow Tracking
     |
     v
Saved Model Artifact
     |
     v
FastAPI Prediction Service
     |
     +----> /predict
     +----> /health
     +----> /metrics
     |
     v
Docker Container
     |
     v
GitHub Actions CI
```

## 📂 Project Structure

```text
ml-deployment-monitoring-platform/
├── app/
│   ├── __init__.py
│   └── main.py
├── artifacts/
├── tests/
│   └── test_api.py
├── .github/
│   └── workflows/
│       └── ci.yml
├── train.py
├── requirements.txt
├── Dockerfile
├── docker-compose.yml
├── .dockerignore
└── .gitignore
```

## ⚙️ Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/Nagulanvelu/ml-deployment-monitoring-platform.git
cd ml-deployment-monitoring-platform
```

### 2. Create a virtual environment

Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### 3. Install dependencies

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Train the model

```bash
python train.py
```

The script trains the model and saves the generated model and evaluation metrics in the `artifacts/` directory.

### 5. Start the API

```bash
uvicorn app.main:app --reload
```

Open the interactive API documentation:

**http://127.0.0.1:8000/docs**

## 🔌 API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| GET | `/` | API welcome message |
| GET | `/health` | Check API and model health |
| POST | `/predict` | Generate a model prediction |
| GET | `/metrics` | View basic API monitoring metrics |

### Example Prediction

Send a POST request to `/predict`:

```json
{
  "features": {
    "sepal length (cm)": 5.1,
    "sepal width (cm)": 3.5,
    "petal length (cm)": 1.4,
    "petal width (cm)": 0.2
  }
}
```

The API returns a predicted Iris class, class ID, and confidence score.

## 🐳 Docker Deployment

Build the Docker image:

```bash
docker build -t ml-deployment-platform .
```

Run the container:

```bash
docker run --rm -p 8000:8000 ml-deployment-platform
```

Access the API at http://127.0.0.1:8000/docs.

## 🔬 MLflow Experiment Tracking

Train the model:

```bash
python train.py
```

Start the MLflow UI:

```bash
mlflow ui --port 5000
```

Open http://127.0.0.1:5000 to inspect logged model parameters and metrics.

## 🧪 Run Tests

```bash
python -m pytest -v
```

The test suite covers the home endpoint, model health, prediction requests, invalid input, and monitoring metrics.

## 🔄 GitHub Actions

The CI workflow automatically installs dependencies, trains the model, and runs the automated tests on pushes and pull requests.

[View workflow runs](https://github.com/Nagulanvelu/ml-deployment-monitoring-platform/actions)

## 📊 Model Evaluation

The current local training run achieved **90% test accuracy** on the Iris dataset using a reproducible train/test split.

This is a demonstration result, not a guarantee of performance on other datasets.

## 🚀 Future Improvements

- Prometheus and Grafana monitoring
- Model and data drift detection
- Model versioning and persistent experiment storage
- Cloud deployment
- Authentication and API security
- Improved logging and error handling

## 👨‍💻 Author

**Nagulan V**

- GitHub: [@Nagulanvelu](https://github.com/Nagulanvelu)
- Project: [ML Deployment & Monitoring Platform](https://github.com/Nagulanvelu/ml-deployment-monitoring-platform)

---

⭐ If you find this project useful, consider starring the repository.
