from fastapi.testclient import TestClient

from app import main
from app.main import app

client = TestClient(app)


def test_classification_training_returns_metrics_and_saves_model(tmp_path, monkeypatch):
    monkeypatch.setattr(main, "MODEL_DIR", tmp_path)
    rows = ["age,score,city,purchased"]
    for i in range(60):
        purchased = "yes" if i % 2 == 0 else "no"
        city = "Hyderabad" if i % 3 == 0 else "Pune"
        rows.append(f"{20 + i},{50 + i * 2},{city},{purchased}")

    response = client.post(
        "/api/models/train",
        data={"target": "purchased", "task": "classification"},
        files={"file": ("customers.csv", "\n".join(rows), "text/csv")},
    )

    assert response.status_code == 200, response.text
    data = response.json()
    assert data["task"] == "classification"
    assert data["target"] == "purchased"
    assert data["best_model"]
    assert len(data["results"]) == 3
    assert all(item["status"] == "success" for item in data["results"])
    assert "accuracy" in data["results"][0]["metrics"]
    assert "confusion_matrix" in data["results"][0]["metrics"]
    assert (tmp_path / f"{data['model_id']}.joblib").exists()

    registry = client.get("/api/models")
    assert registry.status_code == 200
    saved = next(item for item in registry.json()["models"] if item["model_id"] == data["model_id"])
    assert saved["target"] == "purchased"
    assert saved["feature_count"] == 3

    prediction = client.post(
        f"/api/models/{data['model_id']}/predict",
        json={"age": 32, "score": 74, "city": "Hyderabad"},
    )
    assert prediction.status_code == 200, prediction.text
    assert "prediction" in prediction.json()
    assert "probabilities" in prediction.json()


def test_regression_training_returns_regression_metrics(tmp_path, monkeypatch):
    monkeypatch.setattr(main, "MODEL_DIR", tmp_path)
    rows = ["size,bedrooms,price"]
    for i in range(60):
        rows.append(f"{500 + i * 10},{1 + i % 4},{50000 + i * 2500}")

    response = client.post(
        "/api/models/train",
        data={"target": "price", "task": "regression"},
        files={"file": ("houses.csv", "\n".join(rows), "text/csv")},
    )

    assert response.status_code == 200, response.text
    data = response.json()
    assert data["task"] == "regression"
    assert len(data["results"]) == 3
    assert all(item["status"] == "success" for item in data["results"])
    metrics = data["results"][0]["metrics"]
    assert "mae" in metrics
    assert "rmse" in metrics
    assert "r2" in metrics


def test_training_rejects_unknown_target():
    response = client.post(
        "/api/models/train",
        data={"target": "missing", "task": "classification"},
        files={"file": ("data.csv", "a,b\n1,yes\n2,no\n3,yes\n4,no\n5,yes\n6,no\n7,yes\n8,no\n9,yes\n10,no", "text/csv")},
    )
    assert response.status_code == 400
    assert "not found" in response.json()["detail"]
