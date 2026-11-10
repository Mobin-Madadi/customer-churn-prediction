from fastapi.testclient import TestClient

from app.main import app


client = TestClient(app)


def test_home():
    response = client.get("/")

    assert response.status_code == 200
    assert response.json()["status"] == "running"


def test_health():
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json()["model_loaded"] is True


def test_predict():
    customer = {
        "Age": 35,
        "Tenure": 12,
        "MonthlyCharges": 85.5,
        "SupportCalls": 4,
        "ContractType": "Monthly",
        "InternetService": "Fiber",
        "PaymentMethod": "Electronic Check"
    }

    response = client.post("/predict", json=customer)

    assert response.status_code == 200

    result = response.json()

    assert result["prediction"] in ["Yes", "No"]
    assert 0 <= result["churn_probability"] <= 1
    assert 0 <= result["retention_probability"] <= 1

    assert abs(
        result["churn_probability"] +
        result["retention_probability"] - 1
    ) < 1e-6


def test_predict_invalid_age():
    customer = {
        "Age": 10,
        "Tenure": 12,
        "MonthlyCharges": 85.5,
        "SupportCalls": 4,
        "ContractType": "Monthly",
        "InternetService": "Fiber",
        "PaymentMethod": "Electronic Check"
    }

    response = client.post("/predict", json=customer)

    assert response.status_code == 422