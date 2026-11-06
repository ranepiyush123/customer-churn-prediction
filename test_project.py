from fastapi.testclient import TestClient
from src.api import app
from src.data import make_dataset
from src.train import train

client = TestClient(app)

def test_data_is_reproducible():
    assert make_dataset(50).equals(make_dataset(50))
    assert set(make_dataset(50).churn.unique()) <= {0, 1}

def test_training(tmp_path):
    result = train(tmp_path)
    assert (tmp_path / "churn_pipeline.joblib").exists()
    assert 0 <= result["metrics"][result["selected_model"]]["roc_auc"] <= 1

def test_health():
    assert client.get("/health").status_code == 200

def test_invalid_payload():
    response = client.post("/predict", json={"tenure_months": -2, "monthly_charges": 50, "support_tickets": 1, "contract_type": "monthly", "autopay": "yes"})
    assert response.status_code == 422
