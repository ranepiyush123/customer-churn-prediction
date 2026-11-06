"""FastAPI inference endpoint. Train the model before starting the server."""
from functools import lru_cache
from pathlib import Path
import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import Literal

MODEL_PATH = Path(__file__).resolve().parents[1] / "models" / "churn_pipeline.joblib"
app = FastAPI(title="Customer Churn Prediction API", version="1.0.0")

class Customer(BaseModel):
    tenure_months: int = Field(ge=0, le=120)
    monthly_charges: float = Field(ge=0, le=10000)
    support_tickets: int = Field(ge=0, le=100)
    contract_type: Literal["monthly", "annual"]
    autopay: Literal["yes", "no"]

@lru_cache

def get_model():
    if not MODEL_PATH.exists():
        raise HTTPException(status_code=503, detail="Model not found. Run python -m src.train first.")
    return joblib.load(MODEL_PATH)

@app.get("/health")
def health():
    return {"status": "ok", "model_ready": MODEL_PATH.exists()}

@app.post("/predict")
def predict(customer: Customer):
    model = get_model()
    row = pd.DataFrame([customer.model_dump()])
    probability = float(model.predict_proba(row)[0, 1])
    return {"churn_probability": round(probability, 4), "predicted_churn": probability >= 0.5, "risk_level": "high" if probability >= 0.7 else "medium" if probability >= 0.4 else "low"}
