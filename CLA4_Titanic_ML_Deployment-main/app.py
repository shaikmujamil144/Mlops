"""Task 5: FastAPI model-serving API."""
from pathlib import Path
from typing import Literal
import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel, Field

ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "model.pkl"
model = joblib.load(MODEL_PATH) if MODEL_PATH.exists() else None

app = FastAPI(title="Titanic Survival Prediction API", version="1.0.0")

class Passenger(BaseModel):
    Pclass: int = Field(..., ge=1, le=3, description="Passenger class: 1, 2, or 3")
    Sex: Literal["male", "female"]
    Age: float = Field(..., ge=0, le=100)
    SibSp: int = Field(..., ge=0, le=10)
    Parch: int = Field(..., ge=0, le=10)
    Fare: float = Field(..., ge=0)
    Embarked: Literal["C", "Q", "S"]

@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": model is not None}

@app.post("/predict")
def predict(passenger: Passenger):
    if model is None:
        return {"error": "Model is not loaded. Run `python train.py` to create model.pkl."}
    data = pd.DataFrame([passenger.model_dump()])
    prediction = int(model.predict(data)[0])
    return {
        "prediction": prediction,
        "label": "Survived" if prediction == 1 else "Did not survive"
    }
