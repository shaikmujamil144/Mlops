"""Task 2: Run one online prediction for an unseen passenger record."""
from pathlib import Path
import joblib
import pandas as pd

ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "model.pkl"

def main():
    if not MODEL_PATH.exists():
        raise FileNotFoundError("model.pkl not found. Run `python train.py` first.")
    model = joblib.load(MODEL_PATH)

    # Example passenger input. This is an inference example, not a training row.
    passenger = {
        "Pclass": 3, "Sex": "male", "Age": 22, "SibSp": 1,
        "Parch": 0, "Fare": 7.25, "Embarked": "S"
    }
    X = pd.DataFrame([passenger])
    prediction = int(model.predict(X)[0])
    label = "Survived" if prediction == 1 else "Did not survive"
    print("Online inference")
    print("Passenger details:", passenger)
    print(f"Prediction: {prediction} ({label})")

if __name__ == "__main__":
    main()
