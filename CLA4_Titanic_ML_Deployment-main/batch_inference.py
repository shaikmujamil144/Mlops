"""Task 3: Predict 20 held-out records together and save a CSV."""
from pathlib import Path
import joblib
import pandas as pd

ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "data" / "train.csv"
MODEL_PATH = ROOT / "model.pkl"
TEST_ROWS_PATH = ROOT / "outputs" / "test_row_indices.csv"
OUTPUT_PATH = ROOT / "outputs" / "batch_predictions.csv"
FEATURES = ["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked"]

def main():
    for path in [MODEL_PATH, DATA_PATH, TEST_ROWS_PATH]:
        if not path.exists():
            raise FileNotFoundError(f"{path} not found. Run `python train.py` after adding Kaggle train.csv.")
    df = pd.read_csv(DATA_PATH)
    test_indices = pd.read_csv(TEST_ROWS_PATH)["row_index"].astype(int).tolist()
    unseen = df.loc[test_indices[:20]].copy()
    if len(unseen) < 20:
        raise ValueError("Need at least 20 held-out test records.")
    X = unseen[FEATURES].copy()
    predictions = joblib.load(MODEL_PATH).predict(X).astype(int)
    result = X.copy()
    result.insert(0, "PassengerId", unseen["PassengerId"].values if "PassengerId" in unseen else unseen.index)
    result["Predicted_Survived"] = predictions
    result["Prediction_Label"] = result["Predicted_Survived"].map({1: "Survived", 0: "Did not survive"})
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(OUTPUT_PATH, index=False)
    print(f"Generated predictions for {len(result)} held-out records:")
    print(result.to_string(index=False))
    print(f"\\nSaved results to: {OUTPUT_PATH}")

if __name__ == "__main__":
    main()
