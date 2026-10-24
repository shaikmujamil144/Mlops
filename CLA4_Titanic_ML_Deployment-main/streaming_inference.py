"""Task 4: Simulate a stream using 10 held-out records, one per second."""
from pathlib import Path
import time
import joblib
import pandas as pd

ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "data" / "train.csv"
MODEL_PATH = ROOT / "model.pkl"
TEST_ROWS_PATH = ROOT / "outputs" / "test_row_indices.csv"
FEATURES = ["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked"]

def main():
    for path in [MODEL_PATH, DATA_PATH, TEST_ROWS_PATH]:
        if not path.exists():
            raise FileNotFoundError(f"{path} not found. Run `python train.py` after adding Kaggle train.csv.")
    df = pd.read_csv(DATA_PATH)
    test_indices = pd.read_csv(TEST_ROWS_PATH)["row_index"].astype(int).tolist()
    records = df.loc[test_indices[:10]]
    if len(records) < 10:
        raise ValueError("Need at least 10 held-out test records.")
    model = joblib.load(MODEL_PATH)
    print("Streaming inference started (one record at a time, 1-second delay)...")
    for i, (_, row) in enumerate(records.iterrows(), start=1):
        prediction = int(model.predict(row[FEATURES].to_frame().T)[0])
        label = "Survived" if prediction == 1 else "Did not survive"
        pid = row["PassengerId"] if "PassengerId" in row.index else row.name
        print(f"Incoming record {i}/10 | PassengerId={pid} | prediction={prediction} ({label})")
        if i < len(records):
            time.sleep(1)
    print("Streaming inference finished.")

if __name__ == "__main__":
    main()
