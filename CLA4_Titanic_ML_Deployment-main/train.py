"""Task 1: Train and save a Titanic survival classification pipeline."""
from pathlib import Path
import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "data" / "train.csv"
MODEL_PATH = ROOT / "model.pkl"
TEST_ROWS_PATH = ROOT / "outputs" / "test_row_indices.csv"

FEATURES = ["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked"]
NUMERIC = ["Pclass", "Age", "SibSp", "Parch", "Fare"]
CATEGORICAL = ["Sex", "Embarked"]

def main():
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Missing {DATA_PATH}. Download train.csv from the official Kaggle "
            "Titanic competition data page and place it in the data folder."
        )
    df = pd.read_csv(DATA_PATH)
    missing = set(FEATURES + ["Survived"]) - set(df.columns)
    if missing:
        raise ValueError(f"train.csv is missing required columns: {sorted(missing)}")

    X = df[FEATURES].copy()
    y = df["Survived"].astype(int)
    train_idx, test_idx = train_test_split(
        df.index.to_numpy(), test_size=0.20, random_state=42, stratify=y
    )
    X_train, X_test = X.loc[train_idx], X.loc[test_idx]
    y_train, y_test = y.loc[train_idx], y.loc[test_idx]

    numeric_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    categorical_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])
    preprocessing = ColumnTransformer([
        ("numeric", numeric_pipe, NUMERIC),
        ("categorical", categorical_pipe, CATEGORICAL),
    ])
    pipeline = Pipeline([
        ("preprocessing", preprocessing),
        ("classifier", LogisticRegression(max_iter=1000, random_state=42)),
    ])
    pipeline.fit(X_train, y_train)
    predictions = pipeline.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)

    joblib.dump(pipeline, MODEL_PATH)
    TEST_ROWS_PATH.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"row_index": test_idx}).to_csv(TEST_ROWS_PATH, index=False)
    print(f"Test accuracy: {accuracy:.4f} ({accuracy:.2%})")
    print("\\nClassification report:")
    print(classification_report(y_test, predictions, zero_division=0))
    print(f"Saved complete pipeline to: {MODEL_PATH}")
    print(f"Saved held-out test row indices to: {TEST_ROWS_PATH}")

if __name__ == "__main__":
    main()
