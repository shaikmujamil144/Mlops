"""Train and evaluate a simple Iris classification model."""

from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

MODEL_NAME = "Decision Tree Classifier"
DATASET_NAME = "Iris dataset"
RANDOM_STATE = 42


def train_model():
    """Return a fitted model and its held-out test accuracy."""
    iris = load_iris()
    X_train, X_test, y_train, y_test = train_test_split(
        iris.data,
        iris.target,
        test_size=0.2,
        random_state=RANDOM_STATE,
        stratify=iris.target,
    )
    model = DecisionTreeClassifier(random_state=RANDOM_STATE)
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    return model, accuracy


def main():
    model, accuracy = train_model()
    iris = load_iris()
    prediction = int(model.predict([iris.data[0]])[0])
    print(f"Model: {MODEL_NAME}")
    print(f"Dataset: {DATASET_NAME}")
    print(f"Test accuracy: {accuracy:.2%}")
    print(f"Example prediction: {iris.target_names[prediction]}")


if __name__ == "__main__":
    main()
