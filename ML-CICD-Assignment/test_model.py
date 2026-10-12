"""Basic regression tests for the Iris model."""

from sklearn.datasets import load_iris

from model import train_model


def test_model_trains_and_returns_valid_accuracy():
    model, accuracy = train_model()
    assert model is not None
    assert 0.0 <= accuracy <= 1.0


def test_model_predicts_valid_iris_classes():
    model, _ = train_model()
    iris = load_iris()
    predictions = model.predict(iris.data[:5])
    assert len(predictions) == 5
    assert set(predictions).issubset(set(iris.target))
