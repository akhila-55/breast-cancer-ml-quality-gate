import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib


def test_model_accuracy():
    df = pd.read_csv("breast_cancer_v1_100.csv")

    df = df.loc[:, ~df.columns.str.contains("^Unnamed")]

    X = df.iloc[:, :-1]
    y = df.iloc[:, -1]

    if y.dtype == "object":
        y = y.astype("category").cat.codes

    X = pd.get_dummies(X, drop_first=True)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    model = joblib.load("breast_cancer_model.pkl")

    predictions = model.predict(X_test)

    accuracy = accuracy_score(y_test, predictions)

    assert accuracy >= 0.90, (
        f"Accuracy {accuracy:.4f} is below 0.90"
    )
