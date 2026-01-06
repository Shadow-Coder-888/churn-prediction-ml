import pandas as pd
import joblib
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


def load_data():
    test = pd.read_csv("data/processed/test.csv")
    X_test = test.drop(columns=["Churn"])
    y_test = test["Churn"]
    return X_test, y_test


def evaluate_model(name, model, X_test, y_test):
    y_pred = model.predict(X_test)
    return {
        "model": name,
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "f1_score": f1_score(y_test, y_pred)
    }


if __name__ == "__main__":
    X_test, y_test = load_data()

    models = {
        "logistic_regression": joblib.load("models/baseline_logreg.pkl"),
        "random_forest": joblib.load("models/random_forest.pkl"),
        "random_forest_tuned": joblib.load("models/random_forest_tuned.pkl")
    }

    results = []

    for name, model in models.items():
        metrics = evaluate_model(name, model, X_test, y_test)
        results.append(metrics)

    results_df = pd.DataFrame(results)
    results_df.to_csv("models/model_comparison.csv", index=False)

    print(results_df)
