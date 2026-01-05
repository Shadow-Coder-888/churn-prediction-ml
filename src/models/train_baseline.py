import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
import joblib


def load_data():
    train = pd.read_csv("data/processed/train.csv")
    test = pd.read_csv("data/processed/test.csv")
    return train, test


def train_model(X_train, y_train):
    model = LogisticRegression(max_iter=1000, class_weight="balanced")
    model.fit(X_train, y_train)
    return model


def evaluate(model, X_test, y_test):
    y_pred = model.predict(X_test)
    print("Classification Report:")
    print(classification_report(y_test, y_pred))
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))


def save_model(model, path):
    joblib.dump(model, path)


if __name__ == "__main__":
    train, test = load_data()

    X_train = train.drop(columns=["Churn"])
    y_train = train["Churn"]

    X_test = test.drop(columns=["Churn"])
    y_test = test["Churn"]

    model = train_model(X_train, y_train)
    evaluate(model, X_test, y_test)

    save_model(model, "models/baseline_logreg.pkl")
