import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix
import joblib


def load_data():
    train = pd.read_csv("data/processed/train.csv")
    test = pd.read_csv("data/processed/test.csv")
    return train, test


def tune_model(X_train, y_train):
    param_grid = {
        "n_estimators": [100, 200],
        "max_depth": [None, 10, 20],
        "min_samples_split": [2, 5],
        "min_samples_leaf": [1, 2]
    }

    rf = RandomForestClassifier(
        random_state=42,
        class_weight="balanced",
        n_jobs=-1
    )

    grid = GridSearchCV(
        estimator=rf,
        param_grid=param_grid,
        scoring="recall",
        cv=5,
        n_jobs=-1,
        verbose=1
    )

    grid.fit(X_train, y_train)
    return grid


def evaluate(model, X_test, y_test):
    y_pred = model.predict(X_test)
    print("Classification Report:")
    print(classification_report(y_test, y_pred))
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))


if __name__ == "__main__":
    train, test = load_data()

    X_train = train.drop(columns=["Churn"])
    y_train = train["Churn"]

    X_test = test.drop(columns=["Churn"])
    y_test = test["Churn"]

    grid = tune_model(X_train, y_train)

    print("Best Parameters:")
    print(grid.best_params_)

    best_model = grid.best_estimator_

    evaluate(best_model, X_test, y_test)

    joblib.dump(best_model, "models/random_forest_tuned.pkl")
