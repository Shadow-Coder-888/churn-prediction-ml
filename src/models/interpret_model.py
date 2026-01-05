import pandas as pd
import joblib


def load_artifacts():
    model = joblib.load("models/baseline_logreg.pkl")
    train = pd.read_csv("data/processed/train.csv")
    return model, train


def extract_feature_importance(model, train_df):
    X = train_df.drop(columns=["Churn"])
    coefficients = model.coef_[0]

    importance_df = pd.DataFrame({
        "feature": X.columns,
        "coefficient": coefficients,
        "abs_coefficient": abs(coefficients)
    })

    importance_df = importance_df.sort_values(
        by="abs_coefficient", ascending=False
    )

    return importance_df


def save_importance(df):
    df.to_csv("models/feature_importance.csv", index=False)


if __name__ == "__main__":
    model, train = load_artifacts()
    importance_df = extract_feature_importance(model, train)
    save_importance(importance_df)

    print("Top 10 Features Driving Churn:")
    print(importance_df.head(10))
