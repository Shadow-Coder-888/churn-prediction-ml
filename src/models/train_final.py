import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier


if __name__ == "__main__":
    train_df = pd.read_csv("data/processed/train.csv")

    X_train = train_df.drop(columns=["Churn"])
    y_train = train_df["Churn"]

    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=10,
        random_state=42
    )

    model.fit(X_train, y_train)

    joblib.dump(
        {
            "model": model,
            "features": X_train.columns.tolist()
        },
        "models/final_model.joblib"
    )

    print("Final model + feature schema saved")
