import pandas as pd
from sklearn.model_selection import train_test_split


def load_data(path):
    return pd.read_csv(path)


def preprocess(df):
    # Drop identifier
    df = df.drop(columns=['customerID'])

    # Convert target
    df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0})

    # Fix TotalCharges
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    df['TotalCharges'] = df['TotalCharges'].fillna(0)

    # Binary categorical encoding
    binary_cols = [
        'gender', 'Partner', 'Dependents', 'PhoneService',
        'PaperlessBilling'
    ]

    for col in binary_cols:
        df[col] = df[col].map({'Yes': 1, 'No': 0})

    # One-hot encode remaining categorical columns
    df = pd.get_dummies(df, drop_first=True)

    return df


def split_and_save(df, output_path):
    X = df.drop(columns=['Churn'])
    y = df['Churn']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    train = pd.concat([X_train, y_train], axis=1)
    test = pd.concat([X_test, y_test], axis=1)

    train.to_csv(f"{output_path}/train.csv", index=False)
    test.to_csv(f"{output_path}/test.csv", index=False)


if __name__ == "__main__":
    df = load_data("data/raw/telco_churn.csv")
    df = preprocess(df)
    split_and_save(df, "data/processed")
