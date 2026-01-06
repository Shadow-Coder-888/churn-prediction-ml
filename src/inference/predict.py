import joblib
import pandas as pd


class ChurnPredictor:
    def __init__(self, model_path="models/final_model.joblib"):
        artifact = joblib.load(model_path)
        self.model = artifact["model"]
        self.features = artifact["features"]

    def predict(self, data: dict):
        df = pd.DataFrame([data])

        # Align inference data to training schema
        df = df.reindex(columns=self.features, fill_value=0)

        prediction = self.model.predict(df)[0]
        probability = self.model.predict_proba(df)[0][1]

        return {
            "churn_prediction": int(prediction),
            "churn_probability": float(probability)
        }
