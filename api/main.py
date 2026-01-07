from fastapi import FastAPI
from pydantic import BaseModel
from src.inference.predict import ChurnPredictor

app = FastAPI(
    title="Churn Prediction API",
    version="1.0"
)

predictor = ChurnPredictor()


class ChurnRequest(BaseModel):
    gender: int
    SeniorCitizen: int
    Partner: int
    Dependents: int
    tenure: int
    MonthlyCharges: float
    TotalCharges: float
    Contract_One_year: int = 0
    Contract_Two_year: int = 0
    PaymentMethod_Electronic_check: int = 0


@app.get("/")
def health_check():
    return {"status": "ok"}


@app.post("/predict")
def predict_churn(request: ChurnRequest):
    result = predictor.predict(request.dict())
    return result
