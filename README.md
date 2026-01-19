
---

# Churn Prediction ML API

## Problem Statement

Customer churn is a direct revenue risk for subscription-based businesses (telecom, SaaS, fintech).
Most organizations identify churn **after** the customer leaves, which is operationally useless.

This project solves the **early churn prediction** problem by exposing a machine-learning model as a **production-ready REST API** that predicts whether a customer is likely to churn based on behavioral and contract attributes.

---

## What Real Problem Does This Solve?

**Business pain points addressed:**

* Inability to proactively retain high-risk customers
* Manual, non-scalable churn analysis
* No standardized way to integrate ML predictions into backend systems

**Who benefits:**

* Retention teams (targeted offers)
* Product teams (understanding churn drivers)
* Backend systems (real-time churn scoring)

---

## How This Project Solves It

1. **Trains a supervised ML model** on historical customer data
2. **Encapsulates preprocessing + model** into a single pipeline (no feature mismatch)
3. **Exposes predictions via FastAPI**
4. **Accepts raw business data** (no one-hot encoding required from client)
5. Returns:

   * Binary churn prediction
   * Probability score for decision-making

This mirrors how ML models are deployed in real systems.

---

## System Architecture

```
Client (JSON)
   |
   v
FastAPI (/predict)
   |
   v
ML Pipeline
[Preprocessing → Model]
   |
   v
Prediction + Probability
```

---

## Tech Stack

### Machine Learning

* Python 3.10+
* pandas
* scikit-learn
* joblib

### Backend / API

* FastAPI
* Uvicorn

### Dev & Ops

* Git / GitHub
* Virtualenv
* JSON-based API contracts

---

## Dataset

* Based on a telecom churn dataset
* Features include:

  * Demographics (gender, senior citizen)
  * Services used (internet, streaming, security)
  * Contract details
  * Billing and payment methods

**Target variable:** `Churn` (Yes / No)

---

## Design Decisions (Important)

### 1. Single ML Pipeline (Critical)

* Preprocessing + model are trained and saved together
* Prevents feature mismatch during inference
* Industry-standard approach

### 2. Raw Input API

* API accepts business-level fields (strings, numbers)
* No one-hot encoding required from client
* Encoding handled internally

### 3. Stateless API

* No session storage
* Horizontally scalable
* Safe for containerization

### 4. Probability-Based Output

* Returns churn probability, not just Yes/No
* Enables threshold tuning by business teams

---

## Project Structure

```
churn-prediction-ml/
│
├── api/
│   └── main.py              # FastAPI app
│
├── src/
│   ├── models/
│   │   └── train_final.py   # Training script
│   └── inference/
│       └── predict.py       # Inference logic
│
├── artifacts/
│   └── churn_pipeline.joblib
│
├── data/
│   └── processed/
│
├── requirements.txt
└── README.md
```

---

## How to Run the Project (Exact Steps)

### 1. Clone Repository

```bash
git clone https://github.com/<your-username>/churn-prediction-ml.git
cd churn-prediction-ml
```

### 2. Create Virtual Environment

```bash
python -m venv venv
source venv/Scripts/activate   # Windows (Git Bash)
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Train the Model (Once)

```bash
python src/models/train_final.py
```

This generates:

```
artifacts/churn_pipeline.joblib
```

### 5. Start API Server

```bash
uvicorn api.main:app --reload
```

Server runs at:

```
http://127.0.0.1:8000
```

---

## API Usage

### Endpoint

```
POST /predict
```

### Sample Request

```json
{
  "gender": "Female",
  "SeniorCitizen": 0,
  "Partner": "Yes",
  "Dependents": "No",
  "tenure": 12,
  "PhoneService": "Yes",
  "MultipleLines": "No",
  "InternetService": "Fiber optic",
  "OnlineSecurity": "No",
  "OnlineBackup": "Yes",
  "DeviceProtection": "No",
  "TechSupport": "No",
  "StreamingTV": "Yes",
  "StreamingMovies": "No",
  "Contract": "Month-to-month",
  "PaperlessBilling": "Yes",
  "PaymentMethod": "Electronic check",
  "MonthlyCharges": 89.5,
  "TotalCharges": 1074.0
}
```

### Response

```json
{
  "churn_prediction": 1,
  "churn_probability": 0.7421
}
```

---

## Common Failure Modes (Handled)

* Unknown categorical values → safely ignored
* Missing fields → rejected by schema validation
* Feature mismatch → eliminated via pipeline

---

## Future Improvements

* Model versioning
* Threshold configuration via env vars
* Batch prediction endpoint
* Dockerized deployment
* Monitoring (drift + confidence tracking)

---

## Why This Project Matters

This is **not a notebook demo**.
This is an **end-to-end ML system** with:

* Correct training/inference separation
* Real API contracts
* Production-ready architecture

---


