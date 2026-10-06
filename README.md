# Customer Churn Prediction API

A production-grade REST API that predicts telecommunications customer churn using a machine learning pipeline trained on a real-world tabular dataset from Kaggle.

Built for the **Getting Started with Machine Learning in Production** assignment (Option B: Custom Dataset & Model).

---

## 1. Project Overview & Problem Statement
Customer attrition (churn) directly impacts recurring revenue in telecommunications and subscription-based businesses. This service exposes a trained scikit-learn classification pipeline through a FastAPI REST application, allowing client applications to submit customer subscription details and receive real-time churn predictions, probabilities, and categorized risk levels (`High`, `Medium`, `Low`).

---

## 2. Dataset Information
- **Dataset Name:** [Telco Customer Churn](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)
- **Kaggle Source:** Hosted by `blastchar` on Kaggle (IBM Sample Data)
- **Dataset URL:** `https://www.kaggle.com/datasets/blastchar/telco-customer-churn`
- **Total Records:** 7,043 customers (21 attributes)
- **Target Variable:** `Churn` (Binary: `Yes` / `No`)
- **Key Features Used:**
  - `tenure`: Months the customer has remained with the provider (Numeric)
  - `MonthlyCharges`: Monthly bill in USD (Numeric)
  - `TotalCharges`: Lifetime cumulative charges in USD (Numeric)
  - `Contract`: Term commitment (`Month-to-month`, `One year`, `Two year`)
  - `InternetService`: Connection type (`Fiber optic`, `DSL`, `No`)
  - `OnlineSecurity`: Cyber security add-on (`Yes`, `No`, `No internet service`)
  - `TechSupport`: Technical support add-on (`Yes`, `No`, `No internet service`)
  - `PaperlessBilling`: Paperless billing enrollment (`Yes`, `No`)
  - `PaymentMethod`: Electronic check, Mailed check, Bank transfer, Credit card

---

## 3. Machine Learning Model & Evaluation
- **Algorithm:** Logistic Regression with balanced class weighting (`class_weight="balanced"`) combined with `StandardScaler` for continuous metrics and `OneHotEncoder` for categorical factors in a unified scikit-learn `Pipeline`.
- **Serialization:** Exported directly to `model.pkl` via `joblib.dump()`.
- **Performance Metrics (on held-out 20% test split, 1,409 samples):**
  - **Accuracy:** 73.67%
  - **ROC-AUC Score:** 0.8386
  - **Recall (Churn class):** 81% (identifies over 80% of at-risk customers)
  - **Precision (Retained class):** 91%

---

## 4. API Specification & Endpoints

### `GET /health`
Verifies server health and model artifact loading.
- **Response:**
  ```json
  {
    "status": "ok",
    "model_loaded": true
  }
  ```

### `POST /predict`
Performs real-time churn inference for a given customer profile.
- **Request Headers:** `Content-Type: application/json`
- **Example Request:**
  ```json
  {
    "tenure": 2,
    "MonthlyCharges": 89.90,
    "TotalCharges": 179.80,
    "Contract": "Month-to-month",
    "InternetService": "Fiber optic",
    "OnlineSecurity": "No",
    "TechSupport": "No",
    "PaperlessBilling": "Yes",
    "PaymentMethod": "Electronic check"
  }
  ```
- **Example Response:**
  ```json
  {
    "churn_prediction": 1,
    "churn_label": "Churn",
    "churn_probability": 0.8918,
    "risk_level": "High"
  }
  ```

### `GET /test`
Interactive web-based UI embedded directly in FastAPI for testing predictions in any web browser without external tools.

---

## 5. Local Setup & Execution

### Prerequisites
- Python 3.10+
- `pip` package manager

### Installation
```bash
# Clone the repository
git clone https://github.com/24f1001527/customer-churn-production-api.git
cd customer-churn-production-api

# Create and activate virtual environment
python -m venv venv
# On Windows:
.\venv\Scripts\activate
# On Linux/macOS:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### Run Locally
```bash
uvicorn main:app --reload --port 8000
```
- Interactive Swagger UI: [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)
- Interactive Tester: [http://127.0.0.1:8000/test](http://127.0.0.1:8000/test)
- Health Check: [http://127.0.0.1:8000/health](http://127.0.0.1:8000/health)

---

## 6. Docker Containerization

### Build Image
```bash
docker build -t customer-churn-api .
```

### Run Container
```bash
docker run -p 8000:8000 customer-churn-api
```
Access at `http://localhost:8000`.

---

## 7. Retraining the Model

To retrain the model and regenerate `model.pkl`:
```bash
python training/train_model.py
```

---

## 8. Deployment on Render

This application is configured for deployment on [Render](https://render.com) Web Services:
- **Environment:** Python 3
- **Build Command:** `pip install -r requirements.txt`
- **Start Command:** `uvicorn main:app --host 0.0.0.0 --port $PORT`
