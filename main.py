"""
Customer Churn Prediction API
Built with FastAPI for production model serving.
Dataset: Kaggle Telco Customer Churn
"""

import os
import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

# Locate trained pipeline artifact
MODEL_PATH = os.path.join(os.path.dirname(__file__), "model.pkl")

app = FastAPI(
    title="Customer Churn Prediction API",
    description="Production REST API predicting customer churn using a scikit-learn pipeline trained on Kaggle Telco data.",
    version="1.0.0"
)

# Load model pipeline on startup
try:
    pipeline = joblib.load(MODEL_PATH)
except Exception as e:
    pipeline = None

# Pydantic Request Schema
class CustomerData(BaseModel):
    tenure: int = Field(..., ge=0, le=120, description="Number of months customer has stayed with company", example=12)
    MonthlyCharges: float = Field(..., ge=0.0, description="Monthly subscription amount in USD", example=65.5)
    TotalCharges: float = Field(..., ge=0.0, description="Total amount charged to customer in USD", example=786.0)
    Contract: str = Field(..., description="Contract term ('Month-to-month', 'One year', 'Two year')", example="Month-to-month")
    InternetService: str = Field(..., description="Internet service type ('DSL', 'Fiber optic', 'No')", example="Fiber optic")
    OnlineSecurity: str = Field(..., description="Online security add-on ('Yes', 'No', 'No internet service')", example="No")
    TechSupport: str = Field(..., description="Tech support add-on ('Yes', 'No', 'No internet service')", example="No")
    PaperlessBilling: str = Field(..., description="Paperless billing ('Yes', 'No')", example="Yes")
    PaymentMethod: str = Field(
        ...,
        description="Payment method ('Electronic check', 'Mailed check', 'Bank transfer (automatic)', 'Credit card (automatic)')",
        example="Electronic check"
    )

# Pydantic Response Schema
class ChurnPredictionResponse(BaseModel):
    churn_prediction: int = Field(..., description="1 for Churn, 0 for Retained")
    churn_label: str = Field(..., description="'Churn' or 'Retained'")
    churn_probability: float = Field(..., description="Estimated probability of churning (0.0 to 1.0)")
    risk_level: str = Field(..., description="'High', 'Medium', or 'Low'")

@app.get("/")
def root():
    return {
        "message": "Customer Churn Prediction API is running.",
        "docs_url": "/docs",
        "health_check": "/health",
        "interactive_tester": "/test"
    }

@app.get("/health")
def health():
    """Health check endpoint required by deployment platforms and load balancers."""
    return {
        "status": "ok",
        "model_loaded": pipeline is not None
    }

@app.post("/predict", response_model=ChurnPredictionResponse)
def predict(customer: CustomerData):
    """Predicts customer churn probability and classification based on account & service features."""
    if pipeline is None:
        raise HTTPException(
            status_code=503,
            detail="Model artifact is not loaded. Ensure model.pkl is present."
        )

    # Convert request to single-row pandas DataFrame
    input_df = pd.DataFrame([customer.model_dump()])

    try:
        pred = int(pipeline.predict(input_df)[0])
        probas = pipeline.predict_proba(input_df)[0]
        churn_prob = float(probas[1])

        if churn_prob >= 0.70:
            risk = "High"
        elif churn_prob >= 0.40:
            risk = "Medium"
        else:
            risk = "Low"

        return ChurnPredictionResponse(
            churn_prediction=pred,
            churn_label="Churn" if pred == 1 else "Retained",
            churn_probability=round(churn_prob, 4),
            risk_level=risk
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Inference error: {str(e)}")

@app.get("/test", response_class=HTMLResponse)
def interactive_tester():
    """Interactive in-browser testing UI."""
    return """
    <!DOCTYPE html>
    <html lang="en">
    <head>
      <meta charset="UTF-8">
      <title>Customer Churn Predictor</title>
      <style>
        body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; background: #f8fafc; color: #1e293b; padding: 30px; margin: 0; }
        .container { max-width: 600px; margin: 0 auto; background: white; padding: 25px 30px; border-radius: 12px; box-shadow: 0 4px 6px -1px rgb(0 0 0 / 0.1); }
        h2 { margin-top: 0; color: #0f172a; border-bottom: 2px solid #e2e8f0; padding-bottom: 10px; }
        .form-group { margin-bottom: 14px; }
        label { display: block; font-weight: 600; font-size: 13px; margin-bottom: 4px; color: #475569; }
        input, select { width: 100%; box-sizing: border-box; padding: 8px 12px; border: 1px solid #cbd5e1; border-radius: 6px; font-size: 14px; }
        button { width: 100%; padding: 12px; background: #2563eb; color: white; border: none; border-radius: 6px; font-weight: 600; font-size: 15px; cursor: pointer; margin-top: 10px; transition: background 0.2s; }
        button:hover { background: #1d4ed8; }
        #result { margin-top: 20px; padding: 15px; border-radius: 8px; display: none; }
        .high-risk { background: #fee2e2; border: 1px solid #ef4444; color: #991b1b; }
        .low-risk { background: #dcfce7; border: 1px solid #22c55e; color: #166534; }
      </style>
    </head>
    <body>
      <div class="container">
        <h2>Customer Churn Predictor</h2>
        <div class="form-group">
          <label>Tenure (Months with company)</label>
          <input type="number" id="tenure" value="3">
        </div>
        <div class="form-group">
          <label>Monthly Charges ($)</label>
          <input type="number" step="0.01" id="MonthlyCharges" value="85.50">
        </div>
        <div class="form-group">
          <label>Total Charges ($)</label>
          <input type="number" step="0.01" id="TotalCharges" value="256.50">
        </div>
        <div class="form-group">
          <label>Contract Type</label>
          <select id="Contract">
            <option value="Month-to-month" selected>Month-to-month</option>
            <option value="One year">One year</option>
            <option value="Two year">Two year</option>
          </select>
        </div>
        <div class="form-group">
          <label>Internet Service</label>
          <select id="InternetService">
            <option value="Fiber optic" selected>Fiber optic</option>
            <option value="DSL">DSL</option>
            <option value="No">No</option>
          </select>
        </div>
        <div class="form-group">
          <label>Online Security</label>
          <select id="OnlineSecurity">
            <option value="No" selected>No</option>
            <option value="Yes">Yes</option>
            <option value="No internet service">No internet service</option>
          </select>
        </div>
        <div class="form-group">
          <label>Tech Support</label>
          <select id="TechSupport">
            <option value="No" selected>No</option>
            <option value="Yes">Yes</option>
            <option value="No internet service">No internet service</option>
          </select>
        </div>
        <div class="form-group">
          <label>Paperless Billing</label>
          <select id="PaperlessBilling">
            <option value="Yes" selected>Yes</option>
            <option value="No">No</option>
          </select>
        </div>
        <div class="form-group">
          <label>Payment Method</label>
          <select id="PaymentMethod">
            <option value="Electronic check" selected>Electronic check</option>
            <option value="Mailed check">Mailed check</option>
            <option value="Bank transfer (automatic)">Bank transfer (automatic)</option>
            <option value="Credit card (automatic)">Credit card (automatic)</option>
          </select>
        </div>
        <button onclick="runPredict()">Predict Churn</button>
        <div id="result"></div>
      </div>
      <script>
        async function runPredict() {
          const payload = {
            tenure: parseInt(document.getElementById('tenure').value),
            MonthlyCharges: parseFloat(document.getElementById('MonthlyCharges').value),
            TotalCharges: parseFloat(document.getElementById('TotalCharges').value),
            Contract: document.getElementById('Contract').value,
            InternetService: document.getElementById('InternetService').value,
            OnlineSecurity: document.getElementById('OnlineSecurity').value,
            TechSupport: document.getElementById('TechSupport').value,
            PaperlessBilling: document.getElementById('PaperlessBilling').value,
            PaymentMethod: document.getElementById('PaymentMethod').value
          };
          try {
            const res = await fetch('/predict', {
              method: 'POST',
              headers: { 'Content-Type': 'application/json' },
              body: JSON.stringify(payload)
            });
            const data = await res.json();
            const resBox = document.getElementById('result');
            resBox.style.display = 'block';
            if (res.ok) {
              resBox.className = data.churn_prediction === 1 ? 'high-risk' : 'low-risk';
              resBox.innerHTML = `<strong>Result:</strong> ${data.churn_label} (Risk: ${data.risk_level})<br>` +
                                 `<strong>Churn Probability:</strong> ${(data.churn_probability * 100).toFixed(1)}%`;
            } else {
              resBox.className = 'high-risk';
              resBox.innerText = 'Error: ' + JSON.stringify(data);
            }
          } catch (err) {
            alert('Request failed: ' + err);
          }
        }
      </script>
    </body>
    </html>
    """
