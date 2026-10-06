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

@app.api_route("/", methods=["GET", "HEAD"], response_class=HTMLResponse)
@app.api_route("/test", methods=["GET", "HEAD"], response_class=HTMLResponse)
def interactive_tester():
    """Compact, clean interactive test interface fitting completely in one desktop viewport."""
    return """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <title>Customer Churn Prediction</title>
  <style>
    * { box-sizing: border-box; }
    body {
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Arial, sans-serif;
      background: #f1f5f9;
      color: #0f172a;
      margin: 0;
      padding: 16px 12px;
      display: flex;
      justify-content: center;
      align-items: flex-start;
      min-height: 100vh;
    }
    .card {
      width: 100%;
      max-width: 640px;
      background: #ffffff;
      padding: 18px 24px 16px;
      border-radius: 10px;
      box-shadow: 0 4px 12px rgba(0,0,0,0.06);
      border: 1px solid #e2e8f0;
    }
    .header-row {
      display: flex;
      justify-content: space-between;
      align-items: center;
      margin-bottom: 2px;
    }
    h2 {
      margin: 0;
      font-size: 20px;
      font-weight: 700;
      color: #0f172a;
    }
    .status-badge {
      font-size: 11px;
      font-weight: 600;
      padding: 2px 8px;
      border-radius: 12px;
      background: #e2e8f0;
      color: #64748b;
    }
    .status-badge.online {
      background: #dcfce7;
      color: #15803d;
    }
    .status-badge.offline {
      background: #fee2e2;
      color: #b91c1c;
    }
    p.subtitle {
      color: #64748b;
      font-size: 13px;
      margin: 2px 0 12px;
    }
    .grid-container {
      display: grid;
      grid-template-columns: repeat(2, 1fr);
      gap: 8px 14px;
    }
    .form-group {
      margin-bottom: 0;
    }
    .full-width {
      grid-column: span 2;
    }
    label {
      display: block;
      font-size: 12px;
      font-weight: 600;
      color: #334155;
      margin-bottom: 2px;
    }
    input[type="number"], select {
      width: 100%;
      padding: 6px 10px;
      font-size: 13px;
      border: 1px solid #cbd5e1;
      border-radius: 5px;
      background: #ffffff;
      color: #0f172a;
      height: 33px;
    }
    input:focus, select:focus {
      outline: none;
      border-color: #2563eb;
    }
    button {
      width: 100%;
      padding: 9px;
      margin-top: 10px;
      font-size: 14px;
      font-weight: 600;
      color: #ffffff;
      background: #2563eb;
      border: none;
      border-radius: 6px;
      cursor: pointer;
      transition: background 0.15s;
    }
    button:hover {
      background: #1d4ed8;
    }
    button:disabled {
      background: #94a3b8;
      cursor: not-allowed;
    }
    #result-box {
      margin-top: 10px;
      padding: 10px 14px;
      border-radius: 6px;
      display: none;
      font-size: 13px;
      line-height: 1.5;
    }
    .result-churn {
      background: #fef2f2;
      border: 1px solid #f87171;
      color: #991b1b;
    }
    .result-retained {
      background: #f0fdf4;
      border: 1px solid #4ade80;
      color: #166534;
    }
    .result-error {
      background: #fffbeb;
      border: 1px solid #facc15;
      color: #92400e;
    }
    @media (max-width: 540px) {
      .grid-container {
        grid-template-columns: 1fr;
      }
      .full-width {
        grid-column: span 1;
      }
    }
  </style>
</head>
<body>
  <div class="card">
    <div class="header-row">
      <h2>Customer Churn Prediction</h2>
      <span id="api-status" class="status-badge">Checking...</span>
    </div>
    <p class="subtitle">Enter customer details to estimate churn risk.</p>

    <form id="churn-form" onsubmit="event.preventDefault(); predictChurn();">
      <div class="grid-container">
        <div class="form-group">
          <label for="tenure">Tenure (months)</label>
          <input type="number" id="tenure" min="0" max="120" value="2" required>
        </div>

        <div class="form-group">
          <label for="MonthlyCharges">Monthly Charges ($)</label>
          <input type="number" id="MonthlyCharges" step="0.01" min="0" value="89.90" required>
        </div>

        <div class="form-group">
          <label for="TotalCharges">Total Charges ($)</label>
          <input type="number" id="TotalCharges" step="0.01" min="0" value="179.80" required>
        </div>

        <div class="form-group">
          <label for="Contract">Contract</label>
          <select id="Contract">
            <option value="Month-to-month" selected>Month-to-month</option>
            <option value="One year">One year</option>
            <option value="Two year">Two year</option>
          </select>
        </div>

        <div class="form-group">
          <label for="InternetService">Internet Service</label>
          <select id="InternetService">
            <option value="Fiber optic" selected>Fiber optic</option>
            <option value="DSL">DSL</option>
            <option value="No">No</option>
          </select>
        </div>

        <div class="form-group">
          <label for="OnlineSecurity">Online Security</label>
          <select id="OnlineSecurity">
            <option value="No" selected>No</option>
            <option value="Yes">Yes</option>
            <option value="No internet service">No internet service</option>
          </select>
        </div>

        <div class="form-group">
          <label for="TechSupport">Tech Support</label>
          <select id="TechSupport">
            <option value="No" selected>No</option>
            <option value="Yes">Yes</option>
            <option value="No internet service">No internet service</option>
          </select>
        </div>

        <div class="form-group">
          <label for="PaperlessBilling">Paperless Billing</label>
          <select id="PaperlessBilling">
            <option value="Yes" selected>Yes</option>
            <option value="No">No</option>
          </select>
        </div>

        <div class="form-group full-width">
          <label for="PaymentMethod">Payment Method</label>
          <select id="PaymentMethod">
            <option value="Electronic check" selected>Electronic check</option>
            <option value="Mailed check">Mailed check</option>
            <option value="Bank transfer (automatic)">Bank transfer (automatic)</option>
            <option value="Credit card (automatic)">Credit card (automatic)</option>
          </select>
        </div>
      </div>

      <button type="submit" id="btn-predict">Predict Churn</button>
    </form>

    <div id="result-box"></div>
  </div>

  <script>
    // Check health status on page load
    async function checkHealth() {
      const badge = document.getElementById('api-status');
      try {
        const res = await fetch('/health');
        const data = await res.json();
        if (data.status === 'ok' && data.model_loaded === true) {
          badge.textContent = '● API Online';
          badge.className = 'status-badge online';
        } else {
          badge.textContent = '● API Unavailable';
          badge.className = 'status-badge offline';
        }
      } catch (e) {
        badge.textContent = '● API Unavailable';
        badge.className = 'status-badge offline';
      }
    }
    checkHealth();

    // Perform prediction
    async function predictChurn() {
      const btn = document.getElementById('btn-predict');
      const resultBox = document.getElementById('result-box');

      btn.disabled = true;
      btn.textContent = 'Predicting...';
      resultBox.style.display = 'none';

      const payload = {
        tenure: parseInt(document.getElementById('tenure').value, 10),
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
        resultBox.style.display = 'block';

        if (res.ok) {
          const isChurn = data.churn_prediction === 1;
          resultBox.className = isChurn ? 'result-churn' : 'result-retained';
          resultBox.innerHTML =
            '<strong>Prediction:</strong> ' + data.churn_label + '<br>' +
            '<strong>Risk Level:</strong> ' + data.risk_level + '<br>' +
            '<strong>Churn Probability:</strong> ' + (data.churn_probability * 100).toFixed(2) + '%';
        } else {
          resultBox.className = 'result-error';
          resultBox.textContent = 'Unable to make prediction. Please check the entered values.';
        }
      } catch (err) {
        resultBox.style.display = 'block';
        resultBox.className = 'result-error';
        resultBox.textContent = 'Unable to make prediction. Please check the entered values.';
      } finally {
        btn.disabled = false;
        btn.textContent = 'Predict Churn';
      }
    }
  </script>
</body>
</html>"""

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

