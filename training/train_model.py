"""
Customer Churn Prediction Model Training Pipeline
Source Dataset: Kaggle - Telco Customer Churn (blastchar/telco-customer-churn)
Author: 24f1001527
"""

import os
import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report, confusion_matrix

def train_and_export():
    # 1. Load Dataset
    data_path = os.path.join(os.path.dirname(__file__), "..", "data", "Telco-Customer-Churn.csv")
    print(f"Loading dataset from: {data_path}")
    df = pd.read_csv(data_path)
    print(f"Dataset shape: {df.shape}")

    # 2. Data Cleaning & Preparation
    # TotalCharges has 11 whitespace entries in the raw dataset
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"].replace(" ", np.nan))
    # Impute missing TotalCharges with MonthlyCharges * tenure
    df["TotalCharges"] = df["TotalCharges"].fillna(df["MonthlyCharges"] * df["tenure"])

    # Target variable: Churn ("Yes" -> 1, "No" -> 0)
    y = (df["Churn"] == "Yes").astype(int)

    # 3. Select Production Feature Subset
    # 3 Numerical features + 6 Categorical features representing customer tenure, billing, and services
    num_features = ["tenure", "MonthlyCharges", "TotalCharges"]
    cat_features = [
        "Contract",
        "InternetService",
        "OnlineSecurity",
        "TechSupport",
        "PaperlessBilling",
        "PaymentMethod"
    ]
    feature_cols = num_features + cat_features
    X = df[feature_cols].copy()

    print(f"Selected {len(feature_cols)} features: {feature_cols}")

    # 4. Train-Test Split (80/20 Stratified)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    print(f"Training samples: {len(X_train)} | Test samples: {len(X_test)}")

    # 5. Build Preprocessing & Modeling Pipeline
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), num_features),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_features)
        ]
    )

    model_pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", LogisticRegression(random_state=42, max_iter=1000, class_weight="balanced"))
    ])

    # 6. Model Training
    print("Training LogisticRegression pipeline...")
    model_pipeline.fit(X_train, y_train)

    # 7. Model Evaluation
    y_pred = model_pipeline.predict(X_test)
    y_proba = model_pipeline.predict_proba(X_test)[:, 1]

    acc = accuracy_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_proba)

    print("\n" + "="*50)
    print("MODEL EVALUATION RESULTS")
    print("="*50)
    print(f"Test Accuracy : {acc:.4f} ({acc*100:.2f}%)")
    print(f"ROC-AUC Score : {roc_auc:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=["Retained (0)", "Churned (1)"]))
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    print("="*50)

    # 8. Export Model Artifact
    export_path = os.path.join(os.path.dirname(__file__), "..", "model.pkl")
    joblib.dump(model_pipeline, export_path)
    print(f"\nModel exported successfully to: {export_path}")
    print(f"File size: {os.path.getsize(export_path)} bytes")

    # 9. Smoke Test Serialized Artifact
    loaded_model = joblib.load(export_path)
    sample_request = pd.DataFrame([{
        "tenure": 3,
        "MonthlyCharges": 85.5,
        "TotalCharges": 256.5,
        "Contract": "Month-to-month",
        "InternetService": "Fiber optic",
        "OnlineSecurity": "No",
        "TechSupport": "No",
        "PaperlessBilling": "Yes",
        "PaymentMethod": "Electronic check"
    }])
    pred = loaded_model.predict(sample_request)[0]
    proba = loaded_model.predict_proba(sample_request)[0]
    print(f"Smoke test inference: Churn={pred} (Probabilities: {proba.round(4)})")

if __name__ == "__main__":
    train_and_export()
