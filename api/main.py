import logging
import joblib
import pandas as pd
from pathlib import Path
from datetime import date
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, ConfigDict

# Import custom transformer so Joblib can load the saved pipeline
from src.feature_engineering import FeatureEngineering

logger = logging.getLogger(__name__)
BASE_DIR = Path(__file__).resolve().parents[1]
MODEL_PATH = BASE_DIR / "artifacts" / "models" / "xgboost_pipeline.joblib"
THRESHOLD = 0.5

# Step 1: Define request schema
class ClaimRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    Patient_Age: int = Field(ge=0, le=120)
    Claim_Amount: float = Field(ge=0)
    Days_Between_Service_and_Claim: int = Field(ge=0)
    Number_of_Claims_Per_Provider_Monthly: int = Field(ge=0)
    Length_of_Stay: int = Field(ge=0)
    Prior_Visits_12m: int = Field(ge=0)
    Diagnosis_Code: str
    Procedure_Code: str
    Insurance_Type: str
    Provider_Specialty: str
    Visit_Type: str
    Chronic_Condition_Flag: int | str
    Claim_Submission_Date: date

# Step 2: Define response schema
class PredictionResponse(BaseModel):
    predicted_fraud: int
    fraud_score: float
    risk_level: str
    decision_threshold: float
    message: str

# Step 3: Load model once when API starts
@asynccontextmanager
async def lifespan(app: FastAPI):
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model not found: {MODEL_PATH}")
    app.state.model = joblib.load(MODEL_PATH)
    print(f"Model loaded successfully: {MODEL_PATH}")
    yield
    app.state.model = None

# Step 4: Initialize FastAPI
app = FastAPI(
    title="Insurance Fraud Detection API",
    description="Insurance claim fraud prediction using XGBoost",
    version="1.0.0",
    lifespan=lifespan
)

# Step 5: Health endpoint
@app.get("/health")
def health():
    return {
        "status": "healthy",
        "model_loaded": app.state.model is not None
    }

# Step 6: Model information endpoint
@app.get("/model-info")
def model_info():
    model = app.state.model
    return {
        "model": "XGBoost Classifier",
        "model_file": MODEL_PATH.name,
        "pipeline_steps": list(model.named_steps.keys()),
        "decision_threshold": THRESHOLD,
        "status": "ready"
    }

# Step 7: Prediction endpoint
@app.post("/predict", response_model=PredictionResponse)
def predict_fraud(claim: ClaimRequest):
    try:
        # Convert validated JSON request into DataFrame
        claim_data = claim.model_dump(mode="json")
        df = pd.DataFrame([claim_data])

        # Pipeline automatically performs feature engineering,
        # preprocessing and feature selection (if present)
        model = app.state.model
        fraud_score = float(model.predict_proba(df)[0, 1])

        # Convert model score into binary classification
        prediction = int(fraud_score >= THRESHOLD)
        risk_level = "HIGH" if prediction == 1 else "LOW"

        message = (
            "Potential fraud detected. Manual review recommended."
            if prediction == 1
            else "No fraud flagged by the model."
        )

        return PredictionResponse(
            predicted_fraud=prediction,
            fraud_score=round(fraud_score, 4),
            risk_level=risk_level,
            decision_threshold=THRESHOLD,
            message=message
        )

    except Exception:
        logger.exception("Fraud prediction failed")
        raise HTTPException(
            status_code=500,
            detail="Unable to generate prediction"
        )