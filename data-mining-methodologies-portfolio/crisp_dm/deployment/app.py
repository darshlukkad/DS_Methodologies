"""
FastAPI deployment for Walmart Sales Forecasting model.

This API provides:
- /predict: Single or batch predictions
- /health: Health check endpoint
- /model-info: Model metadata and performance
- /drift-report: Data drift detection

Production considerations:
- Input validation with Pydantic
- Error handling and logging
- Model versioning
- Monitoring hooks
"""

from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel, Field, validator
from typing import List, Optional, Dict, Any
from datetime import datetime, date
import pandas as pd
import numpy as np
import joblib
import logging
from pathlib import Path
import json

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Walmart Sales Forecasting API",
    description="CRISP-DM methodology deployment for weekly sales predictions",
    version="1.0.0"
)

# Global model and pipeline (loaded once at startup)
MODEL = None
PIPELINE = None
MODEL_METADATA = {}


class PredictionRequest(BaseModel):
    """Schema for prediction input."""
    
    Store: int = Field(..., ge=1, le=45, description="Store number (1-45)")
    Dept: int = Field(..., ge=1, le=99, description="Department number (1-99)")
    Date: date = Field(..., description="Forecast date (YYYY-MM-DD)")
    IsHoliday: bool = Field(False, description="Is this week a holiday?")
    Temperature: Optional[float] = Field(None, ge=-20, le=120, description="Temperature (F)")
    Fuel_Price: Optional[float] = Field(None, ge=2.0, le=5.0, description="Fuel price ($)")
    CPI: Optional[float] = Field(None, ge=100, le=250, description="Consumer Price Index")
    Unemployment: Optional[float] = Field(None, ge=3.0, le=15.0, description="Unemployment rate (%)")
    Type: Optional[str] = Field("A", pattern="^[ABC]$", description="Store type (A, B, or C)")
    Size: Optional[int] = Field(150000, ge=30000, le=250000, description="Store size (sqft)")
    MarkDown1: Optional[float] = Field(None, ge=0, description="Markdown event 1")
    MarkDown2: Optional[float] = Field(None, ge=0, description="Markdown event 2")
    MarkDown3: Optional[float] = Field(None, ge=0, description="Markdown event 3")
    MarkDown4: Optional[float] = Field(None, ge=0, description="Markdown event 4")
    MarkDown5: Optional[float] = Field(None, ge=0, description="Markdown event 5")
    
    @validator('Date')
    def validate_date_range(cls, v):
        """Ensure date is within reasonable range."""
        if v.year < 2010 or v.year > 2030:
            raise ValueError(f"Date {v} is outside valid range (2010-2030)")
        return v
    
    class Config:
        schema_extra = {
            "example": {
                "Store": 1,
                "Dept": 1,
                "Date": "2012-11-02",
                "IsHoliday": False,
                "Temperature": 58.0,
                "Fuel_Price": 3.50,
                "CPI": 215.0,
                "Unemployment": 7.5,
                "Type": "A",
                "Size": 151315,
                "MarkDown1": 5000.0,
                "MarkDown2": 2000.0
            }
        }


class PredictionResponse(BaseModel):
    """Schema for prediction output."""
    
    Store: int
    Dept: int
    Date: date
    predicted_sales: float
    confidence_interval_lower: Optional[float] = None
    confidence_interval_upper: Optional[float] = None
    model_version: str
    timestamp: datetime


class BatchPredictionRequest(BaseModel):
    """Schema for batch predictions."""
    predictions: List[PredictionRequest]


@app.on_event("startup")
async def load_model():
    """Load model and pipeline at startup."""
    global MODEL, PIPELINE, MODEL_METADATA
    
    try:
        MODEL_PATH = Path("../models/final_model.joblib")
        PIPELINE_PATH = Path("../models/feature_pipeline.joblib")
        METADATA_PATH = Path("../models/model_metadata.json")
        
        if not MODEL_PATH.exists():
            logger.error(f"Model file not found: {MODEL_PATH}")
            raise FileNotFoundError("Model not found. Please train model first.")
        
        MODEL = joblib.load(MODEL_PATH)
        logger.info(f"✓ Model loaded from {MODEL_PATH}")
        
        if PIPELINE_PATH.exists():
            PIPELINE = joblib.load(PIPELINE_PATH)
            logger.info(f"✓ Pipeline loaded from {PIPELINE_PATH}")
        
        if METADATA_PATH.exists():
            with open(METADATA_PATH, 'r') as f:
                MODEL_METADATA = json.load(f)
            logger.info(f"✓ Metadata loaded")
        else:
            MODEL_METADATA = {
                "version": "1.0.0",
                "trained_date": datetime.now().isoformat(),
                "model_type": type(MODEL).__name__
            }
        
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        raise


def preprocess_input(request: PredictionRequest) -> pd.DataFrame:
    """
    Convert request to DataFrame and apply feature engineering.
    
    Args:
        request: PredictionRequest object
        
    Returns:
        DataFrame ready for model prediction
    """
    # Convert to dict then DataFrame
    data = request.dict()
    data['Date'] = pd.to_datetime(data['Date'])
    df = pd.DataFrame([data])
    
    # Apply feature engineering pipeline if available
    if PIPELINE:
        df = PIPELINE.transform(df)
    else:
        # Basic feature engineering (fallback)
        df['year'] = df['Date'].dt.year
        df['month'] = df['Date'].dt.month
        df['week'] = df['Date'].dt.isocalendar().week
        df['day_of_week'] = df['Date'].dt.dayofweek
        
        # One-hot encode Type
        df = pd.get_dummies(df, columns=['Type'], prefix='store_type')
        
        # Fill missing markdowns with 0
        markdown_cols = [f'MarkDown{i}' for i in range(1, 6)]
        for col in markdown_cols:
            if col in df.columns:
                df[col].fillna(0, inplace=True)
    
    return df


@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "name": "Walmart Sales Forecasting API",
        "version": MODEL_METADATA.get("version", "unknown"),
        "status": "running",
        "model_type": MODEL_METADATA.get("model_type", "unknown"),
        "endpoints": {
            "/predict": "POST - Single prediction",
            "/predict/batch": "POST - Batch predictions",
            "/health": "GET - Health check",
            "/model-info": "GET - Model metadata",
            "/drift-report": "POST - Data drift analysis"
        }
    }


@app.get("/health")
async def health_check():
    """Health check endpoint for load balancers."""
    model_loaded = MODEL is not None
    
    return {
        "status": "healthy" if model_loaded else "unhealthy",
        "model_loaded": model_loaded,
        "timestamp": datetime.now().isoformat()
    }


@app.get("/model-info")
async def model_info():
    """Return model metadata and performance metrics."""
    return {
        "model_metadata": MODEL_METADATA,
        "model_type": type(MODEL).__name__ if MODEL else "not loaded",
        "features_count": MODEL.n_features_in_ if hasattr(MODEL, 'n_features_in_') else "unknown",
        "api_version": "1.0.0"
    }


@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    """
    Make a single sales prediction.
    
    Args:
        request: PredictionRequest with store, dept, date, and features
        
    Returns:
        PredictionResponse with predicted sales and metadata
    """
    try:
        # Validate model is loaded
        if MODEL is None:
            raise HTTPException(status_code=503, detail="Model not loaded")
        
        # Preprocess input
        df = preprocess_input(request)
        
        # Make prediction
        prediction = MODEL.predict(df)[0]
        
        # TODO: Calculate confidence intervals (requires quantile regression or ensemble)
        # For now, use simple ±20% as placeholder
        ci_lower = prediction * 0.8
        ci_upper = prediction * 1.2
        
        # Log prediction for monitoring
        logger.info(
            f"Prediction: Store={request.Store}, Dept={request.Dept}, "
            f"Date={request.Date}, Predicted=${prediction:.2f}"
        )
        
        return PredictionResponse(
            Store=request.Store,
            Dept=request.Dept,
            Date=request.Date,
            predicted_sales=float(prediction),
            confidence_interval_lower=float(ci_lower),
            confidence_interval_upper=float(ci_upper),
            model_version=MODEL_METADATA.get("version", "1.0.0"),
            timestamp=datetime.now()
        )
        
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


@app.post("/predict/batch")
async def predict_batch(request: BatchPredictionRequest):
    """
    Make batch predictions.
    
    Args:
        request: BatchPredictionRequest with list of predictions
        
    Returns:
        List of PredictionResponse objects
    """
    try:
        if MODEL is None:
            raise HTTPException(status_code=503, detail="Model not loaded")
        
        results = []
        for pred_request in request.predictions:
            result = await predict(pred_request)
            results.append(result)
        
        logger.info(f"Batch prediction: {len(results)} predictions made")
        
        return {
            "predictions": results,
            "count": len(results),
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Batch prediction error: {e}")
        raise HTTPException(status_code=500, detail=f"Batch prediction failed: {str(e)}")


@app.post("/drift-report")
async def drift_report(background_tasks: BackgroundTasks):
    """
    Generate data drift report (placeholder for Evidently integration).
    
    In production, this would:
    1. Load recent prediction data
    2. Compare to training data distribution
    3. Generate Evidently drift report
    4. Alert if significant drift detected
    """
    # TODO: Implement with Evidently
    logger.info("Drift report requested (not yet implemented)")
    
    return {
        "status": "not_implemented",
        "message": "Drift detection will be implemented with Evidently library",
        "recommended_action": "Manually compare recent predictions to training data"
    }


if __name__ == "__main__":
    import uvicorn
    
    # Run with: python app.py
    # Or: uvicorn app:app --reload --host 0.0.0.0 --port 8000
    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
