"""
Modeling utilities for Walmart Sales Forecasting.

This module provides functions for:
- Model training and evaluation
- Hyperparameter tuning
- Cross-validation
- Performance metrics
- Model comparison
"""

from typing import Dict, List, Tuple, Any, Optional
import pandas as pd
import numpy as np
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.linear_model import Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb
import lightgbm as lgb
import joblib
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def smape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calculate Symmetric Mean Absolute Percentage Error (sMAPE).
    
    Args:
        y_true: Actual values
        y_pred: Predicted values
        
    Returns:
        sMAPE score (0-100%)
    """
    denominator = (np.abs(y_true) + np.abs(y_pred)) / 2
    # Avoid division by zero
    denominator = np.where(denominator == 0, 1e-10, denominator)
    return np.mean(np.abs(y_true - y_pred) / denominator) * 100


def wape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calculate Weighted Absolute Percentage Error (WAPE).
    
    Args:
        y_true: Actual values
        y_pred: Predicted values
        
    Returns:
        WAPE score (0-100%)
    """
    return np.sum(np.abs(y_true - y_pred)) / np.sum(np.abs(y_true)) * 100


def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """
    Calculate comprehensive regression metrics.
    
    Args:
        y_true: Actual values
        y_pred: Predicted values
        
    Returns:
        Dictionary of metrics
    """
    # Canonical (capitalized) keys for internal use and lowercase aliases for notebook compatibility
    metrics = {
        'MAE': mean_absolute_error(y_true, y_pred),
        'RMSE': np.sqrt(mean_squared_error(y_true, y_pred)),
        'R2': r2_score(y_true, y_pred),
        'sMAPE': smape(y_true, y_pred),
        'WAPE': wape(y_true, y_pred)
    }

    # Add lowercase aliases expected by the notebook (smape, wape, rmse, mae, r2)
    metrics.update({
        'mae': metrics['MAE'],
        'rmse': metrics['RMSE'],
        'r2': metrics['R2'],
        'smape': metrics['sMAPE'],
        'wape': metrics['WAPE']
    })

    return metrics


def naive_baseline_last_week(
    train: pd.DataFrame,
    val: pd.DataFrame,
    group_cols: List[str] = ['Store', 'Dept']
) -> Tuple[np.ndarray, Dict[str, float]]:
    """
    Naive baseline: Predict last week's sales.
    
    Args:
        train: Training data
        val: Validation data
        group_cols: Grouping columns
        
    Returns:
        Predictions and metrics
    """
    # Get last week's sales for each Store-Dept
    last_sales = train.groupby(group_cols)['Weekly_Sales'].last().to_dict()
    
    # Predict
    predictions = val.apply(
        lambda row: last_sales.get((row['Store'], row['Dept']), train['Weekly_Sales'].median()),
        axis=1
    ).values
    
    metrics = calculate_metrics(val['Weekly_Sales'].values, predictions)
    logger.info(f"Naive baseline (last week) - MAE: {metrics['MAE']:.2f}, sMAPE: {metrics['sMAPE']:.2f}%")
    
    return predictions, metrics


def naive_baseline_last_year(
    train: pd.DataFrame,
    val: pd.DataFrame,
    group_cols: List[str] = ['Store', 'Dept']
) -> Tuple[np.ndarray, Dict[str, float]]:
    """
    Naive baseline: Predict same week last year.
    
    Args:
        train: Training data
        val: Validation data
        group_cols: Grouping columns
        
    Returns:
        Predictions and metrics
    """
    train = train.copy()
    val = val.copy()
    
    # Create year-week identifier
    train['year_week'] = train['Date'].dt.year.astype(str) + '_' + train['week'].astype(str)
    val['year_week'] = val['Date'].dt.year.astype(str) + '_' + val['week'].astype(str)
    
    # For each validation row, find same week last year
    predictions = []
    for idx, row in val.iterrows():
        prev_year = row['year'] - 1
        prev_year_week = f"{prev_year}_{row['week']}"
        
        # Find matching record
        match = train[
            (train['Store'] == row['Store']) &
            (train['Dept'] == row['Dept']) &
            (train['year_week'] == prev_year_week)
        ]
        
        if len(match) > 0:
            predictions.append(match['Weekly_Sales'].iloc[0])
        else:
            # Fallback to median
            predictions.append(train['Weekly_Sales'].median())
    
    predictions = np.array(predictions)
    metrics = calculate_metrics(val['Weekly_Sales'].values, predictions)
    logger.info(f"Naive baseline (last year) - MAE: {metrics['MAE']:.2f}, sMAPE: {metrics['sMAPE']:.2f}%")
    
    return predictions, metrics


def train_ridge(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    X_val: pd.DataFrame,
    y_val: np.ndarray,
    alpha: float = 1.0
) -> Tuple[Ridge, np.ndarray, Dict[str, float]]:
    """
    Train Ridge regression model.
    
    Args:
        X_train, y_train: Training data
        X_val, y_val: Validation data
        alpha: Regularization strength
        
    Returns:
        Trained model and validation metrics
    """
    model = Ridge(alpha=alpha, random_state=42)
    model.fit(X_train, y_train)
    
    predictions = model.predict(X_val)
    metrics = calculate_metrics(y_val, predictions)

    logger.info(f"Ridge (alpha={alpha}) - MAE: {metrics['MAE']:.2f}, sMAPE: {metrics['sMAPE']:.2f}%")
    return model, predictions, metrics


def train_random_forest(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    X_val: pd.DataFrame,
    y_val: np.ndarray,
    n_estimators: int = 100,
    max_depth: int = 10
) -> Tuple[RandomForestRegressor, np.ndarray, Dict[str, float]]:
    """
    Train Random Forest model.
    
    Args:
        X_train, y_train: Training data
        X_val, y_val: Validation data
        n_estimators: Number of trees
        max_depth: Maximum tree depth
        
    Returns:
        Trained model and validation metrics
    """
    model = RandomForestRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train, y_train)
    
    predictions = model.predict(X_val)
    metrics = calculate_metrics(y_val, predictions)

    logger.info(f"Random Forest - MAE: {metrics['MAE']:.2f}, sMAPE: {metrics['sMAPE']:.2f}%")
    return model, predictions, metrics


def train_xgboost(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    X_val: pd.DataFrame,
    y_val: np.ndarray,
    params: Optional[Dict[str, Any]] = None
) -> Tuple[xgb.XGBRegressor, np.ndarray, Dict[str, float]]:
    """
    Train XGBoost model.
    
    Args:
        X_train, y_train: Training data
        X_val, y_val: Validation data
        params: XGBoost parameters
        
    Returns:
        Trained model and validation metrics
    """
    if params is None:
        params = {
            'max_depth': 6,
            'learning_rate': 0.1,
            'n_estimators': 100,
            'objective': 'reg:squarederror',
            'random_state': 42,
            'n_jobs': -1
        }
    
    model = xgb.XGBRegressor(**params)
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=False
    )
    
    predictions = model.predict(X_val)
    metrics = calculate_metrics(y_val, predictions)

    logger.info(f"XGBoost - MAE: {metrics['MAE']:.2f}, sMAPE: {metrics['sMAPE']:.2f}%")
    return model, predictions, metrics


def train_lightgbm(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    X_val: pd.DataFrame,
    y_val: np.ndarray,
    params: Optional[Dict[str, Any]] = None
) -> Tuple[lgb.LGBMRegressor, np.ndarray, Dict[str, float]]:
    """
    Train LightGBM model.
    
    Args:
        X_train, y_train: Training data
        X_val, y_val: Validation data
        params: LightGBM parameters
        
    Returns:
        Trained model and validation metrics
    """
    if params is None:
        params = {
            'max_depth': 6,
            'learning_rate': 0.1,
            'n_estimators': 100,
            'objective': 'regression',
            'random_state': 42,
            'n_jobs': -1,
            'verbose': -1
        }
    
    model = lgb.LGBMRegressor(**params)
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        callbacks=[lgb.log_evaluation(period=0)]
    )
    
    predictions = model.predict(X_val)
    metrics = calculate_metrics(y_val, predictions)

    logger.info(f"LightGBM - MAE: {metrics['MAE']:.2f}, sMAPE: {metrics['sMAPE']:.2f}%")
    return model, predictions, metrics


def time_series_cv(
    X: pd.DataFrame,
    y: np.ndarray,
    model_func: callable,
    n_splits: int = 5,
    **model_kwargs
) -> List[Dict[str, float]]:
    """
    Perform time series cross-validation.
    
    Args:
        X, y: Features and target
        model_func: Function to train model
        n_splits: Number of CV splits
        **model_kwargs: Arguments for model_func
        
    Returns:
        List of metrics for each fold
    """
    tscv = TimeSeriesSplit(n_splits=n_splits)
    fold_metrics = []
    
    for fold, (train_idx, val_idx) in enumerate(tscv.split(X), 1):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]
        
        res = model_func(X_train, y_train, X_val, y_val, **model_kwargs)
        # support functions that return (model, predictions, metrics) or (model, metrics) or (predictions, metrics)
        if isinstance(res, tuple) and len(res) == 3:
            _, _, metrics = res
        elif isinstance(res, tuple) and len(res) == 2:
            # could be (model, metrics) or (preds, metrics)
            _, metrics = res
        else:
            raise ValueError("model_func must return a tuple of length 2 or 3")
        metrics['fold'] = fold
        fold_metrics.append(metrics)
        
        logger.info(f"Fold {fold}/{n_splits} - MAE: {metrics['MAE']:.2f}")
    
    return fold_metrics


def save_model(model: Any, filepath: str) -> None:
    """
    Save trained model to disk.
    
    Args:
        model: Trained model
        filepath: Path to save model
    """
    Path(filepath).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, filepath)
    logger.info(f"✓ Model saved to {filepath}")


def load_model(filepath: str) -> Any:
    """
    Load trained model from disk.
    
    Args:
        filepath: Path to saved model
        
    Returns:
        Loaded model
    """
    model = joblib.load(filepath)
    logger.info(f"✓ Model loaded from {filepath}")
    return model


if __name__ == "__main__":
    print("Modeling utilities loaded successfully")
