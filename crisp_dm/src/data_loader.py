"""
Data loading utilities for Walmart Sales Forecasting.

This module handles downloading data from Kaggle, unzipping, 
and loading into pandas DataFrames with proper types.
"""

import os
import zipfile
from pathlib import Path
from typing import Dict, Tuple
import subprocess
import pandas as pd
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def ensure_kaggle_configured() -> bool:
    """
    Check if Kaggle API is properly configured.
    
    Returns:
        bool: True if kaggle.json exists in ~/.kaggle/
    """
    kaggle_path = Path.home() / '.kaggle' / 'kaggle.json'
    if not kaggle_path.exists():
        logger.error(
            "Kaggle API token not found. "
            "Please download kaggle.json from https://www.kaggle.com/account "
            "and place it in ~/.kaggle/"
        )
        return False
    
    # Set correct permissions (required by Kaggle API)
    os.chmod(kaggle_path, 0o600)
    logger.info("✓ Kaggle API configured")
    return True


def download_walmart_data(data_dir: str = "data/raw") -> bool:
    """
    Download Walmart Sales Forecasting competition data from Kaggle.
    
    Args:
        data_dir: Directory to store downloaded files
        
    Returns:
        bool: True if download successful
    """
    if not ensure_kaggle_configured():
        return False
    
    data_path = Path(data_dir)
    data_path.mkdir(parents=True, exist_ok=True)
    
    # Check if files already exist
    required_files = ['train.csv', 'test.csv', 'stores.csv', 'features.csv']
    if all((data_path / f).exists() for f in required_files):
        logger.info("✓ Walmart data already exists")
        return True
    
    logger.info("Downloading Walmart Sales Forecasting data from Kaggle...")
    
    try:
        # Download using Kaggle API
        result = subprocess.run(
            [
                'kaggle', 'competitions', 'download',
                '-c', 'walmart-recruiting-store-sales-forecasting',
                '-p', data_dir
            ],
            capture_output=True,
            text=True,
            check=True
        )
        logger.info("✓ Download complete")
        
        # Unzip files
        zip_files = list(data_path.glob('*.zip'))
        for zip_file in zip_files:
            logger.info(f"Extracting {zip_file.name}...")
            with zipfile.ZipFile(zip_file, 'r') as zip_ref:
                zip_ref.extractall(data_path)
            zip_file.unlink()  # Remove zip after extraction
        
        logger.info("✓ Data extraction complete")
        return True
        
    except subprocess.CalledProcessError as e:
        logger.error(f"Failed to download data: {e.stderr}")
        return False
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        return False


def load_walmart_data(data_dir: str = "data/raw") -> Dict[str, pd.DataFrame]:
    """
    Load Walmart datasets into pandas DataFrames.
    
    Args:
        data_dir: Directory containing CSV files
        
    Returns:
        Dict with keys: 'train', 'test', 'stores', 'features'
    """
    data_path = Path(data_dir)
    
    logger.info("Loading Walmart datasets...")
    
    # Load train data
    train = pd.read_csv(
        data_path / 'train.csv',
        parse_dates=['Date'],
        dtype={'Store': int, 'Dept': int, 'IsHoliday': bool}
    )
    logger.info(f"✓ Loaded train: {train.shape}")
    
    # Load test data
    test = pd.read_csv(
        data_path / 'test.csv',
        parse_dates=['Date'],
        dtype={'Store': int, 'Dept': int, 'IsHoliday': bool}
    )
    logger.info(f"✓ Loaded test: {test.shape}")
    
    # Load stores metadata
    stores = pd.read_csv(
        data_path / 'stores.csv',
        dtype={'Store': int, 'Type': 'category', 'Size': int}
    )
    logger.info(f"✓ Loaded stores: {stores.shape}")
    
    # Load features (additional data)
    features = pd.read_csv(
        data_path / 'features.csv',
        parse_dates=['Date'],
        dtype={'Store': int, 'IsHoliday': bool}
    )
    logger.info(f"✓ Loaded features: {features.shape}")
    
    return {
        'train': train,
        'test': test,
        'stores': stores,
        'features': features
    }


def merge_datasets(
    train: pd.DataFrame,
    stores: pd.DataFrame,
    features: pd.DataFrame,
    test: pd.DataFrame = None
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Merge train/test with stores and features data.
    
    Args:
        train: Training data
        stores: Store metadata
        features: Additional features (temperature, markdowns, etc.)
        test: Test data (optional)
        
    Returns:
        Tuple of (train_merged, test_merged)
    """
    logger.info("Merging datasets...")
    
    # Merge train with stores
    train_merged = train.merge(stores, on='Store', how='left')
    
    # Merge with features
    train_merged = train_merged.merge(
        features, 
        on=['Store', 'Date', 'IsHoliday'], 
        how='left'
    )
    
    logger.info(f"✓ Train merged: {train_merged.shape}")
    
    # Merge test if provided
    test_merged = None
    if test is not None:
        test_merged = test.merge(stores, on='Store', how='left')
        test_merged = test_merged.merge(
            features,
            on=['Store', 'Date', 'IsHoliday'],
            how='left'
        )
        logger.info(f"✓ Test merged: {test_merged.shape}")
    
    return train_merged, test_merged


def get_data_summary(df: pd.DataFrame) -> pd.DataFrame:
    """
    Generate comprehensive data summary.
    
    Args:
        df: DataFrame to summarize
        
    Returns:
        Summary DataFrame with types, missing values, unique counts
    """
    summary = pd.DataFrame({
        'dtype': df.dtypes,
        'missing': df.isnull().sum(),
        'missing_pct': (df.isnull().sum() / len(df) * 100).round(2),
        'unique': df.nunique(),
        'sample_values': [df[col].dropna().head(3).tolist() for col in df.columns]
    })
    
    return summary


if __name__ == "__main__":
    # Test download and load
    success = download_walmart_data()
    if success:
        data = load_walmart_data()
        print("\n=== Data Summary ===")
        for name, df in data.items():
            print(f"\n{name.upper()}: {df.shape}")
            print(get_data_summary(df))
