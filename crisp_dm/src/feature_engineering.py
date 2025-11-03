"""
Feature engineering for Walmart Sales Forecasting.

This module creates time-series features including:
- Temporal features (date parts, holidays)
- Lag features (previous weeks' sales)
- Rolling window features (moving averages)
- Holiday and event features
"""

from typing import List, Optional, Tuple
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# Holiday definitions
HOLIDAYS = {
    'Super Bowl': [
        '2010-02-12', '2011-02-11', '2012-02-10', '2013-02-08'
    ],
    'Labor Day': [
        '2010-09-10', '2011-09-09', '2012-09-07', '2013-09-06'
    ],
    'Thanksgiving': [
        '2010-11-26', '2011-11-25', '2012-11-23', '2013-11-29'
    ],
    'Christmas': [
        '2010-12-31', '2011-12-30', '2012-12-28', '2013-12-27'
    ]
}


def create_temporal_features(df: pd.DataFrame, date_col: str = 'Date') -> pd.DataFrame:
    """
    Create date-based features.
    
    Args:
        df: DataFrame with date column
        date_col: Name of date column
        
    Returns:
        DataFrame with additional temporal features
    """
    df = df.copy()
    
    df['year'] = df[date_col].dt.year
    df['month'] = df[date_col].dt.month
    df['week'] = df[date_col].dt.isocalendar().week
    df['day_of_week'] = df[date_col].dt.dayofweek
    df['day_of_year'] = df[date_col].dt.dayofyear
    df['quarter'] = df[date_col].dt.quarter
    
    # Boolean features
    df['is_month_start'] = df[date_col].dt.is_month_start.astype(int)
    df['is_month_end'] = df[date_col].dt.is_month_end.astype(int)
    df['is_quarter_start'] = df[date_col].dt.is_quarter_start.astype(int)
    df['is_quarter_end'] = df[date_col].dt.is_quarter_end.astype(int)
    
    logger.info(f"✓ Created {11} temporal features")
    return df


def create_holiday_features(df: pd.DataFrame, date_col: str = 'Date') -> pd.DataFrame:
    """
    Create holiday-specific features.
    
    Args:
        df: DataFrame with date column
        date_col: Name of date column
        
    Returns:
        DataFrame with holiday features
    """
    df = df.copy()
    
    # Convert dates to datetime for comparison
    df[date_col] = pd.to_datetime(df[date_col])
    
    # Specific holiday flags
    for holiday_name, dates in HOLIDAYS.items():
        col_name = f'is_{holiday_name.lower().replace(" ", "_")}'
        holiday_dates = pd.to_datetime(dates)
        df[col_name] = df[date_col].isin(holiday_dates).astype(int)
    
    # Weeks before/after holidays
    df['holiday_week_before'] = 0
    df['holiday_week_after'] = 0
    
    for dates in HOLIDAYS.values():
        holiday_dates = pd.to_datetime(dates)
        for hdate in holiday_dates:
            week_before = hdate - timedelta(days=7)
            week_after = hdate + timedelta(days=7)
            
            df.loc[df[date_col] == week_before, 'holiday_week_before'] = 1
            df.loc[df[date_col] == week_after, 'holiday_week_after'] = 1
    
    # Days until major holidays
    christmas_dates = pd.to_datetime(HOLIDAYS['Christmas'])
    thanksgiving_dates = pd.to_datetime(HOLIDAYS['Thanksgiving'])
    
    df['weeks_to_christmas'] = 52  # default
    df['weeks_to_thanksgiving'] = 52
    
    for idx, row in df.iterrows():
        curr_date = row[date_col]
        curr_year = curr_date.year
        
        # Find next Christmas
        xmas = christmas_dates[christmas_dates.year == curr_year]
        if len(xmas) > 0:
            xmas_date = xmas[0] if isinstance(xmas, pd.DatetimeIndex) else xmas.iloc[0]
            days_diff = (xmas_date - curr_date).days
            df.at[idx, 'weeks_to_christmas'] = max(0, days_diff // 7)
        
        # Find next Thanksgiving
        tgiving = thanksgiving_dates[thanksgiving_dates.year == curr_year]
        if len(tgiving) > 0:
            tgiving_date = tgiving[0] if isinstance(tgiving, pd.DatetimeIndex) else tgiving.iloc[0]
            days_diff = (tgiving_date - curr_date).days
            df.at[idx, 'weeks_to_thanksgiving'] = max(0, days_diff // 7)
    
    logger.info(f"✓ Created holiday features")
    return df


def create_lag_features(
    df: pd.DataFrame,
    target_col: str = 'Weekly_Sales',
    group_cols: List[str] = ['Store', 'Dept'],
    lags: List[int] = [1, 2, 4, 52]
) -> pd.DataFrame:
    """
    Create lag features (previous periods' values).
    
    Args:
        df: DataFrame sorted by date
        target_col: Column to create lags for
        group_cols: Columns to group by (e.g., Store, Dept)
        lags: List of lag periods (in weeks)
        
    Returns:
        DataFrame with lag features
    """
    df = df.copy()
    
    # Skip if target column doesn't exist (e.g., test data)
    if target_col not in df.columns:
        logger.info(f"⚠ Skipping lag features - {target_col} not found in DataFrame")
        # Create empty lag columns filled with NaN
        for lag in lags:
            col_name = f'{target_col.lower()}_lag_{lag}'
            df[col_name] = pd.NA
        return df
    
    df = df.sort_values(group_cols + ['Date'])
    
    for lag in lags:
        col_name = f'{target_col.lower()}_lag_{lag}'
        df[col_name] = df.groupby(group_cols)[target_col].shift(lag)
    
    logger.info(f"✓ Created {len(lags)} lag features")
    return df


def create_rolling_features(
    df: pd.DataFrame,
    target_col: str = 'Weekly_Sales',
    group_cols: List[str] = ['Store', 'Dept'],
    windows: List[int] = [4, 8, 52]
) -> pd.DataFrame:
    """
    Create rolling window features (moving averages, std).
    
    Args:
        df: DataFrame sorted by date
        target_col: Column to compute rolling stats for
        group_cols: Columns to group by
        windows: List of window sizes (in weeks)
        
    Returns:
        DataFrame with rolling features
    """
    df = df.copy()
    
    # Skip if target column doesn't exist (e.g., test data)
    if target_col not in df.columns:
        logger.info(f"⚠ Skipping rolling features - {target_col} not found in DataFrame")
        # Create empty rolling columns filled with NaN
        for window in windows:
            col_mean = f'{target_col.lower()}_rolling_mean_{window}'
            col_std = f'{target_col.lower()}_rolling_std_{window}'
            df[col_mean] = pd.NA
            df[col_std] = pd.NA
        return df
    
    df = df.sort_values(group_cols + ['Date'])
    
    for window in windows:
        # Rolling mean
        col_mean = f'{target_col.lower()}_rolling_mean_{window}'
        df[col_mean] = df.groupby(group_cols)[target_col].transform(
            lambda x: x.shift(1).rolling(window=window, min_periods=1).mean()
        )
        
        # Rolling std
        col_std = f'{target_col.lower()}_rolling_std_{window}'
        df[col_std] = df.groupby(group_cols)[target_col].transform(
            lambda x: x.shift(1).rolling(window=window, min_periods=1).std()
        )
        
        # Handle NaN in std (when window has 1 value)
        df[col_std].fillna(0, inplace=True)
    
    logger.info(f"✓ Created {len(windows) * 2} rolling features")
    return df


def create_markdown_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create features from markdown events.
    
    Args:
        df: DataFrame with MarkDown1-5 columns
        
    Returns:
        DataFrame with markdown aggregation features
    """
    df = df.copy()
    
    markdown_cols = [f'MarkDown{i}' for i in range(1, 6)]
    
    # Fill NaN with 0 (no markdown)
    for col in markdown_cols:
        if col in df.columns:
            df[col].fillna(0, inplace=True)
    
    # Total markdown amount
    df['markdown_total'] = df[markdown_cols].sum(axis=1)
    
    # Number of active markdowns
    df['markdown_active_count'] = (df[markdown_cols] > 0).sum(axis=1)
    
    # Markdown intensity (relative to store size)
    if 'Size' in df.columns:
        df['markdown_per_sqft'] = df['markdown_total'] / (df['Size'] + 1)
    
    logger.info(f"✓ Created markdown features")
    return df


def create_store_dept_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create store and department categorical features.
    
    Args:
        df: DataFrame with Store, Dept, Type, Size
        
    Returns:
        DataFrame with encoded features
    """
    df = df.copy()
    
    # One-hot encode store type
    if 'Type' in df.columns:
        df = pd.get_dummies(df, columns=['Type'], prefix='store_type')
    
    # Normalize store size
    if 'Size' in df.columns:
        df['size_normalized'] = (df['Size'] - df['Size'].mean()) / df['Size'].std()
        
        # Size quartiles
        df['size_quartile'] = pd.qcut(
            df['Size'], 
            q=4, 
            labels=['small', 'medium', 'large', 'xlarge']
        )
        df = pd.get_dummies(df, columns=['size_quartile'], prefix='size')
    
    # Department categories (group similar departments)
    if 'Dept' in df.columns:
        df['dept_category'] = pd.cut(
            df['Dept'],
            bins=[0, 20, 40, 60, 80, 100],
            labels=['cat_1', 'cat_2', 'cat_3', 'cat_4', 'cat_5']
        )
        df = pd.get_dummies(df, columns=['dept_category'], prefix='dept')
    
    # Interaction: Department × Holiday
    if 'IsHoliday' in df.columns and 'Dept' in df.columns:
        df['dept_x_is_holiday'] = df['Dept'] * df['IsHoliday'].astype(int)
    
    logger.info(f"✓ Created store/dept features")
    return df


def engineer_all_features(
    df: pd.DataFrame,
    is_train: bool = True,
    target_col: str = 'Weekly_Sales'
) -> pd.DataFrame:
    """
    Apply all feature engineering steps.
    
    Args:
        df: Input DataFrame
        is_train: Whether this is training data (has target)
        target_col: Name of target column
        
    Returns:
        DataFrame with all engineered features
    """
    logger.info("Starting feature engineering...")
    
    df = create_temporal_features(df)
    df = create_holiday_features(df)
    df = create_markdown_features(df)
    df = create_store_dept_features(df)
    
    # Create lag/rolling features only if training data
    if is_train and target_col in df.columns:
        df = create_lag_features(df, target_col=target_col)
        df = create_rolling_features(df, target_col=target_col)
    
    logger.info(f"✓ Feature engineering complete. Shape: {df.shape}")
    return df


def check_data_leakage(
    train: pd.DataFrame,
    val: pd.DataFrame,
    test: pd.DataFrame,
    date_col: str = 'Date'
) -> bool:
    """
    Check for data leakage between splits.
    
    Args:
        train, val, test: DataFrames for each split
        date_col: Date column name
        
    Returns:
        True if no leakage detected
    """
    train_dates = set(train[date_col])
    val_dates = set(val[date_col])
    test_dates = set(test[date_col])
    
    train_val_overlap = train_dates & val_dates
    train_test_overlap = train_dates & test_dates
    val_test_overlap = val_dates & test_dates
    
    if train_val_overlap:
        logger.error(f"❌ Leakage: {len(train_val_overlap)} dates overlap between train/val")
        return False
    
    if train_test_overlap:
        logger.error(f"❌ Leakage: {len(train_test_overlap)} dates overlap between train/test")
        return False
    
    if val_test_overlap:
        logger.error(f"❌ Leakage: {len(val_test_overlap)} dates overlap between val/test")
        return False
    
    logger.info("✓ No data leakage detected")
    return True


if __name__ == "__main__":
    # Example usage
    print("Feature engineering module loaded successfully")
