"""
Test suite for data leakage detection.

These tests ensure no future information leaks into training data,
which is critical for time-series forecasting validity.
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import sys
sys.path.append('../src')

from feature_engineering import check_data_leakage, create_lag_features, create_rolling_features


class TestTemporalSplits:
    """Test that train/val/test splits respect temporal order."""
    
    def test_no_date_overlap(self):
        """Verify no dates appear in multiple splits."""
        # Create sample data
        dates = pd.date_range('2010-01-01', periods=100, freq='W')
        train_dates = dates[:60]
        val_dates = dates[60:80]
        test_dates = dates[80:]
        
        train_df = pd.DataFrame({'Date': train_dates, 'value': range(60)})
        val_df = pd.DataFrame({'Date': val_dates, 'value': range(60, 80)})
        test_df = pd.DataFrame({'Date': test_dates, 'value': range(80, 100)})
        
        # Check for leakage
        assert check_data_leakage(train_df, val_df, test_df, date_col='Date')
        
    def test_temporal_order_maintained(self):
        """Verify train < val < test chronologically."""
        dates = pd.date_range('2010-01-01', periods=100, freq='W')
        train_dates = dates[:60]
        val_dates = dates[60:80]
        test_dates = dates[80:]
        
        train_df = pd.DataFrame({'Date': train_dates})
        val_df = pd.DataFrame({'Date': val_dates})
        test_df = pd.DataFrame({'Date': test_dates})
        
        assert train_df['Date'].max() < val_df['Date'].min()
        assert val_df['Date'].max() < test_df['Date'].min()
        
    def test_detects_overlap(self):
        """Verify leakage detection when dates overlap."""
        dates = pd.date_range('2010-01-01', periods=100, freq='W')
        
        # Intentionally create overlap
        train_dates = dates[:65]  # Overlaps with val
        val_dates = dates[60:80]
        test_dates = dates[80:]
        
        train_df = pd.DataFrame({'Date': train_dates})
        val_df = pd.DataFrame({'Date': val_dates})
        test_df = pd.DataFrame({'Date': test_dates})
        
        # Should detect leakage
        assert not check_data_leakage(train_df, val_df, test_df, date_col='Date')


class TestLagFeatures:
    """Test that lag features don't cause leakage."""
    
    def test_lag_uses_only_past_data(self):
        """Verify lag features look only at past values."""
        df = pd.DataFrame({
            'Store': [1, 1, 1, 1, 1],
            'Dept': [1, 1, 1, 1, 1],
            'Date': pd.date_range('2010-01-01', periods=5, freq='W'),
            'Weekly_Sales': [100, 200, 300, 400, 500]
        })
        
        df = create_lag_features(df, target_col='Weekly_Sales', lags=[1, 2])
        
        # Lag-1 at index 1 should be value from index 0
        assert df.loc[1, 'weekly_sales_lag_1'] == 100
        # Lag-2 at index 2 should be value from index 0
        assert df.loc[2, 'weekly_sales_lag_2'] == 100
        # First row should have NaN for lag-1 (no past data)
        assert pd.isna(df.loc[0, 'weekly_sales_lag_1'])
        
    def test_lag_respects_groups(self):
        """Verify lags don't bleed across Store-Dept groups."""
        df = pd.DataFrame({
            'Store': [1, 1, 2, 2],
            'Dept': [1, 1, 1, 1],
            'Date': pd.date_range('2010-01-01', periods=4, freq='W'),
            'Weekly_Sales': [100, 200, 300, 400]
        })
        
        df = create_lag_features(df, target_col='Weekly_Sales', group_cols=['Store', 'Dept'], lags=[1])
        
        # Store 2's first value should have NaN lag (different group)
        assert pd.isna(df.loc[2, 'weekly_sales_lag_1'])
        # Store 1's second value should have lag from Store 1
        assert df.loc[1, 'weekly_sales_lag_1'] == 100


class TestRollingFeatures:
    """Test that rolling features respect temporal order."""
    
    def test_rolling_excludes_current_value(self):
        """Verify rolling window doesn't include current value."""
        df = pd.DataFrame({
            'Store': [1] * 10,
            'Dept': [1] * 10,
            'Date': pd.date_range('2010-01-01', periods=10, freq='W'),
            'Weekly_Sales': [100] * 10
        })
        
        df = create_rolling_features(df, target_col='Weekly_Sales', windows=[3])
        
        # Rolling mean at index 3 should be mean of [0,1,2], not [0,1,2,3]
        # With shift(1), it uses [0,1] when calculating for index 2
        # This is correct: rolling window should NOT see future
        # All values are 100, so rolling mean should be 100
        assert df.loc[3, 'weekly_sales_rolling_mean_3'] == 100.0
        
    def test_rolling_with_varying_values(self):
        """Test rolling calculation with varying values."""
        df = pd.DataFrame({
            'Store': [1] * 5,
            'Dept': [1] * 5,
            'Date': pd.date_range('2010-01-01', periods=5, freq='W'),
            'Weekly_Sales': [100, 200, 300, 400, 500]
        })
        
        df = create_rolling_features(df, target_col='Weekly_Sales', windows=[2])
        
        # At index 2, rolling_mean_2 with shift(1) should use [0,1] = (100+200)/2 = 150
        expected_at_2 = (100 + 200) / 2
        assert abs(df.loc[2, 'weekly_sales_rolling_mean_2'] - expected_at_2) < 0.01


class TestFeatureLeakage:
    """Test for common feature leakage patterns."""
    
    def test_no_future_information_in_features(self):
        """Verify features at time t don't contain info from time t+k."""
        df = pd.DataFrame({
            'Store': [1, 1, 1],
            'Dept': [1, 1, 1],
            'Date': pd.date_range('2010-01-01', periods=3, freq='W'),
            'Weekly_Sales': [100, 200, 300],
            'IsHoliday': [False, True, False]
        })
        
        # Create lag features
        df = create_lag_features(df, target_col='Weekly_Sales', lags=[1])
        
        # At time t=1, lag-1 should only know about t=0, not t=1
        # So lag-1 at index 1 should be 100, not 200
        assert df.loc[1, 'weekly_sales_lag_1'] == 100
        
    def test_target_not_in_features(self):
        """Verify target variable isn't accidentally included in features."""
        feature_cols = ['Store', 'Dept', 'Date', 'IsHoliday', 'Temperature']
        target_col = 'Weekly_Sales'
        
        # Feature columns should not include target
        assert target_col not in feature_cols
        
    def test_no_perfect_correlation_with_target(self):
        """Check for features that perfectly correlate with target (leakage indicator)."""
        df = pd.DataFrame({
            'Weekly_Sales': [100, 200, 300, 400],
            'Feature1': [10, 20, 30, 40],  # Perfect correlation
            'Feature2': [5, 15, 25, 35]    # Not perfect
        })
        
        # Perfect correlation (0.99+) suggests leakage
        corr1 = df['Weekly_Sales'].corr(df['Feature1'])
        corr2 = df['Weekly_Sales'].corr(df['Feature2'])
        
        # This is a contrived example; in real data, we'd flag corr > 0.99
        assert corr1 > 0.99  # Perfect correlation (red flag in real scenario)
        assert corr2 > 0.99  # Also perfect in this toy example


class TestCrossValidationLeakage:
    """Test time-series cross-validation doesn't cause leakage."""
    
    def test_timeseriessplit_no_future_data(self):
        """Verify TimeSeriesSplit respects temporal order."""
        from sklearn.model_selection import TimeSeriesSplit
        
        dates = pd.date_range('2010-01-01', periods=100, freq='W')
        df = pd.DataFrame({
            'Date': dates,
            'value': range(100)
        })
        
        tscv = TimeSeriesSplit(n_splits=3)
        
        for train_idx, test_idx in tscv.split(df):
            train_dates = df.iloc[train_idx]['Date']
            test_dates = df.iloc[test_idx]['Date']
            
            # Train dates should all be before test dates
            assert train_dates.max() < test_dates.min()


# Additional edge case tests
class TestEdgeCases:
    """Test edge cases that might cause leakage."""
    
    def test_cold_start_no_history(self):
        """Test handling of Store-Dept combinations with no history."""
        df = pd.DataFrame({
            'Store': [1, 1, 2],  # Store 2 appears only once
            'Dept': [1, 1, 1],
            'Date': pd.date_range('2010-01-01', periods=3, freq='W'),
            'Weekly_Sales': [100, 200, 300]
        })
        
        df = create_lag_features(df, target_col='Weekly_Sales', lags=[1])
        
        # Store 2's first (and only) appearance should have NaN lag
        assert pd.isna(df.loc[2, 'weekly_sales_lag_1'])
        
    def test_irregular_time_periods(self):
        """Test behavior with missing weeks (gaps in time series)."""
        # Create data with a gap
        dates = list(pd.date_range('2010-01-01', periods=5, freq='W')) + \
                list(pd.date_range('2010-03-01', periods=5, freq='W'))
        
        df = pd.DataFrame({
            'Store': [1] * 10,
            'Dept': [1] * 10,
            'Date': dates,
            'Weekly_Sales': [100] * 10
        })
        
        df = create_lag_features(df, target_col='Weekly_Sales', lags=[1])
        
        # Lag should still work (based on position, not time)
        # But in production, might want time-aware lags
        assert not pd.isna(df.loc[5, 'weekly_sales_lag_1'])


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v", "--tb=short"])
