"""
SEMMA Data Quality Tests
Comprehensive test suite for student performance data validation.
"""

import pandas as pd
import numpy as np
import pytest
from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from data_loader import StudentDataLoader, load_student_data
from preprocessing import StudentPreprocessor


class TestDataQuality:
    """Test data quality and integrity."""
    
    @pytest.fixture
    def sample_data(self):
        """Create sample data for testing."""
        loader = StudentDataLoader()
        df = loader._create_sample_data(n_samples=100)
        return loader.create_target(df)
    
    def test_no_missing_values(self, sample_data):
        """Test that there are no missing values."""
        missing = sample_data.isnull().sum()
        assert missing.sum() == 0, f"Found {missing.sum()} missing values"
    
    def test_target_distribution(self, sample_data):
        """Test that target has both classes."""
        assert 'Pass' in sample_data.columns
        assert sample_data['Pass'].nunique() == 2
        assert set(sample_data['Pass'].unique()) == {0, 1}
    
    def test_grade_ranges(self, sample_data):
        """Test that grades are in valid range [0, 20]."""
        for grade_col in ['G1', 'G2', 'G3']:
            assert (sample_data[grade_col] >= 0).all()
            assert (sample_data[grade_col] <= 20).all()
    
    def test_age_ranges(self, sample_data):
        """Test that ages are realistic."""
        assert (sample_data['age'] >= 15).all()
        assert (sample_data['age'] <= 22).all()
    
    def test_stratified_split(self, sample_data):
        """Test that stratified split preserves proportions."""
        loader = StudentDataLoader()
        train_df, val_df, test_df = loader.stratified_split(sample_data)
        
        # Check proportions
        assert len(train_df) / len(sample_data) == pytest.approx(0.6, abs=0.05)
        assert len(val_df) / len(sample_data) == pytest.approx(0.2, abs=0.05)
        assert len(test_df) / len(sample_data) == pytest.approx(0.2, abs=0.05)
        
        # Check pass rates are similar
        total_pass_rate = sample_data['Pass'].mean()
        train_pass_rate = train_df['Pass'].mean()
        val_pass_rate = val_df['Pass'].mean()
        test_pass_rate = test_df['Pass'].mean()
        
        assert train_pass_rate == pytest.approx(total_pass_rate, abs=0.1)
        assert val_pass_rate == pytest.approx(total_pass_rate, abs=0.1)
        assert test_pass_rate == pytest.approx(total_pass_rate, abs=0.1)


class TestFeatureEngineering:
    """Test feature engineering functions."""
    
    @pytest.fixture
    def sample_data(self):
        """Create sample data for testing."""
        loader = StudentDataLoader()
        df = loader._create_sample_data(n_samples=100)
        return loader.create_target(df)
    
    def test_feature_creation(self, sample_data):
        """Test that new features are created."""
        preprocessor = StudentPreprocessor()
        df_transformed = preprocessor.create_features(sample_data)
        
        # Check new features exist
        expected_features = [
            'parent_edu_avg',
            'parent_edu_max',
            'study_failure_interaction',
            'grade_improvement',
            'grade_avg'
        ]
        
        for feature in expected_features:
            assert feature in df_transformed.columns
    
    def test_no_data_leakage(self, sample_data):
        """Test that G3 is not used in features."""
        preprocessor = StudentPreprocessor()
        loader = StudentDataLoader()
        train_df, val_df, test_df = loader.stratified_split(sample_data)
        
        X_train, y_train, X_val, y_val, X_test, y_test = \
            preprocessor.full_pipeline(train_df, val_df, test_df)
        
        # Ensure G3 not in feature names
        assert 'G3' not in preprocessor.feature_names
        assert 'Pass' not in preprocessor.feature_names
    
    def test_scaling_preserves_shape(self, sample_data):
        """Test that scaling preserves data shape."""
        preprocessor = StudentPreprocessor()
        loader = StudentDataLoader()
        train_df, val_df, test_df = loader.stratified_split(sample_data)
        
        # Get shapes before preprocessing
        train_shape = train_df.shape[0]
        val_shape = val_df.shape[0]
        test_shape = test_df.shape[0]
        
        X_train, y_train, X_val, y_val, X_test, y_test = \
            preprocessor.full_pipeline(train_df, val_df, test_df)
        
        # Check shapes preserved
        assert X_train.shape[0] == train_shape
        assert X_val.shape[0] == val_shape
        assert X_test.shape[0] == test_shape
        
        # Check same number of features
        assert X_train.shape[1] == X_val.shape[1] == X_test.shape[1]


class TestModelReadiness:
    """Test that data is ready for modeling."""
    
    @pytest.fixture
    def prepared_data(self):
        """Prepare data for testing."""
        loader = StudentDataLoader()
        df = loader._create_sample_data(n_samples=100)
        df = loader.create_target(df)
        train_df, val_df, test_df = loader.stratified_split(df)
        
        preprocessor = StudentPreprocessor()
        X_train, y_train, X_val, y_val, X_test, y_test = \
            preprocessor.full_pipeline(train_df, val_df, test_df)
        
        return X_train, y_train, X_val, y_val, X_test, y_test
    
    def test_no_nans_in_features(self, prepared_data):
        """Test that features have no NaN values."""
        X_train, y_train, X_val, y_val, X_test, y_test = prepared_data
        
        assert not np.isnan(X_train).any()
        assert not np.isnan(X_val).any()
        assert not np.isnan(X_test).any()
    
    def test_no_nans_in_targets(self, prepared_data):
        """Test that targets have no NaN values."""
        X_train, y_train, X_val, y_val, X_test, y_test = prepared_data
        
        assert not np.isnan(y_train).any()
        assert not np.isnan(y_val).any()
        assert not np.isnan(y_test).any()
    
    def test_features_are_numeric(self, prepared_data):
        """Test that all features are numeric."""
        X_train, y_train, X_val, y_val, X_test, y_test = prepared_data
        
        assert X_train.dtype in [np.float64, np.float32, np.int64, np.int32]
        assert X_val.dtype in [np.float64, np.float32, np.int64, np.int32]
        assert X_test.dtype in [np.float64, np.float32, np.int64, np.int32]
    
    def test_targets_are_binary(self, prepared_data):
        """Test that targets are binary (0 or 1)."""
        X_train, y_train, X_val, y_val, X_test, y_test = prepared_data
        
        assert set(np.unique(y_train)) <= {0, 1}
        assert set(np.unique(y_val)) <= {0, 1}
        assert set(np.unique(y_test)) <= {0, 1}


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, '-v'])
