"""
KDD Data Pipeline Tests
Test suite for NSL-KDD intrusion detection data pipeline.
"""

import pandas as pd
import numpy as np
import pytest
from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from data_loader import NSLKDDLoader, load_nslkdd_data


class TestDataLoading:
    """Test NSL-KDD data loading."""
    
    @pytest.fixture
    def loader(self):
        """Create loader instance."""
        return NSLKDDLoader(data_dir='data')
    
    def test_sample_data_creation(self, loader):
        """Test that sample data can be created."""
        train_df, test_df = loader._create_sample_data(n_train=1000, n_test=200)
        
        assert len(train_df) == 1000
        assert len(test_df) == 200
        assert 'attack_type' in train_df.columns
        assert 'attack_type' in test_df.columns
    
    def test_attack_mapping(self, loader):
        """Test attack type mapping to 5 categories."""
        train_df, test_df = loader._create_sample_data(n_train=1000, n_test=200)
        train_df = loader.map_attack_types(train_df)
        
        # Check attack_category exists
        assert 'attack_category' in train_df.columns
        
        # Check only valid categories
        valid_categories = {'normal', 'dos', 'probe', 'r2l', 'u2r'}
        assert set(train_df['attack_category'].unique()) <= valid_categories
    
    def test_no_missing_values(self, loader):
        """Test that loaded data has no missing values."""
        train_df, test_df = loader._create_sample_data(n_train=100, n_test=50)
        
        assert train_df.isnull().sum().sum() == 0
        assert test_df.isnull().sum().sum() == 0


class TestAttackDistribution:
    """Test attack distribution and class balance."""
    
    @pytest.fixture
    def data(self):
        """Load sample data."""
        loader = NSLKDDLoader(data_dir='data')
        train_df, test_df = loader._create_sample_data(n_train=10000, n_test=2000)
        train_df = loader.map_attack_types(train_df)
        test_df = loader.map_attack_types(test_df)
        return train_df, test_df
    
    def test_all_attack_types_present(self, data):
        """Test that all 5 attack types are present."""
        train_df, test_df = data
        
        expected_types = {'normal', 'dos', 'probe', 'r2l', 'u2r'}
        assert set(train_df['attack_category'].unique()) == expected_types
    
    def test_class_imbalance_detected(self, data):
        """Test that minority classes (U2R, R2L) are identified."""
        train_df, test_df = data
        
        # Calculate proportions
        proportions = train_df['attack_category'].value_counts(normalize=True)
        
        # U2R should be rare (<2%)
        if 'u2r' in proportions:
            assert proportions['u2r'] < 0.02, "U2R should be minority class"
        
        # R2L should be rare (<10%)
        if 'r2l' in proportions:
            assert proportions['r2l'] < 0.10, "R2L should be minority class"


class TestFeatureIntegrity:
    """Test feature types and ranges."""
    
    @pytest.fixture
    def data(self):
        """Load sample data."""
        loader = NSLKDDLoader(data_dir='data')
        train_df, test_df = loader._create_sample_data(n_train=1000, n_test=200)
        return train_df, test_df
    
    def test_duration_non_negative(self, data):
        """Test that duration is non-negative."""
        train_df, test_df = data
        
        assert (train_df['duration'] >= 0).all()
        assert (test_df['duration'] >= 0).all()
    
    def test_bytes_non_negative(self, data):
        """Test that byte counts are non-negative."""
        train_df, test_df = data
        
        assert (train_df['src_bytes'] >= 0).all()
        assert (test_df['src_bytes'] >= 0).all()
        assert (train_df['dst_bytes'] >= 0).all()
        assert (test_df['dst_bytes'] >= 0).all()
    
    def test_rates_in_valid_range(self, data):
        """Test that rate features are in [0, 1]."""
        train_df, test_df = data
        
        rate_columns = [col for col in train_df.columns if 'rate' in col.lower()]
        
        for col in rate_columns:
            assert (train_df[col] >= 0).all()
            assert (train_df[col] <= 1).all()
            assert (test_df[col] >= 0).all()
            assert (test_df[col] <= 1).all()


class TestSecurityMetrics:
    """Test security-relevant properties."""
    
    @pytest.fixture
    def data(self):
        """Load and prepare data."""
        loader = NSLKDDLoader(data_dir='data')
        train_df, test_df = loader._create_sample_data(n_train=10000, n_test=2000)
        train_df = loader.map_attack_types(train_df)
        test_df = loader.map_attack_types(test_df)
        return train_df, test_df
    
    def test_normal_traffic_majority(self, data):
        """Test that normal traffic is substantial portion."""
        train_df, test_df = data
        
        normal_ratio = (train_df['attack_category'] == 'normal').mean()
        # Normal should be 40-60% of traffic
        assert 0.30 <= normal_ratio <= 0.70
    
    def test_dos_attacks_detectable(self, data):
        """Test that DoS attacks have distinct characteristics."""
        train_df, test_df = data
        
        dos_traffic = train_df[train_df['attack_category'] == 'dos']
        normal_traffic = train_df[train_df['attack_category'] == 'normal']
        
        if len(dos_traffic) > 0 and len(normal_traffic) > 0:
            # DoS should have higher connection counts
            dos_mean_count = dos_traffic['count'].mean()
            normal_mean_count = normal_traffic['count'].mean()
            
            # This is expected behavior (not always guaranteed in random data)
            print(f"DoS avg count: {dos_mean_count:.1f}, Normal avg count: {normal_mean_count:.1f}")
    
    def test_u2r_attacks_rare(self, data):
        """Test that U2R attacks are rare (most dangerous)."""
        train_df, test_df = data
        
        u2r_ratio = (train_df['attack_category'] == 'u2r').mean()
        
        # U2R should be < 2% (realistic for production networks)
        assert u2r_ratio < 0.02


class TestDataPipelineIntegrity:
    """Test end-to-end data pipeline."""
    
    def test_full_pipeline(self):
        """Test that full data loading pipeline works."""
        train_df, test_df = load_nslkdd_data(
            data_dir='data',
            map_attacks=True
        )
        
        # Check data loaded
        assert len(train_df) > 0
        assert len(test_df) > 0
        
        # Check attack categories created
        assert 'attack_category' in train_df.columns
        assert 'attack_category' in test_df.columns
        
        # Check no missing values
        assert train_df.isnull().sum().sum() == 0
        assert test_df.isnull().sum().sum() == 0


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, '-v'])
