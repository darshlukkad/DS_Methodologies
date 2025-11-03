"""
SEMMA Data Loader Module
Handles loading and splitting of Student Performance dataset.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple
from sklearn.model_selection import train_test_split


class StudentDataLoader:
    """Load and prepare student performance data."""
    
    def __init__(self, data_path: str = 'data/student-mat.csv'):
        """
        Initialize data loader.
        
        Args:
            data_path: Path to student performance CSV file
        """
        self.data_path = Path(data_path)
        self.df = None
        
    def load_data(self) -> pd.DataFrame:
        """
        Load student performance data from CSV.
        
        Returns:
            DataFrame with student data
        """
        try:
            self.df = pd.read_csv(self.data_path, sep=';')
            print(f"✓ Loaded {len(self.df):,} records from {self.data_path}")
        except FileNotFoundError:
            print(f"⚠️  File not found: {self.data_path}")
            print("Creating sample dataset...")
            self.df = self._create_sample_data()
            
        return self.df
    
    def _create_sample_data(self, n_samples: int = 395) -> pd.DataFrame:
        """
        Create sample student performance data for demonstration.
        
        Args:
            n_samples: Number of sample records to create
            
        Returns:
            DataFrame with sample student data
        """
        np.random.seed(42)
        
        data = {
            'age': np.random.randint(15, 23, n_samples),
            'sex': np.random.choice(['F', 'M'], n_samples),
            'studytime': np.random.randint(1, 5, n_samples),
            'failures': np.random.choice([0, 0, 0, 1, 2], n_samples),
            'absences': np.random.randint(0, 30, n_samples),
            'G1': np.random.randint(0, 20, n_samples),
            'G2': np.random.randint(0, 20, n_samples),
            'G3': np.random.randint(0, 20, n_samples),
            'Medu': np.random.randint(0, 5, n_samples),
            'Fedu': np.random.randint(0, 5, n_samples),
            'goout': np.random.randint(1, 6, n_samples),
            'health': np.random.randint(1, 6, n_samples),
        }
        
        df = pd.DataFrame(data)
        print(f"✓ Created sample dataset: {len(df):,} records")
        
        return df
    
    def create_target(self, df: pd.DataFrame, threshold: int = 10) -> pd.DataFrame:
        """
        Create binary Pass/Fail target from final grade (G3).
        
        Args:
            df: Input DataFrame
            threshold: Minimum grade for passing (default 10 out of 20)
            
        Returns:
            DataFrame with 'Pass' column added
        """
        df['Pass'] = (df['G3'] >= threshold).astype(int)
        
        pass_rate = df['Pass'].mean() * 100
        print(f"\nTarget created: Pass rate = {pass_rate:.1f}%")
        print(f"  Pass (1): {(df['Pass']==1).sum():,}")
        print(f"  Fail (0): {(df['Pass']==0).sum():,}")
        
        return df
    
    def stratified_split(
        self,
        df: pd.DataFrame,
        train_size: float = 0.6,
        val_size: float = 0.2,
        test_size: float = 0.2,
        random_state: int = 42
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Perform stratified train/val/test split.
        
        Args:
            df: Input DataFrame with 'Pass' column
            train_size: Proportion for training set (default 0.6)
            val_size: Proportion for validation set (default 0.2)
            test_size: Proportion for test set (default 0.2)
            random_state: Random seed for reproducibility
            
        Returns:
            Tuple of (train_df, val_df, test_df)
        """
        assert abs(train_size + val_size + test_size - 1.0) < 1e-6, \
            "Split sizes must sum to 1.0"
        
        # First split: train vs temp (val+test)
        train_df, temp_df = train_test_split(
            df,
            test_size=(val_size + test_size),
            random_state=random_state,
            stratify=df['Pass']
        )
        
        # Second split: val vs test
        val_ratio = val_size / (val_size + test_size)
        val_df, test_df = train_test_split(
            temp_df,
            test_size=(1 - val_ratio),
            random_state=random_state,
            stratify=temp_df['Pass']
        )
        
        # Report splits
        print("\n" + "=" * 60)
        print("STRATIFIED SPLIT")
        print("=" * 60)
        print(f"Train: {len(train_df):,} ({len(train_df)/len(df)*100:.1f}%) - "
              f"Pass rate: {train_df['Pass'].mean()*100:.1f}%")
        print(f"Val:   {len(val_df):,} ({len(val_df)/len(df)*100:.1f}%) - "
              f"Pass rate: {val_df['Pass'].mean()*100:.1f}%")
        print(f"Test:  {len(test_df):,} ({len(test_df)/len(df)*100:.1f}%) - "
              f"Pass rate: {test_df['Pass'].mean()*100:.1f}%")
        print("✓ Stratification preserved")
        
        return train_df, val_df, test_df
    
    def get_feature_info(self, df: pd.DataFrame) -> None:
        """
        Print feature information and statistics.
        
        Args:
            df: Input DataFrame
        """
        print("\n" + "=" * 60)
        print("FEATURE INFORMATION")
        print("=" * 60)
        print(f"Shape: {df.shape}")
        print(f"\nData types:")
        print(df.dtypes.value_counts())
        print(f"\nMissing values:")
        missing = df.isnull().sum()
        if missing.sum() == 0:
            print("✓ No missing values")
        else:
            print(missing[missing > 0])
        print(f"\nNumeric features: {df.select_dtypes(include=[np.number]).shape[1]}")
        print(f"Categorical features: {df.select_dtypes(include=['object']).shape[1]}")


def load_student_data(
    data_path: str = 'data/student-mat.csv',
    threshold: int = 10,
    train_size: float = 0.6,
    val_size: float = 0.2,
    test_size: float = 0.2,
    random_state: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Convenience function to load and split data in one call.
    
    Args:
        data_path: Path to student CSV file
        threshold: Minimum passing grade
        train_size: Training set proportion
        val_size: Validation set proportion
        test_size: Test set proportion
        random_state: Random seed
        
    Returns:
        Tuple of (train_df, val_df, test_df)
    """
    loader = StudentDataLoader(data_path)
    df = loader.load_data()
    df = loader.create_target(df, threshold)
    loader.get_feature_info(df)
    
    train_df, val_df, test_df = loader.stratified_split(
        df, train_size, val_size, test_size, random_state
    )
    
    return train_df, val_df, test_df


if __name__ == "__main__":
    # Test the data loader
    print("Testing StudentDataLoader...")
    train_df, val_df, test_df = load_student_data()
    print("\n✅ Data loader test complete!")
