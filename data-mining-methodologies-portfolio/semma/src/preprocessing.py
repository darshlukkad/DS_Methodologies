"""
SEMMA Preprocessing Module
Feature engineering, encoding, and scaling for student performance data.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from typing import Tuple, List


class StudentPreprocessor:
    """Preprocessing pipeline for student performance data."""
    
    def __init__(self):
        """Initialize preprocessor with scalers and encoders."""
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.feature_names = []
        
    def create_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Engineer new features from existing ones.
        
        Args:
            df: Input DataFrame
            
        Returns:
            DataFrame with engineered features added
        """
        df = df.copy()
        
        # Parent education features
        if 'Medu' in df.columns and 'Fedu' in df.columns:
            df['parent_edu_avg'] = (df['Medu'] + df['Fedu']) / 2
            df['parent_edu_max'] = df[['Medu', 'Fedu']].max(axis=1)
            df['parent_edu_diff'] = abs(df['Medu'] - df['Fedu'])
        
        # Study behavior features
        if 'studytime' in df.columns and 'failures' in df.columns:
            df['study_failure_interaction'] = df['studytime'] * (1 + df['failures'])
            df['has_failures'] = (df['failures'] > 0).astype(int)
        
        # Grade progression features
        if 'G1' in df.columns and 'G2' in df.columns:
            df['grade_improvement'] = df['G2'] - df['G1']
            df['grade_avg'] = (df['G1'] + df['G2']) / 2
            df['grade_trend'] = np.where(
                df['grade_improvement'] > 0, 1,
                np.where(df['grade_improvement'] < 0, -1, 0)
            )
        
        # Absence risk
        if 'absences' in df.columns:
            df['high_absences'] = (df['absences'] > 10).astype(int)
            df['absence_category'] = pd.cut(
                df['absences'],
                bins=[0, 5, 10, 100],
                labels=['low', 'medium', 'high']
            )
        
        # Social features
        if 'goout' in df.columns and 'studytime' in df.columns:
            df['study_social_balance'] = df['studytime'] / (df['goout'] + 1)
        
        return df
    
    def encode_categorical(
        self,
        train_df: pd.DataFrame,
        val_df: pd.DataFrame,
        test_df: pd.DataFrame,
        categorical_cols: List[str] = None
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Encode categorical variables using LabelEncoder.
        
        Args:
            train_df: Training DataFrame
            val_df: Validation DataFrame
            test_df: Test DataFrame
            categorical_cols: List of columns to encode (auto-detect if None)
            
        Returns:
            Tuple of (train_df, val_df, test_df) with encoded columns
        """
        if categorical_cols is None:
            categorical_cols = train_df.select_dtypes(include=['object']).columns.tolist()
        
        # Remove target if present
        if 'Pass' in categorical_cols:
            categorical_cols.remove('Pass')
        
        for col in categorical_cols:
            if col in train_df.columns:
                le = LabelEncoder()
                
                # Fit on train
                train_df[f'{col}_encoded'] = le.fit_transform(train_df[col])
                
                # Transform val and test
                val_df[f'{col}_encoded'] = le.transform(val_df[col])
                test_df[f'{col}_encoded'] = le.transform(test_df[col])
                
                # Store encoder
                self.label_encoders[col] = le
                
                print(f"✓ Encoded '{col}': {len(le.classes_)} categories")
        
        return train_df, val_df, test_df
    
    def prepare_features(
        self,
        train_df: pd.DataFrame,
        val_df: pd.DataFrame,
        test_df: pd.DataFrame,
        exclude_cols: List[str] = None
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Prepare feature matrices and target vectors.
        
        Args:
            train_df: Training DataFrame
            val_df: Validation DataFrame
            test_df: Test DataFrame
            exclude_cols: Columns to exclude from features
            
        Returns:
            Tuple of (X_train, y_train, X_val, y_val, X_test, y_test)
        """
        if exclude_cols is None:
            exclude_cols = ['Pass', 'G3']
        
        # Auto-exclude categorical columns (use encoded versions)
        cat_cols = train_df.select_dtypes(include=['object']).columns.tolist()
        exclude_cols.extend(cat_cols)
        
        # Remove duplicates
        exclude_cols = list(set(exclude_cols))
        
        # Select feature columns
        feature_cols = [col for col in train_df.columns if col not in exclude_cols]
        self.feature_names = feature_cols
        
        # Extract features and targets
        X_train = train_df[feature_cols].values
        y_train = train_df['Pass'].values
        
        X_val = val_df[feature_cols].values
        y_val = val_df['Pass'].values
        
        X_test = test_df[feature_cols].values
        y_test = test_df['Pass'].values
        
        print(f"\nFeature preparation:")
        print(f"  Selected {len(feature_cols)} features")
        print(f"  X_train: {X_train.shape}")
        print(f"  X_val:   {X_val.shape}")
        print(f"  X_test:  {X_test.shape}")
        
        return X_train, y_train, X_val, y_val, X_test, y_test
    
    def scale_features(
        self,
        X_train: np.ndarray,
        X_val: np.ndarray,
        X_test: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Scale features using StandardScaler.
        
        Args:
            X_train: Training features
            X_val: Validation features
            X_test: Test features
            
        Returns:
            Tuple of (X_train_scaled, X_val_scaled, X_test_scaled)
        """
        # Fit on train, transform all
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_val_scaled = self.scaler.transform(X_val)
        X_test_scaled = self.scaler.transform(X_test)
        
        print(f"\nFeature scaling:")
        print(f"  Mean: {self.scaler.mean_[:3].round(2)}... (first 3)")
        print(f"  Std:  {self.scaler.scale_[:3].round(2)}... (first 3)")
        print("✓ Features scaled")
        
        return X_train_scaled, X_val_scaled, X_test_scaled
    
    def full_pipeline(
        self,
        train_df: pd.DataFrame,
        val_df: pd.DataFrame,
        test_df: pd.DataFrame
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Run full preprocessing pipeline.
        
        Args:
            train_df: Training DataFrame
            val_df: Validation DataFrame
            test_df: Test DataFrame
            
        Returns:
            Tuple of (X_train_scaled, y_train, X_val_scaled, y_val, X_test_scaled, y_test)
        """
        print("=" * 60)
        print("PREPROCESSING PIPELINE")
        print("=" * 60)
        
        # Feature engineering
        print("\n1. Feature Engineering")
        train_df = self.create_features(train_df)
        val_df = self.create_features(val_df)
        test_df = self.create_features(test_df)
        print(f"✓ Created features. New shape: {train_df.shape}")
        
        # Encoding
        print("\n2. Categorical Encoding")
        train_df, val_df, test_df = self.encode_categorical(train_df, val_df, test_df)
        
        # Prepare features
        print("\n3. Feature Preparation")
        X_train, y_train, X_val, y_val, X_test, y_test = self.prepare_features(
            train_df, val_df, test_df
        )
        
        # Scaling
        print("\n4. Feature Scaling")
        X_train_scaled, X_val_scaled, X_test_scaled = self.scale_features(
            X_train, X_val, X_test
        )
        
        print("\n✅ Preprocessing complete!")
        
        return X_train_scaled, y_train, X_val_scaled, y_val, X_test_scaled, y_test


if __name__ == "__main__":
    # Test preprocessing
    from data_loader import load_student_data
    
    print("Testing StudentPreprocessor...")
    train_df, val_df, test_df = load_student_data()
    
    preprocessor = StudentPreprocessor()
    X_train, y_train, X_val, y_val, X_test, y_test = preprocessor.full_pipeline(
        train_df, val_df, test_df
    )
    
    print("\n✅ Preprocessing test complete!")
