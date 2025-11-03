"""
KDD Data Loader Module
Handles loading and preprocessing of NSL-KDD intrusion detection dataset.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple, List


class NSLKDDLoader:
    """Load and prepare NSL-KDD network intrusion data."""
    
    # NSL-KDD column names (41 features + label + difficulty)
    COLUMN_NAMES = [
        'duration', 'protocol_type', 'service', 'flag',
        'src_bytes', 'dst_bytes', 'land', 'wrong_fragment', 'urgent',
        'hot', 'num_failed_logins', 'logged_in', 'num_compromised',
        'root_shell', 'su_attempted', 'num_root', 'num_file_creations',
        'num_shells', 'num_access_files', 'num_outbound_cmds',
        'is_host_login', 'is_guest_login', 'count', 'srv_count',
        'serror_rate', 'srv_serror_rate', 'rerror_rate', 'srv_rerror_rate',
        'same_srv_rate', 'diff_srv_rate', 'srv_diff_host_rate',
        'dst_host_count', 'dst_host_srv_count', 'dst_host_same_srv_rate',
        'dst_host_diff_srv_rate', 'dst_host_same_src_port_rate',
        'dst_host_srv_diff_host_rate', 'dst_host_serror_rate',
        'dst_host_srv_serror_rate', 'dst_host_rerror_rate',
        'dst_host_srv_rerror_rate',
        'attack_type', 'difficulty'
    ]
    
    # Attack type mapping to 5 categories
    ATTACK_MAPPING = {
        'normal': 'normal',
        # DoS attacks
        'back': 'dos', 'land': 'dos', 'neptune': 'dos', 'pod': 'dos',
        'smurf': 'dos', 'teardrop': 'dos', 'apache2': 'dos', 'udpstorm': 'dos',
        'processtable': 'dos', 'mailbomb': 'dos',
        # Probe attacks
        'ipsweep': 'probe', 'nmap': 'probe', 'portsweep': 'probe', 'satan': 'probe',
        'mscan': 'probe', 'saint': 'probe',
        # R2L (Remote to Local) attacks
        'ftp_write': 'r2l', 'guess_passwd': 'r2l', 'imap': 'r2l',
        'multihop': 'r2l', 'phf': 'r2l', 'spy': 'r2l', 'warezclient': 'r2l',
        'warezmaster': 'r2l', 'sendmail': 'r2l', 'named': 'r2l',
        'snmpgetattack': 'r2l', 'snmpguess': 'r2l', 'xlock': 'r2l',
        'xsnoop': 'r2l', 'worm': 'r2l',
        # U2R (User to Root) attacks
        'buffer_overflow': 'u2r', 'loadmodule': 'u2r', 'perl': 'u2r',
        'rootkit': 'u2r', 'httptunnel': 'u2r', 'ps': 'u2r', 'sqlattack': 'u2r',
        'xterm': 'u2r'
    }
    
    def __init__(self, data_dir: str = 'data'):
        """
        Initialize NSL-KDD loader.
        
        Args:
            data_dir: Directory containing NSL-KDD files
        """
        self.data_dir = Path(data_dir)
        self.train_df = None
        self.test_df = None
        
    def load_data(
        self,
        train_file: str = 'KDDTrain+.txt',
        test_file: str = 'KDDTest+.txt'
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Load NSL-KDD train and test files.
        
        Args:
            train_file: Training data filename
            test_file: Test data filename
            
        Returns:
            Tuple of (train_df, test_df)
        """
        train_path = self.data_dir / train_file
        test_path = self.data_dir / test_file
        
        try:
            self.train_df = pd.read_csv(train_path, names=self.COLUMN_NAMES)
            self.test_df = pd.read_csv(test_path, names=self.COLUMN_NAMES)
            
            print(f"✓ Loaded NSL-KDD dataset")
            print(f"  Train: {len(self.train_df):,} records")
            print(f"  Test:  {len(self.test_df):,} records")
            
        except FileNotFoundError:
            print(f"⚠️  NSL-KDD files not found in {self.data_dir}")
            print("Creating sample dataset for demonstration...")
            self.train_df, self.test_df = self._create_sample_data()
        
        return self.train_df, self.test_df
    
    def _create_sample_data(
        self,
        n_train: int = 10000,
        n_test: int = 2000
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Create sample NSL-KDD data for demonstration.
        
        Args:
            n_train: Number of training samples
            n_test: Number of test samples
            
        Returns:
            Tuple of (train_df, test_df)
        """
        np.random.seed(42)
        
        # Attack distribution (mimics NSL-KDD)
        attack_probs = {
            'normal': 0.50,
            'dos': 0.30,
            'probe': 0.15,
            'r2l': 0.04,
            'u2r': 0.01
        }
        
        def create_df(n_samples):
            # Generate realistic-looking network features
            df = pd.DataFrame({
                'duration': np.random.exponential(100, n_samples).astype(int),
                'protocol_type': np.random.choice(['tcp', 'udp', 'icmp'], n_samples),
                'service': np.random.choice(['http', 'ftp', 'smtp', 'telnet'], n_samples),
                'flag': np.random.choice(['SF', 'S0', 'REJ', 'RSTO'], n_samples),
                'src_bytes': np.random.exponential(1000, n_samples).astype(int),
                'dst_bytes': np.random.exponential(1000, n_samples).astype(int),
                'land': np.random.choice([0, 1], n_samples, p=[0.99, 0.01]),
                'wrong_fragment': np.random.poisson(0.1, n_samples),
                'urgent': np.random.poisson(0.05, n_samples),
                'hot': np.random.poisson(0.5, n_samples),
                'num_failed_logins': np.random.poisson(0.1, n_samples),
                'logged_in': np.random.choice([0, 1], n_samples, p=[0.3, 0.7]),
                'num_compromised': np.random.poisson(0.05, n_samples),
                'count': np.random.randint(1, 500, n_samples),
                'srv_count': np.random.randint(1, 500, n_samples),
                'serror_rate': np.random.random(n_samples),
                'srv_serror_rate': np.random.random(n_samples),
                'rerror_rate': np.random.random(n_samples),
                'srv_rerror_rate': np.random.random(n_samples),
                'same_srv_rate': np.random.random(n_samples),
                'diff_srv_rate': np.random.random(n_samples),
                'attack_type': np.random.choice(
                    list(attack_probs.keys()),
                    n_samples,
                    p=list(attack_probs.values())
                )
            })
            
            # Add difficulty score
            df['difficulty'] = np.random.randint(0, 22, n_samples)
            
            return df
        
        train_df = create_df(n_train)
        test_df = create_df(n_test)
        
        print(f"✓ Created sample dataset")
        print(f"  Train: {len(train_df):,} records")
        print(f"  Test:  {len(test_df):,} records")
        
        return train_df, test_df
    
    def map_attack_types(
        self,
        df: pd.DataFrame,
        detailed: bool = False
    ) -> pd.DataFrame:
        """
        Map specific attacks to 5 main categories.
        
        Args:
            df: Input DataFrame with 'attack_type' column
            detailed: If True, keep original attack names in separate column
            
        Returns:
            DataFrame with mapped attack categories
        """
        df = df.copy()
        
        if detailed:
            df['attack_detailed'] = df['attack_type'].copy()
        
        # Clean attack names (remove trailing dots if present)
        df['attack_type'] = df['attack_type'].str.rstrip('.')
        
        # Map to 5 categories
        df['attack_category'] = df['attack_type'].map(self.ATTACK_MAPPING)
        
        # Handle unknown attacks
        unknown_count = df['attack_category'].isnull().sum()
        if unknown_count > 0:
            print(f"⚠️  Found {unknown_count} unknown attack types")
            df['attack_category'] = df['attack_category'].fillna('unknown')
        
        return df
    
    def get_attack_distribution(
        self,
        train_df: pd.DataFrame,
        test_df: pd.DataFrame
    ) -> None:
        """
        Print attack type distribution for train and test sets.
        
        Args:
            train_df: Training DataFrame
            test_df: Test DataFrame
        """
        print("\n" + "=" * 60)
        print("ATTACK TYPE DISTRIBUTION")
        print("=" * 60)
        
        print("\nTrain Set:")
        train_dist = train_df['attack_category'].value_counts()
        for attack, count in train_dist.items():
            pct = count / len(train_df) * 100
            print(f"  {attack:10s}: {count:6,} ({pct:5.2f}%)")
        
        print("\nTest Set:")
        test_dist = test_df['attack_category'].value_counts()
        for attack, count in test_dist.items():
            pct = count / len(test_df) * 100
            print(f"  {attack:10s}: {count:6,} ({pct:5.2f}%)")
        
        # Class imbalance warning
        min_class_pct = train_dist.min() / len(train_df) * 100
        if min_class_pct < 1.0:
            print(f"\n⚠️  Severe class imbalance detected!")
            print(f"   Minority class: {min_class_pct:.2f}%")
            print(f"   Recommendation: Use SMOTE or class weights")
    
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
        print(f"\nFeature types:")
        print(df.dtypes.value_counts())
        print(f"\nMissing values:")
        missing = df.isnull().sum()
        if missing.sum() == 0:
            print("✓ No missing values")
        else:
            print(missing[missing > 0])
        
        # Numeric features
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if 'attack_category' in numeric_cols:
            numeric_cols.remove('attack_category')
        if 'difficulty' in numeric_cols:
            numeric_cols.remove('difficulty')
        
        print(f"\nNumeric features: {len(numeric_cols)}")
        
        # Categorical features
        cat_cols = df.select_dtypes(include=['object']).columns.tolist()
        print(f"Categorical features: {len(cat_cols)}")
        for col in cat_cols:
            if col != 'attack_category' and col != 'attack_detailed':
                n_unique = df[col].nunique()
                print(f"  {col}: {n_unique} unique values")


def load_nslkdd_data(
    data_dir: str = 'data',
    train_file: str = 'KDDTrain+.txt',
    test_file: str = 'KDDTest+.txt',
    map_attacks: bool = True
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Convenience function to load NSL-KDD data in one call.
    
    Args:
        data_dir: Directory containing data files
        train_file: Training data filename
        test_file: Test data filename
        map_attacks: Whether to map attacks to 5 categories
        
    Returns:
        Tuple of (train_df, test_df)
    """
    loader = NSLKDDLoader(data_dir)
    train_df, test_df = loader.load_data(train_file, test_file)
    
    if map_attacks:
        train_df = loader.map_attack_types(train_df, detailed=True)
        test_df = loader.map_attack_types(test_df, detailed=True)
    
    loader.get_attack_distribution(train_df, test_df)
    loader.get_feature_info(train_df)
    
    return train_df, test_df


if __name__ == "__main__":
    # Test the data loader
    print("Testing NSLKDDLoader...")
    train_df, test_df = load_nslkdd_data()
    print("\n✅ Data loader test complete!")
