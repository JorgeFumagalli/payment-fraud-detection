"""
Feature Engineering Module
==========================

This module implements advanced feature creation for fraud detection:
- Historical risk indicators (chargeback rates, transaction counts)
- Temporal patterns (night, weekend, business hours)
- Behavioral features (velocity, diversity)
- Cross-risk features (amount × risk)
"""

import numpy as np
import pandas as pd
from typing import Dict, Tuple, Optional
import logging


logger = logging.getLogger("fraud_detection")


class FeatureEngineer:
    """
    Feature engineering pipeline for fraud detection.
    
    This class creates 25 behavioral, temporal, and risk-based features
    from raw transaction data.
    
    Attributes
    ----------
    global_cbk_rate : float
        Global chargeback rate (used for unseen entities)
    global_avg_amount : float
        Global average transaction amount
    entity_stats : dict
        Precomputed statistics for users, merchants, devices
    high_value_threshold : float
        95th percentile of transaction amounts
        
    Example
    -------
    >>> engineer = FeatureEngineer()
    >>> engineer.fit(train_df)
    >>> train_features = engineer.transform(train_df)
    >>> test_features = engineer.transform(test_df)
    """
    
    def __init__(self):
        self.global_cbk_rate = None
        self.global_avg_amount = None
        self.entity_stats = {}
        self.high_value_threshold = None
        self.fitted = False
        
    def fit(self, df: pd.DataFrame) -> 'FeatureEngineer':
        """
        Learn statistics from training data.
        
        Parameters
        ----------
        df : pd.DataFrame
            Training data with 'has_cbk' target
            
        Returns
        -------
        self
            Fitted feature engineer
        """
        logger.info("Fitting feature engineer on training data...")
        
        # Global statistics
        self.global_cbk_rate = df['has_cbk'].mean()
        self.global_avg_amount = df['transaction_amount'].mean()
        self.high_value_threshold = df['transaction_amount'].quantile(0.95)
        
        logger.info(f"  Global chargeback rate: {self.global_cbk_rate*100:.2f}%")
        logger.info(f"  Global avg amount: ${self.global_avg_amount:.2f}")
        logger.info(f"  High-value threshold (95th): ${self.high_value_threshold:.2f}")
        
        # Entity-level statistics
        self.entity_stats['merchant'] = self._compute_entity_stats(df, 'merchant_id')
        self.entity_stats['device'] = self._compute_entity_stats(df, 'device_id')
        self.entity_stats['user'] = self._compute_entity_stats(df, 'user_id')
        
        logger.info(f"  Unique merchants: {len(self.entity_stats['merchant'])}")
        logger.info(f"  Unique devices: {len(self.entity_stats['device'])}")
        logger.info(f"  Unique users: {len(self.entity_stats['user'])}")
        
        # User diversity statistics (reused for both train and test)
        self.user_device_diversity = df.groupby('user_id')['device_id'].nunique().reset_index(name='user_device_div')
        self.user_merchant_diversity = df.groupby('user_id')['merchant_id'].nunique().reset_index(name='user_merchant_div')
        
        self.fitted = True
        logger.info("✅ Feature engineer fitted successfully")
        
        return self
        
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Transform data by creating all engineered features.
        
        Parameters
        ----------
        df : pd.DataFrame
            Data to transform
            
        Returns
        -------
        pd.DataFrame
            Data with all engineered features
            
        Raises
        ------
        RuntimeError
            If feature engineer has not been fitted
        """
        if not self.fitted:
            raise RuntimeError("Feature engineer must be fitted before transform. Call fit() first.")
        
        logger.info(f"Transforming {len(df)} transactions...")
        
        df = df.copy()
        
        # 1. Merge entity statistics
        df = self._merge_entity_stats(df, 'merchant_id', 'merchant')
        df = self._merge_entity_stats(df, 'device_id', 'device')
        df = self._merge_entity_stats(df, 'user_id', 'user')
        
        # 2. Temporal features
        df = self._create_temporal_features(df)
        
        # 3. Cross-risk features
        df = self._create_cross_risk_features(df)
        
        # 4. Behavioral features
        df = self._create_behavioral_features(df)
        
        # 5. Merge diversity features
        df = df.merge(self.user_device_diversity, on='user_id', how='left')
        df = df.merge(self.user_merchant_diversity, on='user_id', how='left')
        
        # Fill remaining NaN
        df['user_device_div'] = df['user_device_div'].fillna(1)
        df['user_merchant_div'] = df['user_merchant_div'].fillna(1)
        
        logger.info("✅ Feature transformation complete")
        
        return df
    
    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Fit and transform in one step (convenience method).
        
        Parameters
        ----------
        df : pd.DataFrame
            Training data
            
        Returns
        -------
        pd.DataFrame
            Transformed training data
        """
        return self.fit(df).transform(df)
    
    def _compute_entity_stats(self, df: pd.DataFrame, entity_key: str) -> pd.DataFrame:
        """
        Compute chargeback rate, transaction count, and average amount per entity.
        
        Parameters
        ----------
        df : pd.DataFrame
            Transaction data
        entity_key : str
            Column name of entity ('user_id', 'merchant_id', 'device_id')
            
        Returns
        -------
        pd.DataFrame
            Entity statistics
        """
        stats = df.groupby(entity_key).agg(
            cbk_rate=('has_cbk', 'mean'),
            tx_count=('has_cbk', 'size'),
            avg_amount=('transaction_amount', 'mean')
        ).reset_index()
        
        stats.columns = [
            entity_key,
            f'{entity_key}_cbk_rate',
            f'{entity_key}_tx_count',
            f'{entity_key}_avg_amount'
        ]
        
        return stats
    
    def _merge_entity_stats(
        self,
        df: pd.DataFrame,
        entity_key: str,
        entity_name: str
    ) -> pd.DataFrame:
        """
        Merge entity statistics and fill missing values with global priors.
        """
        stats = self.entity_stats[entity_name]
        df = df.merge(stats, on=entity_key, how='left')
        
        # Fill unseen entities with global priors
        df[f'{entity_key}_cbk_rate'] = df[f'{entity_key}_cbk_rate'].fillna(self.global_cbk_rate)
        df[f'{entity_key}_tx_count'] = df[f'{entity_key}_tx_count'].fillna(0)
        df[f'{entity_key}_avg_amount'] = df[f'{entity_key}_avg_amount'].fillna(self.global_avg_amount)
        
        return df
    
    def _create_temporal_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create time-based features.
        """
        df['is_night'] = ((df['hour'] < 7) | (df['hour'] > 22)).astype(int)
        df['is_business_hour'] = ((df['hour'] >= 8) & (df['hour'] <= 18)).astype(int)
        df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
        df['is_early_morning'] = ((df['hour'] >= 0) & (df['hour'] <= 5)).astype(int)
        
        return df
    
    def _create_cross_risk_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create features that combine transaction amount with risk indicators.
        """
        df['is_high_value'] = (df['transaction_amount'] >= self.high_value_threshold).astype(int)
        df['amt_x_cbk_merchant'] = df['transaction_amount'] * df['merchant_id_cbk_rate']
        df['amt_x_cbk_device'] = df['transaction_amount'] * df['device_id_cbk_rate']
        df['amt_x_cbk_user'] = df['transaction_amount'] * df['user_id_cbk_rate']
        
        return df
    
    def _create_behavioral_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create velocity and behavioral pattern features.
        """
        # Sort by user and date for sequential features
        df = df.sort_values(['user_id', 'transaction_date']).reset_index(drop=True)
        
        # Time since last transaction (per user)
        df['user_last_tx_diff'] = (
            df.groupby('user_id')['transaction_date']
            .diff()
            .dt.total_seconds()
            .fillna(999999)  # Large value for first transaction
        )
        
        # Count transactions in last 24 hours (per user)
        df['user_tx_24h'] = self._count_tx_last_24h(df)
        
        return df
    
    def _count_tx_last_24h(self, df: pd.DataFrame) -> np.ndarray:
        """
        Count how many transactions each user made in the last 24 hours.
        
        This is a sliding window calculation that looks backward from each
        transaction to count recent activity.
        
        Parameters
        ----------
        df : pd.DataFrame
            Data sorted by ['user_id', 'transaction_date']
            
        Returns
        -------
        np.ndarray
            Count of transactions in last 24h for each row
        """
        tx_24h = []
        
        for user_id, group in df.groupby('user_id'):
            times = group['transaction_date'].values
            counts = []
            
            for i in range(len(times)):
                ref_time = times[i]
                # Count transactions within 24 hours before ref_time
                count = np.sum((ref_time - times) <= np.timedelta64(24, 'h')) - 1
                counts.append(count)
            
            tx_24h.extend(counts)
        
        return np.array(tx_24h)
    
    def get_feature_names(self) -> list:
        """
        Get list of all feature names created by this engineer.
        
        Returns
        -------
        list
            Feature names
        """
        return [
            'transaction_amount', 'hour', 'day_of_week',
            'merchant_id_cbk_rate', 'merchant_id_tx_count', 'merchant_id_avg_amount',
            'device_id_cbk_rate', 'device_id_tx_count', 'device_id_avg_amount',
            'user_id_cbk_rate', 'user_id_tx_count', 'user_id_avg_amount',
            'is_night', 'is_business_hour', 'is_high_value', 'device_missing',
            'amt_x_cbk_merchant', 'amt_x_cbk_device', 'amt_x_cbk_user',
            'user_last_tx_diff', 'user_tx_24h', 'user_device_div', 'user_merchant_div',
            'is_weekend', 'is_early_morning'
        ]
    
    def get_cold_start_feature_names(self) -> list:
        """
        Get list of features suitable for cold-start scenarios.
        
        These features don't rely on historical chargeback rates.
        
        Returns
        -------
        list
            Cold-start feature names
        """
        return [
            "transaction_amount",
            "hour",
            "day_of_week",
            "is_night",
            "is_weekend",
            "is_business_hour",
            "is_high_value",
            "is_early_morning",
            "device_missing",
            "user_tx_24h",
            "user_last_tx_diff",
            "user_device_div",
            "user_merchant_div",
        ]


if __name__ == "__main__":
    """
    Test the feature engineering module
    """
    from src.utils import setup_logging
    from src.data_loader import prepare_data, split_train_test
    
    logger = setup_logging(log_level="INFO")
    
    print("Testing Feature Engineering Module\n")
    
    # Create synthetic test data
    np.random.seed(42)
    n = 1000
    
    test_data = pd.DataFrame({
        'transaction_id': range(n),
        'transaction_date': pd.date_range('2024-01-01', periods=n, freq='H'),
        'transaction_amount': np.random.lognormal(6, 1, n),
        'user_id': np.random.randint(1, 50, n),
        'merchant_id': np.random.randint(1, 20, n),
        'device_id': np.random.choice([1, 2, 3, 4, 5, None], n),
        'has_cbk': np.random.choice([0, 1], n, p=[0.88, 0.12])
    })
    
    print("1. Preparing data...")
    df_prepared = prepare_data(test_data)
    print(f"   ✅ Prepared {len(df_prepared)} transactions")
    
    print("\n2. Splitting train/test...")
    train, test = split_train_test(df_prepared, test_size=0.2)
    print(f"   ✅ Train: {len(train)}, Test: {len(test)}")
    
    print("\n3. Creating feature engineer...")
    engineer = FeatureEngineer()
    
    print("\n4. Fitting on training data...")
    engineer.fit(train)
    
    print("\n5. Transforming training data...")
    train_fe = engineer.transform(train)
    print(f"   ✅ Created {len(engineer.get_feature_names())} features")
    print(f"   ✅ Shape: {train_fe.shape}")
    
    print("\n6. Transforming test data...")
    test_fe = engineer.transform(test)
    print(f"   ✅ Shape: {test_fe.shape}")
    
    print("\n7. Checking feature names...")
    features = engineer.get_feature_names()
    cold_features = engineer.get_cold_start_feature_names()
    print(f"   ✅ Full feature set: {len(features)} features")
    print(f"   ✅ Cold-start feature set: {len(cold_features)} features")
    
    print("\n8. Sample features:")
    print(train_fe[features].head())
    
    print("\n✅ All feature engineering tests passed!")
