"""
Unit Tests for Feature Engineering Module
==========================================

Tests for FeatureEngineer class and related functions.
"""

import pytest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

from src.feature_engineering import FeatureEngineer


@pytest.fixture
def sample_data():
    """Create sample transaction data for testing."""
    np.random.seed(42)
    n = 100
    
    base_date = datetime(2024, 1, 1)
    dates = [base_date + timedelta(hours=i) for i in range(n)]
    
    data = pd.DataFrame({
        'transaction_id': range(n),
        'transaction_date': dates,
        'transaction_amount': np.random.lognormal(6, 1, n),
        'user_id': np.random.randint(1, 10, n),
        'merchant_id': np.random.randint(1, 5, n),
        'device_id': np.random.choice([1, 2, 3, None], n),
        'has_cbk': np.random.choice([0, 1], n, p=[0.88, 0.12]),
        'hour': [d.hour for d in dates],
        'day_of_week': [d.weekday() for d in dates],
        'device_missing': [1 if d is None else 0 for d in np.random.choice([1, 2, 3, None], n)]
    })
    
    return data


class TestFeatureEngineer:
    """Test suite for FeatureEngineer class."""
    
    def test_initialization(self):
        """Test FeatureEngineer initialization."""
        engineer = FeatureEngineer()
        
        assert engineer.global_cbk_rate is None
        assert engineer.global_avg_amount is None
        assert engineer.fitted is False
        assert len(engineer.entity_stats) == 0
    
    def test_fit_creates_entity_stats(self, sample_data):
        """Test that fit creates statistics for all entities."""
        engineer = FeatureEngineer()
        engineer.fit(sample_data)
        
        assert engineer.fitted is True
        assert 'merchant' in engineer.entity_stats
        assert 'device' in engineer.entity_stats
        assert 'user' in engineer.entity_stats
        assert engineer.global_cbk_rate is not None
        assert engineer.global_avg_amount is not None
    
    def test_transform_without_fit_raises_error(self, sample_data):
        """Test that transform raises error if not fitted."""
        engineer = FeatureEngineer()
        
        with pytest.raises(RuntimeError, match="must be fitted"):
            engineer.transform(sample_data)
    
    def test_transform_creates_expected_features(self, sample_data):
        """Test that transform creates all expected features."""
        engineer = FeatureEngineer()
        engineer.fit(sample_data)
        
        transformed = engineer.transform(sample_data)
        expected_features = engineer.get_feature_names()
        
        # Check that all features are present
        for feature in expected_features:
            assert feature in transformed.columns, f"Missing feature: {feature}"
    
    def test_feature_count(self):
        """Test that FeatureEngineer creates exactly 25 features."""
        engineer = FeatureEngineer()
        features = engineer.get_feature_names()
        
        assert len(features) == 25, f"Expected 25 features, got {len(features)}"
    
    def test_cold_start_features_subset(self):
        """Test that cold start features are a subset of all features."""
        engineer = FeatureEngineer()
        all_features = set(engineer.get_feature_names())
        cold_features = set(engineer.get_cold_start_feature_names())
        
        assert cold_features.issubset(all_features)
        assert len(cold_features) < len(all_features)
    
    def test_temporal_features_binary(self, sample_data):
        """Test that temporal features are binary (0 or 1)."""
        engineer = FeatureEngineer()
        engineer.fit(sample_data)
        transformed = engineer.transform(sample_data)
        
        temporal_features = ['is_night', 'is_business_hour', 'is_weekend', 'is_early_morning']
        
        for feature in temporal_features:
            assert transformed[feature].isin([0, 1]).all(), f"{feature} should be binary"
    
    def test_chargeback_rates_in_range(self, sample_data):
        """Test that chargeback rates are between 0 and 1."""
        engineer = FeatureEngineer()
        engineer.fit(sample_data)
        transformed = engineer.transform(sample_data)
        
        rate_features = [
            'merchant_id_cbk_rate',
            'device_id_cbk_rate',
            'user_id_cbk_rate'
        ]
        
        for feature in rate_features:
            assert (transformed[feature] >= 0).all(), f"{feature} should be >= 0"
            assert (transformed[feature] <= 1).all(), f"{feature} should be <= 1"
    
    def test_transaction_counts_non_negative(self, sample_data):
        """Test that transaction counts are non-negative."""
        engineer = FeatureEngineer()
        engineer.fit(sample_data)
        transformed = engineer.transform(sample_data)
        
        count_features = [
            'merchant_id_tx_count',
            'device_id_tx_count',
            'user_id_tx_count',
            'user_tx_24h'
        ]
        
        for feature in count_features:
            assert (transformed[feature] >= 0).all(), f"{feature} should be non-negative"
    
    def test_fit_transform_equivalence(self, sample_data):
        """Test that fit_transform produces same result as fit then transform."""
        engineer1 = FeatureEngineer()
        result1 = engineer1.fit_transform(sample_data)
        
        engineer2 = FeatureEngineer()
        engineer2.fit(sample_data)
        result2 = engineer2.transform(sample_data)
        
        # Compare key features (some randomness in ordering is acceptable)
        pd.testing.assert_frame_equal(
            result1[['transaction_amount', 'is_night', 'user_id_cbk_rate']],
            result2[['transaction_amount', 'is_night', 'user_id_cbk_rate']]
        )
    
    def test_handles_unseen_entities(self, sample_data):
        """Test that transform handles entities not seen during training."""
        # Split data
        train = sample_data[:80]
        test = sample_data[80:]
        
        # Add some completely new entities to test set
        test.loc[test.index[0], 'user_id'] = 9999
        test.loc[test.index[1], 'merchant_id'] = 9999
        
        engineer = FeatureEngineer()
        engineer.fit(train)
        transformed = engineer.transform(test)
        
        # Should fill with global rates, not crash
        assert not transformed['user_id_cbk_rate'].isna().any()
        assert not transformed['merchant_id_cbk_rate'].isna().any()
    
    def test_cross_risk_features_calculation(self, sample_data):
        """Test that cross-risk features are calculated correctly."""
        engineer = FeatureEngineer()
        engineer.fit(sample_data)
        transformed = engineer.transform(sample_data)
        
        # Manually verify calculation for first row
        idx = 0
        amount = transformed.loc[idx, 'transaction_amount']
        user_rate = transformed.loc[idx, 'user_id_cbk_rate']
        expected_cross_risk = amount * user_rate
        actual_cross_risk = transformed.loc[idx, 'amt_x_cbk_user']
        
        assert np.isclose(expected_cross_risk, actual_cross_risk, rtol=1e-5)


class TestEdgeCases:
    """Test edge cases and boundary conditions."""
    
    def test_empty_dataframe(self):
        """Test handling of empty dataframe."""
        engineer = FeatureEngineer()
        empty_df = pd.DataFrame(columns=['user_id', 'merchant_id', 'device_id', 'has_cbk'])
        
        # Should not crash, but may have undefined behavior
        # This is expected to fail gracefully
        with pytest.raises(Exception):
            engineer.fit(empty_df)
    
    def test_single_transaction(self):
        """Test handling of single transaction."""
        single_tx = pd.DataFrame({
            'transaction_date': [datetime(2024, 1, 1)],
            'transaction_amount': [100.0],
            'user_id': [1],
            'merchant_id': [1],
            'device_id': [1],
            'has_cbk': [0],
            'hour': [12],
            'day_of_week': [0],
            'device_missing': [0]
        })
        
        engineer = FeatureEngineer()
        engineer.fit(single_tx)
        transformed = engineer.transform(single_tx)
        
        assert len(transformed) == 1
        assert 'user_id_cbk_rate' in transformed.columns
    
    def test_all_frauds(self):
        """Test handling when all transactions are frauds."""
        fraud_only = pd.DataFrame({
            'transaction_date': pd.date_range('2024-01-01', periods=10, freq='H'),
            'transaction_amount': [100.0] * 10,
            'user_id': [1] * 10,
            'merchant_id': [1] * 10,
            'device_id': [1] * 10,
            'has_cbk': [1] * 10,  # All frauds
            'hour': list(range(10)),
            'day_of_week': [0] * 10,
            'device_missing': [0] * 10
        })
        
        engineer = FeatureEngineer()
        engineer.fit(fraud_only)
        
        # Chargeback rate should be 1.0
        assert engineer.global_cbk_rate == 1.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
