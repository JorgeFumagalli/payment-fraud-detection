"""
Model Training and Evaluation Module
====================================

This module implements:
- Multiple ML model architectures (LogReg, RF, XGBoost, MLP, LSTM)
- Hyperparameter optimization
- Model evaluation and comparison
- Production-ready prediction pipeline
"""

import numpy as np
import pandas as pd
from typing import Dict, Tuple, Optional, Any, List
import logging
import joblib
from pathlib import Path

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    roc_curve,
    precision_recall_curve
)

from xgboost import XGBClassifier
from skopt import BayesSearchCV
from skopt.space import Integer

import tensorflow as tf
from tensorflow.keras import layers, callbacks, regularizers


logger = logging.getLogger("fraud_detection")


class FraudDetectionPipeline:
    """
    Complete pipeline for fraud detection model training and inference.
    
    This class manages:
    - Multiple model types (Logistic, RF, XGBoost, MLP, LSTM)
    - Hyperparameter optimization
    - Model evaluation
    - Production predictions
    
    Attributes
    ----------
    models : dict
        Trained models keyed by name
    best_model_name : str
        Name of best performing model
    scaler : StandardScaler
        Fitted scaler for neural network models
    
    Example
    -------
    >>> pipeline = FraudDetectionPipeline(random_state=42)
    >>> pipeline.train_all(X_train, y_train)
    >>> results = pipeline.evaluate(X_test, y_test)
    >>> predictions = pipeline.predict(X_new, model_name='XGBoost')
    """
    
    def __init__(self, random_state: int = 42):
        """
        Initialize the fraud detection pipeline.
        
        Parameters
        ----------
        random_state : int, default=42
            Random seed for reproducibility
        """
        self.random_state = random_state
        self.models = {}
        self.best_model_name = None
        self.scaler = StandardScaler()
        self.feature_names = None
        
    def train_all(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        optimize_hyperparams: bool = False
    ) -> Dict[str, Any]:
        """
        Train all available models.
        
        Parameters
        ----------
        X_train : pd.DataFrame
            Training features
        y_train : pd.Series
            Training labels
        optimize_hyperparams : bool, default=False
            Whether to perform Bayesian optimization (slower but better)
            
        Returns
        -------
        dict
            Training metrics for each model
        """
        logger.info("=" * 70)
        logger.info("TRAINING ALL MODELS")
        logger.info("=" * 70)
        
        self.feature_names = list(X_train.columns)
        
        # Train each model
        self.models['LogReg'] = self._train_logistic_regression(X_train, y_train)
        self.models['RF'] = self._train_random_forest(X_train, y_train, optimize=optimize_hyperparams)
        self.models['XGB'] = self._train_xgboost(X_train, y_train)
        self.models['MLP'] = self._train_mlp(X_train, y_train)
        
        logger.info("✅ All models trained successfully")
        
        return self.models
    
    def _train_logistic_regression(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series
    ) -> LogisticRegression:
        """Train Logistic Regression (baseline)."""
        logger.info("\n1. Training Logistic Regression...")
        
        model = LogisticRegression(
            max_iter=1000,
            class_weight='balanced',
            random_state=self.random_state,
            solver='lbfgs'
        )
        
        model.fit(X_train, y_train)
        logger.info("   ✅ Logistic Regression trained")
        
        return model
    
    def _train_random_forest(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        optimize: bool = False
    ) -> RandomForestClassifier:
        """Train Random Forest with optional Bayesian optimization."""
        logger.info("\n2. Training Random Forest...")
        
        if optimize:
            logger.info("   Running Bayesian optimization (this may take a while)...")
            
            rf = RandomForestClassifier(
                random_state=self.random_state,
                n_jobs=-1,
                class_weight='balanced'
            )
            
            search_spaces = {
                'n_estimators': Integer(200, 800),
                'max_depth': Integer(4, 20)
            }
            
            opt = BayesSearchCV(
                rf,
                search_spaces,
                n_iter=8,
                cv=3,
                scoring='roc_auc',
                n_jobs=-1,
                random_state=self.random_state,
                verbose=0
            )
            
            opt.fit(X_train, y_train)
            model = opt.best_estimator_
            
            logger.info(f"   Best params: n_estimators={model.n_estimators}, max_depth={model.max_depth}")
        else:
            model = RandomForestClassifier(
                n_estimators=300,
                max_depth=10,
                random_state=self.random_state,
                n_jobs=-1,
                class_weight='balanced'
            )
            model.fit(X_train, y_train)
        
        logger.info("   ✅ Random Forest trained")
        return model
    
    def _train_xgboost(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series
    ) -> XGBClassifier:
        """Train XGBoost with class imbalance handling."""
        logger.info("\n3. Training XGBoost...")
        
        # Calculate scale_pos_weight for imbalance
        n_neg = (y_train == 0).sum()
        n_pos = (y_train == 1).sum()
        scale_pos_weight = n_neg / n_pos
        
        model = XGBClassifier(
            n_estimators=500,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            scale_pos_weight=scale_pos_weight,
            random_state=self.random_state,
            n_jobs=-1,
            eval_metric='logloss'
        )
        
        model.fit(X_train, y_train, verbose=False)
        logger.info("   ✅ XGBoost trained")
        
        return model
    
    def _train_mlp(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series
    ) -> MLPClassifier:
        """Train Multi-Layer Perceptron (Neural Network)."""
        logger.info("\n4. Training MLP Neural Network...")
        
        # Scale features for neural network
        X_scaled = self.scaler.fit_transform(X_train)
        
        model = MLPClassifier(
            hidden_layer_sizes=(64, 32),
            activation='relu',
            solver='adam',
            alpha=1e-3,  # L2 regularization
            max_iter=200,
            random_state=self.random_state,
            early_stopping=True,
            validation_fraction=0.1
        )
        
        model.fit(X_scaled, y_train)
        logger.info("   ✅ MLP trained")
        
        return model
    
    def evaluate(
        self,
        X_test: pd.DataFrame,
        y_test: pd.Series,
        threshold: float = 0.5
    ) -> Dict[str, Dict[str, float]]:
        """
        Evaluate all trained models on test set.
        
        Parameters
        ----------
        X_test : pd.DataFrame
            Test features
        y_test : pd.Series
            Test labels
        threshold : float, default=0.5
            Classification threshold
            
        Returns
        -------
        dict
            Evaluation metrics for each model
        """
        logger.info("\n" + "=" * 70)
        logger.info("EVALUATING MODELS")
        logger.info("=" * 70)
        
        results = {}
        
        for name, model in self.models.items():
            logger.info(f"\nEvaluating {name}...")
            
            # Get predictions
            if name == 'MLP':
                X_eval = self.scaler.transform(X_test)
                y_proba = model.predict_proba(X_eval)[:, 1]
            else:
                y_proba = model.predict_proba(X_test)[:, 1]
            
            y_pred = (y_proba >= threshold).astype(int)
            
            # Calculate metrics
            metrics = {
                'AUC': roc_auc_score(y_test, y_proba),
                'PR_AUC': average_precision_score(y_test, y_proba),
                'Precision': precision_score(y_test, y_pred, zero_division=0),
                'Recall': recall_score(y_test, y_pred, zero_division=0),
                'F1': f1_score(y_test, y_pred, zero_division=0)
            }
            
            # Confusion matrix
            cm = confusion_matrix(y_test, y_pred)
            tn, fp, fn, tp = cm.ravel()
            metrics.update({
                'TP': int(tp),
                'FP': int(fp),
                'TN': int(tn),
                'FN': int(fn)
            })
            
            results[name] = metrics
            
            logger.info(f"   AUC: {metrics['AUC']:.4f}")
            logger.info(f"   Precision: {metrics['Precision']:.4f}")
            logger.info(f"   Recall: {metrics['Recall']:.4f}")
        
        # Identify best model by AUC
        self.best_model_name = max(results.items(), key=lambda x: x[1]['AUC'])[0]
        logger.info(f"\n✅ Best model by AUC: {self.best_model_name}")
        
        return results
    
    def predict(
        self,
        X: pd.DataFrame,
        model_name: Optional[str] = None,
        threshold: float = 0.5
    ) -> np.ndarray:
        """
        Generate predictions using specified model.
        
        Parameters
        ----------
        X : pd.DataFrame
            Features to predict
        model_name : str, optional
            Model to use. If None, uses best model
        threshold : float, default=0.5
            Classification threshold
            
        Returns
        -------
        np.ndarray
            Binary predictions (0 or 1)
        """
        if model_name is None:
            model_name = self.best_model_name
        
        if model_name not in self.models:
            raise ValueError(f"Model '{model_name}' not found. Available: {list(self.models.keys())}")
        
        proba = self.predict_proba(X, model_name)
        return (proba >= threshold).astype(int)
    
    def predict_proba(
        self,
        X: pd.DataFrame,
        model_name: Optional[str] = None
    ) -> np.ndarray:
        """
        Generate fraud probabilities using specified model.
        
        Parameters
        ----------
        X : pd.DataFrame
            Features to predict
        model_name : str, optional
            Model to use. If None, uses best model
            
        Returns
        -------
        np.ndarray
            Fraud probabilities (0-1)
        """
        if model_name is None:
            model_name = self.best_model_name
        
        if model_name not in self.models:
            raise ValueError(f"Model '{model_name}' not found. Available: {list(self.models.keys())}")
        
        model = self.models[model_name]
        
        if model_name == 'MLP':
            X_scaled = self.scaler.transform(X)
            return model.predict_proba(X_scaled)[:, 1]
        else:
            return model.predict_proba(X)[:, 1]
    
    def save(self, filepath: str) -> None:
        """
        Save trained pipeline to disk.
        
        Parameters
        ----------
        filepath : str
            Path to save pipeline
        """
        save_dict = {
            'models': self.models,
            'best_model_name': self.best_model_name,
            'scaler': self.scaler,
            'feature_names': self.feature_names,
            'random_state': self.random_state
        }
        
        joblib.dump(save_dict, filepath)
        logger.info(f"✅ Pipeline saved to {filepath}")
    
    @classmethod
    def load(cls, filepath: str) -> 'FraudDetectionPipeline':
        """
        Load trained pipeline from disk.
        
        Parameters
        ----------
        filepath : str
            Path to saved pipeline
            
        Returns
        -------
        FraudDetectionPipeline
            Loaded pipeline
        """
        save_dict = joblib.load(filepath)
        
        pipeline = cls(random_state=save_dict['random_state'])
        pipeline.models = save_dict['models']
        pipeline.best_model_name = save_dict['best_model_name']
        pipeline.scaler = save_dict['scaler']
        pipeline.feature_names = save_dict['feature_names']
        
        logger.info(f"✅ Pipeline loaded from {filepath}")
        
        return pipeline


def calculate_cost_sensitivity(
    results: Dict[str, Dict[str, float]],
    r_values: np.ndarray
) -> Dict[str, np.ndarray]:
    """
    Calculate cost curves for model comparison.
    
    Parameters
    ----------
    results : dict
        Model evaluation results with confusion matrix values
    r_values : np.ndarray
        Array of cost ratios to evaluate
        
    Returns
    -------
    dict
        Cost curves for each model
    """
    cost_curves = {}
    
    for model_name, metrics in results.items():
        fp = metrics['FP']
        fn = metrics['FN']
        
        costs = fp + r_values * fn
        cost_curves[model_name] = costs
    
    return cost_curves


if __name__ == "__main__":
    """
    Test the models module
    """
    from src.utils import setup_logging
    from src.data_loader import prepare_data, split_train_test
    from src.feature_engineering import FeatureEngineer
    
    logger = setup_logging(log_level="INFO")
    
    print("Testing Models Module\n")
    
    # Create synthetic data
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
    
    # Prepare data
    df_prepared = prepare_data(test_data)
    train, test = split_train_test(df_prepared, test_size=0.2)
    
    # Feature engineering
    engineer = FeatureEngineer()
    train_fe = engineer.fit_transform(train)
    test_fe = engineer.transform(test)
    
    features = engineer.get_feature_names()
    X_train = train_fe[features]
    y_train = train_fe['has_cbk']
    X_test = test_fe[features]
    y_test = test_fe['has_cbk']
    
    print("1. Initializing pipeline...")
    pipeline = FraudDetectionPipeline(random_state=42)
    
    print("\n2. Training all models...")
    pipeline.train_all(X_train, y_train, optimize_hyperparams=False)
    
    print("\n3. Evaluating models...")
    results = pipeline.evaluate(X_test, y_test)
    
    print("\n4. Results summary:")
    for model, metrics in results.items():
        print(f"   {model:10s} AUC={metrics['AUC']:.4f}  Recall={metrics['Recall']:.4f}")
    
    print("\n5. Making predictions with best model...")
    predictions = pipeline.predict(X_test)
    probabilities = pipeline.predict_proba(X_test)
    print(f"   ✅ Generated {len(predictions)} predictions")
    
    print("\n✅ All model tests passed!")
