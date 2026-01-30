"""
Payment Fraud Detection System
================================

A comprehensive machine learning framework for real-time payment fraud detection
with explainable AI capabilities.

Modules:
--------
- data_loader: Data ingestion and preprocessing
- feature_engineering: Feature creation and transformation
- models: Model training and evaluation
- cold_start: Cold start segmentation pipeline
- explainability: SHAP analysis and visualization
- utils: Helper functions and utilities

Example:
--------
>>> from src.models import FraudDetectionPipeline
>>> pipeline = FraudDetectionPipeline()
>>> pipeline.fit(X_train, y_train)
>>> predictions = pipeline.predict(X_test)
"""

__version__ = "1.0.0"
__author__ = "Jorge Fumagalli"
__email__ = "jfumagalli.work@gmail.com"

from src.data_loader import load_data, prepare_data
from src.feature_engineering import FeatureEngineer
from src.models import FraudDetectionPipeline
from src.utils import setup_logging, create_output_dirs

__all__ = [
    "load_data",
    "prepare_data",
    "FeatureEngineer",
    "FraudDetectionPipeline",
    "setup_logging",
    "create_output_dirs",
]
