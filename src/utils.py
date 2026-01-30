"""
Utility Functions for Payment Fraud Detection System
======================================================

This module contains helper functions for:
- Directory management
- Logging configuration
- Configuration loading
- Performance metrics
"""

import os
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional
import json
import numpy as np
import pandas as pd


def setup_logging(
    log_level: str = "INFO",
    log_file: Optional[str] = None,
    format_string: Optional[str] = None
) -> logging.Logger:
    """
    Configure logging for the application.
    
    Parameters
    ----------
    log_level : str, default="INFO"
        Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
    log_file : str, optional
        Path to log file. If None, logs only to console
    format_string : str, optional
        Custom format string for log messages
        
    Returns
    -------
    logging.Logger
        Configured logger instance
        
    Example
    -------
    >>> logger = setup_logging(log_level="DEBUG", log_file="fraud_detection.log")
    >>> logger.info("Starting fraud detection pipeline")
    """
    if format_string is None:
        format_string = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    
    logger = logging.getLogger("fraud_detection")
    logger.setLevel(getattr(logging, log_level.upper()))
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_formatter = logging.Formatter(format_string)
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)
    
    # File handler (if specified)
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.DEBUG)
        file_formatter = logging.Formatter(format_string)
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)
    
    return logger


def create_output_dirs(base_path: str = "outputs") -> Dict[str, Path]:
    """
    Create all necessary output directories.
    
    Parameters
    ----------
    base_path : str, default="outputs"
        Base directory for all outputs
        
    Returns
    -------
    dict
        Dictionary mapping directory names to Path objects
        
    Example
    -------
    >>> dirs = create_output_dirs()
    >>> print(dirs['eda'])
    PosixPath('outputs/eda')
    """
    base = Path(base_path)
    
    directories = {
        "base": base,
        "eda": base / "eda",
        "confusion_matrices": base / "confusion_matrices",
        "curves": base / "curves",
        "shap": base / "shap",
        "tables": base / "tables",
        "models": base / "models",
        "data_raw": Path("data/raw"),
        "data_processed": Path("data/processed"),
    }
    
    for dir_path in directories.values():
        dir_path.mkdir(parents=True, exist_ok=True)
    
    return directories


def load_config(config_path: str = "config.json") -> Dict[str, Any]:
    """
    Load configuration from JSON file.
    
    Parameters
    ----------
    config_path : str, default="config.json"
        Path to configuration file
        
    Returns
    -------
    dict
        Configuration dictionary
        
    Example
    -------
    >>> config = load_config("config.json")
    >>> print(config['random_state'])
    42
    """
    if not os.path.exists(config_path):
        # Return default configuration
        return {
            "random_state": 42,
            "test_size": 0.2,
            "threshold": 0.5,
            "cold_start_threshold": 0.35,
        }
    
    with open(config_path, 'r') as f:
        return json.load(f)


def save_config(config: Dict[str, Any], config_path: str = "config.json") -> None:
    """
    Save configuration to JSON file.
    
    Parameters
    ----------
    config : dict
        Configuration dictionary
    config_path : str, default="config.json"
        Path to save configuration file
        
    Example
    -------
    >>> config = {"random_state": 42, "test_size": 0.2}
    >>> save_config(config, "my_config.json")
    """
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=4)


class Timer:
    """
    Context manager for timing code execution.
    
    Example
    -------
    >>> with Timer("Data loading"):
    ...     df = pd.read_csv("large_file.csv")
    Data loading completed in 2.34 seconds
    """
    
    def __init__(self, name: str = "Operation", logger: Optional[logging.Logger] = None):
        self.name = name
        self.logger = logger or logging.getLogger("fraud_detection")
        self.start_time = None
        
    def __enter__(self):
        import time
        self.start_time = time.time()
        self.logger.info(f"{self.name} started...")
        return self
        
    def __exit__(self, *args):
        import time
        elapsed = time.time() - self.start_time
        self.logger.info(f"{self.name} completed in {elapsed:.2f} seconds")


def format_metric_table(metrics: Dict[str, Dict[str, float]]) -> pd.DataFrame:
    """
    Format model metrics into a nice DataFrame.
    
    Parameters
    ----------
    metrics : dict
        Dictionary of model metrics
        
    Returns
    -------
    pd.DataFrame
        Formatted metrics table
        
    Example
    -------
    >>> metrics = {
    ...     "LogReg": {"AUC": 0.906, "Recall": 0.703, "Precision": 0.881},
    ...     "RF": {"AUC": 0.861, "Recall": 0.662, "Precision": 0.942}
    ... }
    >>> df = format_metric_table(metrics)
    """
    df = pd.DataFrame(metrics).T
    df.index.name = "Model"
    return df.round(4)


def calculate_cost(
    confusion_matrix: np.ndarray,
    cost_ratio: float = 1.0
) -> float:
    """
    Calculate total cost given confusion matrix and cost ratio.
    
    Parameters
    ----------
    confusion_matrix : np.ndarray
        2x2 confusion matrix [[TN, FP], [FN, TP]]
    cost_ratio : float, default=1.0
        Ratio of False Negative cost to False Positive cost
        
    Returns
    -------
    float
        Total normalized cost
        
    Example
    -------
    >>> cm = np.array([[560, 8], [22, 52]])
    >>> cost = calculate_cost(cm, cost_ratio=5.0)
    >>> print(f"Total cost: {cost}")
    """
    tn, fp, fn, tp = confusion_matrix.ravel()
    return fp + cost_ratio * fn


def get_feature_list() -> List[str]:
    """
    Return the canonical list of features used by the models.
    
    Returns
    -------
    list
        List of feature names
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


def get_cold_start_features() -> List[str]:
    """
    Return the feature list for cold-start model (no historical rates).
    
    Returns
    -------
    list
        List of cold-start feature names
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


def print_banner(text: str, char: str = "=", width: int = 70) -> None:
    """
    Print a formatted banner for section headers.
    
    Parameters
    ----------
    text : str
        Text to display in banner
    char : str, default="="
        Character to use for banner
    width : int, default=70
        Width of the banner
        
    Example
    -------
    >>> print_banner("MODEL TRAINING")
    ======================================================================
    MODEL TRAINING
    ======================================================================
    """
    print("\n" + char * width)
    print(text)
    print(char * width)


if __name__ == "__main__":
    # Test the utilities
    print("Testing Payment Fraud Detection Utilities\n")
    
    # Test logging
    logger = setup_logging(log_level="INFO")
    logger.info("✅ Logging configured successfully")
    
    # Test output directories
    dirs = create_output_dirs("test_outputs")
    logger.info(f"✅ Created {len(dirs)} output directories")
    
    # Test timer
    with Timer("Sample operation", logger):
        import time
        time.sleep(0.5)
    
    # Test feature lists
    features = get_feature_list()
    logger.info(f"✅ Full feature set: {len(features)} features")
    
    cold_features = get_cold_start_features()
    logger.info(f"✅ Cold-start feature set: {len(cold_features)} features")
    
    print("\n✅ All utility tests passed!")
