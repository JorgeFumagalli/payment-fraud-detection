"""
Data Loading and Preprocessing Module
=====================================

This module handles:
- Loading transaction data from various formats
- Basic data cleaning and validation
- Train/test splitting with stratification
- Data quality checks
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Tuple, Optional, Dict, Any
import logging
from sklearn.model_selection import train_test_split


logger = logging.getLogger("fraud_detection")


def load_data(
    filepath: str,
    file_format: Optional[str] = None
) -> pd.DataFrame:
    """
    Load transaction data from file.
    
    Supports CSV, Excel, and Parquet formats. Auto-detects format from extension
    if not specified.
    
    Parameters
    ----------
    filepath : str
        Path to the data file
    file_format : str, optional
        File format ('csv', 'excel', 'parquet'). Auto-detected if None
        
    Returns
    -------
    pd.DataFrame
        Loaded transaction data
        
    Raises
    ------
    FileNotFoundError
        If file does not exist
    ValueError
        If file format is not supported
        
    Example
    -------
    >>> df = load_data("data/raw/transactions.xlsx")
    >>> print(df.shape)
    (3199, 7)
    """
    filepath = Path(filepath)
    
    if not filepath.exists():
        raise FileNotFoundError(f"File not found: {filepath}")
    
    # Auto-detect format
    if file_format is None:
        extension = filepath.suffix.lower()
        format_map = {
            '.csv': 'csv',
            '.xlsx': 'excel',
            '.xls': 'excel',
            '.parquet': 'parquet',
        }
        file_format = format_map.get(extension)
        
        if file_format is None:
            raise ValueError(f"Unsupported file extension: {extension}")
    
    logger.info(f"Loading data from {filepath} (format: {file_format})")
    
    # Load based on format
    if file_format == 'csv':
        df = pd.read_csv(filepath)
    elif file_format == 'excel':
        df = pd.read_excel(filepath)
    elif file_format == 'parquet':
        df = pd.read_parquet(filepath)
    else:
        raise ValueError(f"Unsupported file format: {file_format}")
    
    logger.info(f"✅ Loaded {len(df)} transactions with {len(df.columns)} columns")
    
    return df


def validate_data(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Validate data quality and return diagnostics.
    
    Parameters
    ----------
    df : pd.DataFrame
        Transaction data
        
    Returns
    -------
    dict
        Validation results including:
        - missing_values: Count per column
        - fraud_rate: Overall fraud percentage
        - date_range: Min and max transaction dates
        - duplicate_transactions: Count of duplicates
        
    Example
    -------
    >>> diagnostics = validate_data(df)
    >>> print(f"Fraud rate: {diagnostics['fraud_rate']:.2%}")
    """
    required_columns = [
        'transaction_id',
        'transaction_date',
        'transaction_amount',
        'user_id',
        'merchant_id',
        'has_cbk'
    ]
    
    # Check required columns
    missing_cols = set(required_columns) - set(df.columns)
    if missing_cols:
        logger.warning(f"Missing required columns: {missing_cols}")
    
    diagnostics = {
        'n_transactions': len(df),
        'n_columns': len(df.columns),
        'missing_values': df.isnull().sum().to_dict(),
        'fraud_rate': df['has_cbk'].mean() if 'has_cbk' in df.columns else None,
        'date_range': (
            df['transaction_date'].min(),
            df['transaction_date'].max()
        ) if 'transaction_date' in df.columns else None,
        'duplicate_transactions': df.duplicated(subset=['transaction_id']).sum() if 'transaction_id' in df.columns else None,
        'missing_device_rate': df['device_id'].isnull().mean() if 'device_id' in df.columns else None,
    }
    
    return diagnostics


def prepare_data(
    df: pd.DataFrame,
    handle_missing: bool = True,
    parse_dates: bool = True
) -> pd.DataFrame:
    """
    Prepare raw data for feature engineering.
    
    This function performs:
    - Type conversions
    - Date parsing
    - Missing value handling
    - Basic feature creation (hour, day_of_week)
    
    Parameters
    ----------
    df : pd.DataFrame
        Raw transaction data
    handle_missing : bool, default=True
        Whether to handle missing values
    parse_dates : bool, default=True
        Whether to parse transaction_date column
        
    Returns
    -------
    pd.DataFrame
        Prepared data ready for feature engineering
        
    Example
    -------
    >>> df_prepared = prepare_data(df_raw)
    >>> print(df_prepared.columns)
    """
    df = df.copy()
    
    logger.info("Preparing data for feature engineering...")
    
    # Parse dates
    if parse_dates and 'transaction_date' in df.columns:
        df['transaction_date'] = pd.to_datetime(
            df['transaction_date'],
            errors='coerce'
        )
        logger.info("✅ Parsed transaction_date column")
        
        # Extract temporal features
        df['hour'] = df['transaction_date'].dt.hour
        df['day_of_week'] = df['transaction_date'].dt.dayofweek
        logger.info("✅ Created hour and day_of_week features")
    
    # Handle missing device_id
    if handle_missing and 'device_id' in df.columns:
        df['device_missing'] = df['device_id'].isna().astype(int)
        df['device_id'] = df['device_id'].fillna(-1)
        
        missing_pct = df['device_missing'].mean() * 100
        logger.info(f"✅ Handled missing device_id ({missing_pct:.1f}% missing)")
    
    # Convert target to int
    if 'has_cbk' in df.columns:
        df['has_cbk'] = df['has_cbk'].astype(int)
        logger.info(f"✅ Target variable: {df['has_cbk'].sum()} frauds ({df['has_cbk'].mean()*100:.2f}%)")
    
    # Remove exact duplicates
    n_before = len(df)
    df = df.drop_duplicates(subset=['transaction_id'], keep='first')
    n_after = len(df)
    if n_before > n_after:
        logger.warning(f"Removed {n_before - n_after} duplicate transactions")
    
    return df


def split_train_test(
    df: pd.DataFrame,
    test_size: float = 0.2,
    random_state: int = 42,
    stratify_column: str = 'has_cbk'
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split data into train and test sets with stratification.
    
    Parameters
    ----------
    df : pd.DataFrame
        Full dataset
    test_size : float, default=0.2
        Proportion of data for test set
    random_state : int, default=42
        Random seed for reproducibility
    stratify_column : str, default='has_cbk'
        Column to stratify on (maintains fraud rate in both sets)
        
    Returns
    -------
    tuple
        (train_df, test_df)
        
    Example
    -------
    >>> train, test = split_train_test(df, test_size=0.2)
    >>> print(f"Train: {len(train)}, Test: {len(test)}")
    Train: 2557, Test: 642
    """
    logger.info(f"Splitting data: {100*(1-test_size):.0f}% train, {100*test_size:.0f}% test")
    
    # Use stratification if column exists
    stratify = df[stratify_column] if stratify_column in df.columns else None
    
    train_df, test_df = train_test_split(
        df,
        test_size=test_size,
        random_state=random_state,
        stratify=stratify
    )
    
    train_df = train_df.reset_index(drop=True)
    test_df = test_df.reset_index(drop=True)
    
    # Report fraud rates
    if 'has_cbk' in df.columns:
        train_fraud_rate = train_df['has_cbk'].mean() * 100
        test_fraud_rate = test_df['has_cbk'].mean() * 100
        logger.info(f"✅ Train fraud rate: {train_fraud_rate:.2f}%")
        logger.info(f"✅ Test fraud rate: {test_fraud_rate:.2f}%")
    
    return train_df, test_df


def get_data_summary(df: pd.DataFrame) -> pd.DataFrame:
    """
    Generate a summary report of the dataset.
    
    Parameters
    ----------
    df : pd.DataFrame
        Transaction data
        
    Returns
    -------
    pd.DataFrame
        Summary statistics
        
    Example
    -------
    >>> summary = get_data_summary(df)
    >>> print(summary)
    """
    summary = pd.DataFrame({
        'dtype': df.dtypes,
        'missing': df.isnull().sum(),
        'missing_pct': (df.isnull().sum() / len(df) * 100).round(2),
        'unique': df.nunique(),
    })
    
    # Add min/max for numeric columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    summary.loc[numeric_cols, 'min'] = df[numeric_cols].min().values
    summary.loc[numeric_cols, 'max'] = df[numeric_cols].max().values
    summary.loc[numeric_cols, 'mean'] = df[numeric_cols].mean().values
    
    return summary


if __name__ == "__main__":
    """
    Test the data loading module
    """
    import sys
    from src.utils import setup_logging
    
    logger = setup_logging(log_level="INFO")
    
    print("Testing Data Loading Module\n")
    
    # This is a test - in production, use actual data path
    test_data = pd.DataFrame({
        'transaction_id': range(100),
        'transaction_date': pd.date_range('2024-01-01', periods=100, freq='H'),
        'transaction_amount': np.random.lognormal(6, 1, 100),
        'user_id': np.random.randint(1, 20, 100),
        'merchant_id': np.random.randint(1, 10, 100),
        'device_id': np.random.choice([1, 2, 3, None], 100),
        'has_cbk': np.random.choice([0, 1], 100, p=[0.88, 0.12])
    })
    
    print("1. Validating data...")
    diagnostics = validate_data(test_data)
    print(f"   ✅ {diagnostics['n_transactions']} transactions")
    print(f"   ✅ Fraud rate: {diagnostics['fraud_rate']*100:.2f}%")
    
    print("\n2. Preparing data...")
    df_prepared = prepare_data(test_data)
    print(f"   ✅ Created {len(df_prepared.columns)} columns")
    
    print("\n3. Splitting train/test...")
    train, test = split_train_test(df_prepared, test_size=0.2)
    print(f"   ✅ Train: {len(train)} transactions")
    print(f"   ✅ Test: {len(test)} transactions")
    
    print("\n4. Generating summary...")
    summary = get_data_summary(df_prepared)
    print(summary.head())
    
    print("\n✅ All data loading tests passed!")
