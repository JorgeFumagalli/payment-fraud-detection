"""
Script de Exemplo: Fazer Predições em Novas Transações
=======================================================

Este script demonstra como usar o modelo treinado para fazer predições
em novas transações.

Uso:
----
python predict_example.py --input data/raw/new_transactions.xlsx --output outputs/predictions.csv
"""

import argparse
import pandas as pd
from pathlib import Path

from src.models import FraudDetectionPipeline
from src.feature_engineering import FeatureEngineer
from src.data_loader import load_data, prepare_data
from src.utils import setup_logging


def main():
    """Execute prediction pipeline."""
    parser = argparse.ArgumentParser(description="Make fraud predictions on new transactions")
    parser.add_argument("--input", type=str, required=True, help="Path to new transactions file")
    parser.add_argument("--output", type=str, default="outputs/predictions.csv", help="Output file path")
    parser.add_argument("--model", type=str, default="outputs/models/fraud_detection_pipeline.pkl", help="Trained model path")
    parser.add_argument("--threshold", type=float, default=0.5, help="Classification threshold")
    args = parser.parse_args()
    
    # Setup logging
    logger = setup_logging(log_level="INFO")
    logger.info("=" * 70)
    logger.info("FRAUD PREDICTION PIPELINE")
    logger.info("=" * 70)
    
    # 1. Load trained model
    logger.info(f"Loading model from {args.model}...")
    try:
        pipeline = FraudDetectionPipeline.load(args.model)
        logger.info(f"✅ Model loaded: {pipeline.best_model_name}")
    except FileNotFoundError:
        logger.error(f"❌ Model not found at {args.model}")
        logger.error("   Please train a model first: python src/main.py --data data/raw/df.xlsx")
        return
    
    # 2. Load new transactions
    logger.info(f"Loading new transactions from {args.input}...")
    try:
        df = load_data(args.input)
        logger.info(f"✅ Loaded {len(df)} transactions")
    except FileNotFoundError:
        logger.error(f"❌ Input file not found: {args.input}")
        return
    
    # 3. Prepare data
    logger.info("Preparing data...")
    df_prepared = prepare_data(df)
    
    # 4. Feature engineering
    logger.info("Engineering features...")
    logger.warning("⚠️  Note: This script assumes you're using the same feature engineering")
    logger.warning("    as during training. For production, save and load the FeatureEngineer.")
    
    # For this example, we'll retrain the feature engineer on the provided data
    # In production, you should save the fitted FeatureEngineer and load it here
    engineer = FeatureEngineer()
    
    # Check if we have training data to fit the engineer
    if 'has_cbk' in df_prepared.columns:
        logger.info("Training label found - fitting feature engineer...")
        df_features = engineer.fit_transform(df_prepared)
    else:
        logger.warning("⚠️  No training labels found. Using global defaults.")
        logger.warning("    Results may be less accurate. Consider providing training data.")
        # Add dummy labels for feature engineering
        df_prepared['has_cbk'] = 0
        df_features = engineer.fit_transform(df_prepared)
        df_features.drop('has_cbk', axis=1, inplace=True)
    
    # 5. Extract features
    feature_names = pipeline.feature_names
    X = df_features[feature_names].fillna(0)
    
    logger.info(f"✅ Features ready: {X.shape}")
    
    # 6. Make predictions
    logger.info(f"Making predictions (threshold={args.threshold})...")
    probabilities = pipeline.predict_proba(X)
    predictions = (probabilities >= args.threshold).astype(int)
    
    # 7. Add results to dataframe
    df['fraud_probability'] = probabilities
    df['fraud_prediction'] = predictions
    df['fraud_risk'] = pd.cut(
        probabilities,
        bins=[0, 0.3, 0.7, 1.0],
        labels=['Low', 'Medium', 'High']
    )
    
    # 8. Summary statistics
    n_frauds = predictions.sum()
    fraud_rate = predictions.mean() * 100
    avg_prob = probabilities.mean() * 100
    
    logger.info("\n" + "=" * 70)
    logger.info("PREDICTION RESULTS")
    logger.info("=" * 70)
    logger.info(f"Total transactions: {len(df)}")
    logger.info(f"Predicted frauds: {n_frauds} ({fraud_rate:.2f}%)")
    logger.info(f"Average fraud probability: {avg_prob:.2f}%")
    logger.info(f"Model used: {pipeline.best_model_name}")
    
    # Risk distribution
    risk_dist = df['fraud_risk'].value_counts().sort_index()
    logger.info("\nRisk Distribution:")
    for risk, count in risk_dist.items():
        pct = (count / len(df)) * 100
        logger.info(f"  {risk:7s}: {count:5d} ({pct:5.2f}%)")
    
    # 9. Save results
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    logger.info(f"\n✅ Results saved to: {output_path}")
    
    # 10. Show high-risk transactions
    high_risk = df[df['fraud_prediction'] == 1].sort_values('fraud_probability', ascending=False)
    if len(high_risk) > 0:
        logger.info(f"\nTop 5 High-Risk Transactions:")
        logger.info("-" * 70)
        for idx, row in high_risk.head(5).iterrows():
            logger.info(f"  Transaction ID: {row['transaction_id']}")
            logger.info(f"    Amount: ${row['transaction_amount']:.2f}")
            logger.info(f"    Probability: {row['fraud_probability']*100:.2f}%")
            logger.info(f"    User: {row['user_id']} | Merchant: {row['merchant_id']}")
            logger.info("")
    
    logger.info("=" * 70)
    logger.info("✅ PREDICTION COMPLETE")
    logger.info("=" * 70)


if __name__ == "__main__":
    main()
