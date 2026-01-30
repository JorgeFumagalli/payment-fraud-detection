"""
Main Script for Payment Fraud Detection System
================================================

This script provides a complete end-to-end pipeline:
1. Load and prepare data
2. Engineer features
3. Train multiple models
4. Evaluate performance
5. Generate visualizations
6. Export results

Usage:
------
python src/main.py --data data/raw/df.xlsx --output outputs/
"""

import argparse
import sys
from pathlib import Path

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from src.utils import setup_logging, create_output_dirs, print_banner, Timer
from src.data_loader import load_data, prepare_data, split_train_test, validate_data
from src.feature_engineering import FeatureEngineer
from src.models import FraudDetectionPipeline, calculate_cost_sensitivity

# Configure matplotlib
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Payment Fraud Detection - Complete Pipeline"
    )
    
    parser.add_argument(
        "--data",
        type=str,
        required=True,
        help="Path to transaction data file (CSV, Excel, or Parquet)"
    )
    
    parser.add_argument(
        "--output",
        type=str,
        default="outputs",
        help="Output directory for results (default: outputs/)"
    )
    
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.2,
        help="Proportion of data for test set (default: 0.2)"
    )
    
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)"
    )
    
    parser.add_argument(
        "--optimize",
        action="store_true",
        help="Enable hyperparameter optimization (slower but better)"
    )
    
    parser.add_argument(
        "--skip-viz",
        action="store_true",
        help="Skip visualization generation (faster)"
    )
    
    return parser.parse_args()


def main():
    """
    Main execution function.
    """
    # Parse arguments
    args = parse_arguments()
    
    # Setup logging
    logger = setup_logging(log_level="INFO", log_file="fraud_detection.log")
    logger.info("=" * 70)
    logger.info("PAYMENT_FRAUD FRAUD DETECTION SYSTEM")
    logger.info("=" * 70)
    
    # Create output directories
    dirs = create_output_dirs(args.output)
    logger.info(f"✅ Output directory: {args.output}")
    
    # ========================================================================
    # STEP 1: DATA LOADING
    # ========================================================================
    print_banner("STEP 1: DATA LOADING")
    
    with Timer("Data loading", logger):
        df = load_data(args.data)
    
    # Validate data
    diagnostics = validate_data(df)
    logger.info(f"Dataset: {diagnostics['n_transactions']} transactions")
    logger.info(f"Fraud rate: {diagnostics['fraud_rate']*100:.2f}%")
    logger.info(f"Missing device rate: {diagnostics['missing_device_rate']*100:.1f}%")
    
    # Generate EDA plots
    if not args.skip_viz:
        logger.info("Generating EDA plots...")
        generate_eda_plots(df, dirs['eda'])
    
    # ========================================================================
    # STEP 2: DATA PREPARATION
    # ========================================================================
    print_banner("STEP 2: DATA PREPARATION")
    
    with Timer("Data preparation", logger):
        df_prepared = prepare_data(df)
    
    # Split train/test
    train_df, test_df = split_train_test(
        df_prepared,
        test_size=args.test_size,
        random_state=args.random_state
    )
    
    # ========================================================================
    # STEP 3: FEATURE ENGINEERING
    # ========================================================================
    print_banner("STEP 3: FEATURE ENGINEERING")
    
    with Timer("Feature engineering", logger):
        engineer = FeatureEngineer()
        train_fe = engineer.fit_transform(train_df)
        test_fe = engineer.transform(test_df)
    
    # Extract features and labels
    features = engineer.get_feature_names()
    X_train = train_fe[features].fillna(0)
    y_train = train_fe['has_cbk']
    X_test = test_fe[features].fillna(0)
    y_test = test_fe['has_cbk']
    
    logger.info(f"✅ Created {len(features)} features")
    logger.info(f"✅ Train set: {X_train.shape}")
    logger.info(f"✅ Test set: {X_test.shape}")
    
    # Save processed data
    logger.info("Saving processed datasets...")
    train_fe.to_csv(dirs['data_processed'] / 'train_features.csv', index=False)
    test_fe.to_csv(dirs['data_processed'] / 'test_features.csv', index=False)
    logger.info(f"✅ Processed data saved to: {dirs['data_processed']}")
    
    # ========================================================================
    # STEP 4: MODEL TRAINING
    # ========================================================================
    print_banner("STEP 4: MODEL TRAINING")
    
    pipeline = FraudDetectionPipeline(random_state=args.random_state)
    
    with Timer("Model training", logger):
        pipeline.train_all(
            X_train,
            y_train,
            optimize_hyperparams=args.optimize
        )
    
    # ========================================================================
    # STEP 5: MODEL EVALUATION
    # ========================================================================
    print_banner("STEP 5: MODEL EVALUATION")
    
    with Timer("Model evaluation", logger):
        results = pipeline.evaluate(X_test, y_test)
    
    # Create results DataFrame
    results_df = pd.DataFrame(results).T
    results_df = results_df[['AUC', 'Recall', 'Precision', 'F1', 'TP', 'FN', 'FP', 'TN']]
    
    # Display results
    print("\nModel Performance Summary:")
    print(results_df.to_string())
    
    # Save results
    results_df.to_csv(dirs['tables'] / 'model_comparison.csv')
    logger.info(f"✅ Saved: {dirs['tables'] / 'model_comparison.csv'}")
    
    # ========================================================================
    # STEP 6: VISUALIZATIONS (optional)
    # ========================================================================
    if not args.skip_viz:
        print_banner("STEP 6: GENERATING VISUALIZATIONS")
        
        with Timer("Visualization generation", logger):
            generate_visualizations(
                pipeline,
                X_test,
                y_test,
                results,
                dirs
            )
    
    # ========================================================================
    # STEP 7: SAVE PIPELINE
    # ========================================================================
    print_banner("STEP 7: SAVING PIPELINE")
    
    pipeline_path = dirs['models'] / 'fraud_detection_pipeline.pkl'
    pipeline.save(str(pipeline_path))
    
    # ========================================================================
    # FINAL SUMMARY
    # ========================================================================
    print_banner("EXECUTION SUMMARY")
    
    best_model = pipeline.best_model_name
    best_auc = results[best_model]['AUC']
    best_recall = results[best_model]['Recall']
    best_precision = results[best_model]['Precision']
    
    print(f"\n✅ Best Model: {best_model}")
    print(f"   AUC: {best_auc:.4f}")
    print(f"   Recall: {best_recall:.4f}")
    print(f"   Precision: {best_precision:.4f}")
    print(f"\n✅ Pipeline saved to: {pipeline_path}")
    print(f"✅ Results saved to: {dirs['tables']}")
    
    if not args.skip_viz:
        print(f"✅ Visualizations saved to: {dirs['curves']}, {dirs['shap']}")
    
    logger.info("\n" + "=" * 70)
    logger.info("EXECUTION COMPLETED SUCCESSFULLY")
    logger.info("=" * 70)


def generate_eda_plots(df: pd.DataFrame, output_dir: Path) -> None:
    """
    Generate exploratory data analysis plots.
    
    Parameters
    ----------
    df : pd.DataFrame
        Raw transaction data
    output_dir : Path
        Directory to save EDA plots
    """
    print("   Generating EDA plots...")
    
    # 1. Transaction Amount Distribution
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    sns.histplot(df['transaction_amount'], bins=50, kde=True, color='skyblue', ax=axes[0])
    axes[0].set_title('Transaction Amount Distribution', fontsize=12, fontweight='bold')
    axes[0].set_xlabel('Transaction Amount')
    axes[0].set_ylabel('Frequency')
    
    sns.boxplot(x=df['transaction_amount'], color='lightcoral', ax=axes[1])
    axes[1].set_title('Transaction Amount Boxplot', fontsize=12, fontweight='bold')
    axes[1].set_xlabel('Transaction Amount')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'transaction_amount_distribution.png', dpi=300)
    plt.close()
    print("   ✅ Transaction amount distribution saved")
    
    # 2. Fraud Rate by Hour (if hour column exists)
    if 'hour' in df.columns and 'has_cbk' in df.columns:
        fraud_by_hour = df.groupby('hour')['has_cbk'].agg(['mean', 'count'])
        fraud_by_hour['fraud_rate'] = fraud_by_hour['mean'] * 100
        
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.bar(fraud_by_hour.index, fraud_by_hour['fraud_rate'], 
               color='steelblue', alpha=0.8, edgecolor='black')
        ax.axhline(y=df['has_cbk'].mean() * 100, color='red', 
                   linestyle='--', label='Average Fraud Rate', linewidth=2)
        ax.set_xlabel('Hour of Day', fontsize=12)
        ax.set_ylabel('Fraud Rate (%)', fontsize=12)
        ax.set_title('Fraud Rate by Hour of Day', fontsize=14, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(alpha=0.3, axis='y')
        
        plt.tight_layout()
        plt.savefig(output_dir / 'fraud_rate_by_hour.png', dpi=300)
        plt.close()
        print("   ✅ Fraud rate by hour saved")
    
    # 3. Fraud Rate by Day of Week (if available)
    if 'day_of_week' in df.columns and 'has_cbk' in df.columns:
        fraud_by_dow = df.groupby('day_of_week')['has_cbk'].mean() * 100
        days = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
        
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.bar(range(7), fraud_by_dow.values, color='teal', alpha=0.8, edgecolor='black')
        ax.set_xticks(range(7))
        ax.set_xticklabels(days)
        ax.axhline(y=df['has_cbk'].mean() * 100, color='red', 
                   linestyle='--', label='Average', linewidth=2)
        ax.set_xlabel('Day of Week', fontsize=12)
        ax.set_ylabel('Fraud Rate (%)', fontsize=12)
        ax.set_title('Fraud Rate by Day of Week', fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(alpha=0.3, axis='y')
        
        plt.tight_layout()
        plt.savefig(output_dir / 'fraud_rate_by_day.png', dpi=300)
        plt.close()
        print("   ✅ Fraud rate by day saved")
    
    print("   ✅ EDA plots complete")


def generate_visualizations(
    pipeline: FraudDetectionPipeline,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    results: dict,
    dirs: dict
) -> None:
    """
    Generate all visualizations.
    
    Parameters
    ----------
    pipeline : FraudDetectionPipeline
        Trained pipeline
    X_test : pd.DataFrame
        Test features
    y_test : pd.Series
        Test labels
    results : dict
        Evaluation results
    dirs : dict
        Output directories
    """
    from sklearn.metrics import roc_curve, precision_recall_curve, confusion_matrix
    import seaborn as sns
    
    print("   Generating visualizations...")
    
    # 1. ROC Curves
    print("   - ROC curves...")
    fig, ax = plt.subplots(figsize=(10, 7))
    
    for model_name in pipeline.models.keys():
        y_proba = pipeline.predict_proba(X_test, model_name)
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        auc = results[model_name]['AUC']
        ax.plot(fpr, tpr, label=f'{model_name} (AUC={auc:.4f})', lw=2)
    
    ax.plot([0, 1], [0, 1], 'k--', lw=2, label='Random')
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate (Recall)')
    ax.set_title('ROC Curves - Model Comparison')
    ax.legend()
    ax.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(dirs['curves'] / 'roc_curves.png', dpi=300)
    plt.close()
    
    # 2. Precision-Recall Curves
    print("   - PR curves...")
    fig, ax = plt.subplots(figsize=(10, 7))
    
    for model_name in pipeline.models.keys():
        y_proba = pipeline.predict_proba(X_test, model_name)
        precision, recall, _ = precision_recall_curve(y_test, y_proba)
        pr_auc = results[model_name]['PR_AUC']
        ax.plot(recall, precision, label=f'{model_name} (PR-AUC={pr_auc:.4f})', lw=2)
    
    ax.set_xlabel('Recall')
    ax.set_ylabel('Precision')
    ax.set_title('Precision-Recall Curves - Model Comparison')
    ax.legend()
    ax.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(dirs['curves'] / 'pr_curves.png', dpi=300)
    plt.close()
    
    # 3. Cost Sensitivity Analysis
    print("   - Cost sensitivity analysis...")
    r_values = np.linspace(0, 10, 500)
    cost_curves = calculate_cost_sensitivity(results, r_values)
    
    fig, ax = plt.subplots(figsize=(12, 7))
    
    for model_name, costs in cost_curves.items():
        ax.plot(r_values, costs, label=model_name, lw=2.2, alpha=0.9)
    
    ax.axvline(1, color='k', ls=':', lw=1, alpha=0.5)
    ax.text(1.05, ax.get_ylim()[1]*0.95, 'r=1 (equal costs)', fontsize=9)
    ax.set_xlabel("r = Cost(False Negative) / Cost(False Positive)")
    ax.set_ylabel("Normalized Total Cost")
    ax.set_title("Cost Sensitivity Analysis - Optimal Model by Cost Ratio")
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(dirs['curves'] / 'cost_sensitivity.png', dpi=300)
    plt.close()
    
    # 4. Confusion Matrices (individual for each model)
    print("   - Confusion matrices...")
    for model_name in pipeline.models.keys():
        y_proba = pipeline.predict_proba(X_test, model_name)
        y_pred = (y_proba >= 0.5).astype(int)
        cm = confusion_matrix(y_test, y_pred)
        
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
                    xticklabels=['Legitimate', 'Fraud'],
                    yticklabels=['Legitimate', 'Fraud'],
                    ax=ax)
        ax.set_xlabel('Predicted')
        ax.set_ylabel('Actual')
        ax.set_title(f'{model_name} - Confusion Matrix')
        
        plt.tight_layout()
        safe_name = model_name.replace(' ', '_').replace('(', '').replace(')', '')
        plt.savefig(dirs['confusion_matrices'] / f'{safe_name}_cm.png', dpi=300)
        plt.close()
    
    # 5. SHAP Analysis (for best model)
    print("   - SHAP analysis...")
    try:
        import shap
        
        # Use Random Forest for SHAP (most stable)
        if 'RF' in pipeline.models:
            model_for_shap = pipeline.models['RF']
            model_name_shap = 'RF'
        else:
            model_for_shap = pipeline.models[pipeline.best_model_name]
            model_name_shap = pipeline.best_model_name
        
        # Sample for SHAP (max 300 rows for speed)
        sample_size = min(300, len(X_test))
        X_sample = X_test.iloc[:sample_size]
        
        # Create explainer
        explainer = shap.TreeExplainer(model_for_shap)
        shap_values = explainer.shap_values(X_sample)
        
        # Handle different SHAP output formats
        if isinstance(shap_values, list):
            shap_values = shap_values[1]  # For binary classification
        elif hasattr(shap_values, 'shape') and len(shap_values.shape) == 3:
            shap_values = shap_values[:, :, 1]
        
        # SHAP Summary Plot (bar)
        plt.figure(figsize=(10, 7))
        shap.summary_plot(shap_values, X_sample, plot_type='bar', max_display=10, show=False)
        plt.title(f'Global Feature Importance (SHAP) - {model_name_shap}')
        plt.tight_layout()
        plt.savefig(dirs['shap'] / 'shap_bar_top10.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # SHAP Beeswarm Plot
        plt.figure(figsize=(12, 7))
        shap.summary_plot(shap_values, X_sample, max_display=10, show=False)
        plt.title(f'SHAP Beeswarm - Top 10 Features ({model_name_shap})')
        plt.tight_layout()
        plt.savefig(dirs['shap'] / 'shap_beeswarm_top10.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print("   ✅ SHAP plots generated")
        
    except ImportError:
        print("   ⚠️  SHAP not installed - skipping SHAP plots")
    except Exception as e:
        print(f"   ⚠️  SHAP generation failed: {e}")
    
    print("   ✅ All visualizations generated")
    print("   ✅ ROC curves saved")
    print("   ✅ PR curves saved")
    print("   ✅ Cost sensitivity plot saved")
    print("   ✅ Confusion matrices saved")
    print("   ✅ SHAP plots saved")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\n❌ Error: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)
