"""
Generate Visualizations from Trained Model
==========================================

Este script gera todas as visualizações (EDA, confusion matrices, SHAP)
a partir de um modelo já treinado.

Uso:
----
python generate_visualizations.py --data data/raw/df.xlsx --model outputs/models/fraud_detection_pipeline.pkl
"""

import argparse
import sys
from pathlib import Path

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Backend sem display para servidores
import matplotlib.pyplot as plt
import seaborn as sns

from src.utils import setup_logging, create_output_dirs, print_banner
from src.data_loader import load_data, prepare_data, split_train_test
from src.feature_engineering import FeatureEngineer
from src.models import FraudDetectionPipeline

# Configure matplotlib
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Generate visualizations from trained model")
    
    parser.add_argument("--data", type=str, required=True, help="Path to data file")
    parser.add_argument("--model", type=str, default="outputs/models/fraud_detection_pipeline.pkl", 
                        help="Path to trained model")
    parser.add_argument("--output", type=str, default="outputs", help="Output directory")
    parser.add_argument("--test-size", type=float, default=0.2, help="Test set proportion")
    parser.add_argument("--random-state", type=int, default=42, help="Random seed")
    
    return parser.parse_args()


def generate_eda_plots(df: pd.DataFrame, output_dir: Path) -> None:
    """Generate exploratory data analysis plots."""
    print("\n1. Generating EDA plots...")
    
    # Transaction amount distribution
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    sns.histplot(df['transaction_amount'], bins=50, kde=True, color='skyblue', ax=axes[0])
    axes[0].set_title('Transaction Amount Distribution', fontsize=12, fontweight='bold')
    axes[0].set_xlabel('Transaction Amount')
    
    sns.boxplot(x=df['transaction_amount'], color='lightcoral', ax=axes[1])
    axes[1].set_title('Transaction Amount Boxplot', fontsize=12, fontweight='bold')
    axes[1].set_xlabel('Transaction Amount')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'transaction_amount.png', dpi=300)
    plt.close()
    print("   ✅ Transaction amount plots saved")
    
    # Fraud rate by hour
    if 'hour' in df.columns and 'has_cbk' in df.columns:
        fraud_by_hour = df.groupby('hour')['has_cbk'].mean() * 100
        
        plt.figure(figsize=(12, 6))
        plt.bar(fraud_by_hour.index, fraud_by_hour.values, color='steelblue', alpha=0.8)
        plt.axhline(y=df['has_cbk'].mean() * 100, color='red', linestyle='--', 
                    label='Average Fraud Rate', linewidth=2)
        plt.xlabel('Hour of Day')
        plt.ylabel('Fraud Rate (%)')
        plt.title('Fraud Rate by Hour of Day')
        plt.legend()
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(output_dir / 'fraud_by_hour.png', dpi=300)
        plt.close()
        print("   ✅ Fraud by hour plot saved")


def generate_confusion_matrices(
    pipeline: FraudDetectionPipeline,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    output_dir: Path
) -> None:
    """Generate confusion matrix for each model."""
    from sklearn.metrics import confusion_matrix
    
    print("\n2. Generating confusion matrices...")
    
    for model_name in pipeline.models.keys():
        y_proba = pipeline.predict_proba(X_test, model_name)
        y_pred = (y_proba >= 0.5).astype(int)
        cm = confusion_matrix(y_test, y_pred)
        
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
                    xticklabels=['Legitimate', 'Fraud'],
                    yticklabels=['Legitimate', 'Fraud'],
                    ax=ax, annot_kws={'size': 14})
        ax.set_xlabel('Predicted', fontsize=12)
        ax.set_ylabel('Actual', fontsize=12)
        ax.set_title(f'{model_name} - Confusion Matrix', fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        safe_name = model_name.replace(' ', '_').replace('(', '').replace(')', '')
        plt.savefig(output_dir / f'{safe_name}_cm.png', dpi=300)
        plt.close()
        print(f"   ✅ {model_name} confusion matrix saved")


def generate_performance_curves(
    pipeline: FraudDetectionPipeline,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    output_dir: Path
) -> None:
    """Generate ROC and PR curves."""
    from sklearn.metrics import roc_curve, precision_recall_curve, roc_auc_score, average_precision_score
    
    print("\n3. Generating performance curves...")
    
    # ROC Curves
    fig, ax = plt.subplots(figsize=(10, 7))
    
    for model_name in pipeline.models.keys():
        y_proba = pipeline.predict_proba(X_test, model_name)
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        auc = roc_auc_score(y_test, y_proba)
        ax.plot(fpr, tpr, label=f'{model_name} (AUC={auc:.4f})', lw=2.5)
    
    ax.plot([0, 1], [0, 1], 'k--', lw=2, label='Random', alpha=0.5)
    ax.set_xlabel('False Positive Rate', fontsize=12)
    ax.set_ylabel('True Positive Rate (Recall)', fontsize=12)
    ax.set_title('ROC Curves - Model Comparison', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'roc_curves.png', dpi=300)
    plt.close()
    print("   ✅ ROC curves saved")
    
    # PR Curves
    fig, ax = plt.subplots(figsize=(10, 7))
    
    for model_name in pipeline.models.keys():
        y_proba = pipeline.predict_proba(X_test, model_name)
        precision, recall, _ = precision_recall_curve(y_test, y_proba)
        pr_auc = average_precision_score(y_test, y_proba)
        ax.plot(recall, precision, label=f'{model_name} (PR-AUC={pr_auc:.4f})', lw=2.5)
    
    ax.set_xlabel('Recall', fontsize=12)
    ax.set_ylabel('Precision', fontsize=12)
    ax.set_title('Precision-Recall Curves - Model Comparison', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'pr_curves.png', dpi=300)
    plt.close()
    print("   ✅ PR curves saved")


def generate_shap_plots(
    pipeline: FraudDetectionPipeline,
    X_test: pd.DataFrame,
    output_dir: Path
) -> None:
    """Generate SHAP explanations."""
    print("\n4. Generating SHAP plots...")
    
    try:
        import shap
        
        # Use Random Forest (most stable for SHAP)
        if 'RF' in pipeline.models:
            model = pipeline.models['RF']
            model_name = 'Random Forest'
        else:
            model = pipeline.models[pipeline.best_model_name]
            model_name = pipeline.best_model_name
        
        # Sample data
        sample_size = min(300, len(X_test))
        X_sample = X_test.iloc[:sample_size]
        
        print(f"   Computing SHAP values for {model_name}...")
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_sample)
        
        # Handle different formats
        if isinstance(shap_values, list):
            shap_values = shap_values[1]
        elif hasattr(shap_values, 'shape') and len(shap_values.shape) == 3:
            shap_values = shap_values[:, :, 1]
        
        # Bar plot
        plt.figure(figsize=(10, 7))
        shap.summary_plot(shap_values, X_sample, plot_type='bar', 
                         max_display=10, show=False)
        plt.title(f'Global Feature Importance (SHAP) - {model_name}', 
                 fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(output_dir / 'shap_bar_top10.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("   ✅ SHAP bar plot saved")
        
        # Beeswarm plot
        plt.figure(figsize=(12, 7))
        shap.summary_plot(shap_values, X_sample, max_display=10, show=False)
        plt.title(f'SHAP Beeswarm - Top 10 Features ({model_name})', 
                 fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(output_dir / 'shap_beeswarm_top10.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("   ✅ SHAP beeswarm plot saved")
        
    except ImportError:
        print("   ⚠️  SHAP not installed. Install with: pip install shap")
    except Exception as e:
        print(f"   ⚠️  SHAP generation failed: {e}")


def main():
    """Main execution."""
    args = parse_arguments()
    
    # Setup logging
    logger = setup_logging(log_level="INFO")
    logger.info("=" * 70)
    logger.info("VISUALIZATION GENERATION")
    logger.info("=" * 70)
    
    # Create output directories
    dirs = create_output_dirs(args.output)
    
    # Load model
    print("\nLoading trained model...")
    try:
        pipeline = FraudDetectionPipeline.load(args.model)
        print(f"✅ Model loaded: {pipeline.best_model_name}")
    except FileNotFoundError:
        print(f"❌ Model not found at: {args.model}")
        print("   Train a model first: python src/main.py --data data/raw/df.xlsx")
        sys.exit(1)
    
    # Load and prepare data
    print("\nLoading data...")
    df = load_data(args.data)
    df_prepared = prepare_data(df)
    
    # Split data
    train_df, test_df = split_train_test(
        df_prepared,
        test_size=args.test_size,
        random_state=args.random_state
    )
    
    # Feature engineering
    print("\nEngineering features...")
    engineer = FeatureEngineer()
    train_fe = engineer.fit_transform(train_df)
    test_fe = engineer.transform(test_df)
    
    features = engineer.get_feature_names()
    X_test = test_fe[features].fillna(0)
    y_test = test_fe['has_cbk']
    
    # Generate all visualizations
    print_banner("GENERATING ALL VISUALIZATIONS")
    
    generate_eda_plots(df_prepared, dirs['eda'])
    generate_confusion_matrices(pipeline, X_test, y_test, dirs['confusion_matrices'])
    generate_performance_curves(pipeline, X_test, y_test, dirs['curves'])
    generate_shap_plots(pipeline, X_test, dirs['shap'])
    
    # Summary
    print_banner("VISUALIZATION COMPLETE")
    print("\n✅ All visualizations generated!")
    print(f"\nOutputs saved to:")
    print(f"  - EDA plots: {dirs['eda']}")
    print(f"  - Confusion matrices: {dirs['confusion_matrices']}")
    print(f"  - Performance curves: {dirs['curves']}")
    print(f"  - SHAP plots: {dirs['shap']}")
    
    # Count files
    total_files = sum(1 for _ in Path(args.output).rglob('*.png'))
    print(f"\n📊 Total visualizations: {total_files} PNG files")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\n❌ Error: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)
