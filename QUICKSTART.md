# Quick Start Guide

Get started with Payment Fraud Detection in 5 minutes!

## 🚀 Installation

```bash
# 1. Clone the repository
git clone https://github.com/yourusername/payment-fraud-detection.git
cd payment-fraud-detection

# 2. Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt
```

## 📊 Prepare Your Data

Your data should be in Excel/CSV format with these columns:

```
Required columns:
- transaction_id (str)
- transaction_date (datetime)
- transaction_amount (float)
- user_id (str/int)
- merchant_id (str/int)
- device_id (str/int, optional)
- has_cbk (int, 0 or 1)
```

Place your data file in `data/raw/df.xlsx`

## 🎯 Run Your First Model

```bash
# Complete pipeline (training + evaluation)
python src/main.py --data data/raw/df.xlsx --output outputs/
```

This will:
1. ✅ Load and validate your data
2. ✅ Engineer 25 behavioral features
3. ✅ Train 4 different models
4. ✅ Evaluate performance
5. ✅ Generate visualizations
6. ✅ Save trained models

## 📈 View Results

After running, check:

```bash
# Model performance comparison
cat outputs/tables/model_comparison.csv

# Feature importance
open outputs/shap/shap_bar_top10.png

# ROC curves
open outputs/curves/roc_curves.png
```

## 🔮 Make Predictions

```python
from src.models import FraudDetectionPipeline
import pandas as pd

# Load trained model
pipeline = FraudDetectionPipeline.load('outputs/models/fraud_detection_pipeline.pkl')

# Prepare new transactions
new_data = pd.read_csv('new_transactions.csv')

# Get predictions
fraud_probabilities = pipeline.predict_proba(new_data)
predictions = pipeline.predict(new_data, threshold=0.5)

print(f"Detected {predictions.sum()} frauds out of {len(predictions)} transactions")
```

## 🎨 Customize Configuration

Edit `config.json` to adjust:

```json
{
  "models": {
    "random_forest": {
      "n_estimators": 500,  // More trees = better but slower
      "max_depth": 15       // Deeper trees = more complex patterns
    }
  },
  "evaluation": {
    "threshold": 0.45      // Lower = catch more frauds (but more false positives)
  }
}
```

## 🔥 Common Use Cases

### Case 1: High-Precision Mode (Minimize False Positives)

```bash
# Use Random Forest with high threshold
python src/main.py \
    --data data/raw/df.xlsx \
    --output outputs/ \
    --threshold 0.6
```

### Case 2: High-Recall Mode (Catch More Frauds)

```bash
# Use XGBoost with lower threshold
python src/main.py \
    --data data/raw/df.xlsx \
    --output outputs/ \
    --threshold 0.35
```

### Case 3: Optimize Hyperparameters (Slow but Best)

```bash
# Enable Bayesian optimization
python src/main.py \
    --data data/raw/df.xlsx \
    --output outputs/ \
    --optimize
```

## 📚 Next Steps

1. **Read the full documentation**: [README.md](README.md)
2. **Understand the features**: Check feature engineering code in `src/feature_engineering.py`
3. **Deploy to production**: Follow [deployment_guide.md](docs/deployment_guide.md)
4. **Customize for your needs**: See [CONTRIBUTING.md](CONTRIBUTING.md)

## 🆘 Troubleshooting

### Problem: "ModuleNotFoundError: No module named 'src'"

**Solution:**
```bash
# Install package in editable mode
pip install -e .
```

### Problem: "File not found: data/raw/df.xlsx"

**Solution:**
```bash
# Make sure your data file exists
ls -la data/raw/

# Or specify custom path
python src/main.py --data /path/to/your/data.xlsx
```

### Problem: "Out of memory"

**Solution:**
```bash
# Reduce batch size or use simpler model
# Edit config.json:
{
  "models": {
    "random_forest": {
      "n_estimators": 100  // Reduce from 300
    }
  }
}
```

## 💡 Tips

1. **Start small**: Test with 1000 transactions first
2. **Check data quality**: Run validation before training
3. **Monitor performance**: Track metrics over time
4. **Retrain regularly**: Models degrade without fresh data

## 🎓 Learning Resources

- **Case Study PDF**: `docs/case_study.pdf` - Complete technical analysis
- **Jupyter Notebooks**: `notebooks/` - Interactive exploration
- **Code Examples**: `src/` - Well-documented source code

## 📞 Get Help

- **Issues**: Open a GitHub issue
- **Email**: jfumagalli.work@gmail.com
- **Discussions**: Use GitHub Discussions for questions

---

**Happy Fraud Detecting! 🛡️**
