#%% ======================================================================
# BLOCK — COLD START SEGMENTATION + SEGMENTED SCORING (PIPELINE)
#=========================================================================

from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import (
    roc_auc_score, average_precision_score,
    precision_score, recall_score, f1_score,
    confusion_matrix
)

print("\n" + "=" * 70)
print("BLOCK: COLD START SEGMENTATION + SEGMENTED SCORING")
print("=" * 70)

# ----------------------------------------------------------------------
# 0) Parameters (tune these)
# ----------------------------------------------------------------------
COLD_USER_MIN_TX     = 3
COLD_MERCHANT_MIN_TX = 5
COLD_DEVICE_MIN_TX   = 3

THR_MAIN = 0.50   # threshold for main model (non-cold)
THR_COLD = 0.35   # threshold for cold-start model (usually stricter/lower)

OUT_TABLES = Path("outputs/tables")
OUT_TABLES.mkdir(parents=True, exist_ok=True)

# ----------------------------------------------------------------------
# 1) Define cold-start mask using engineered counts (test_fe)
# ----------------------------------------------------------------------
required_cols = [
    "user_id_tx_count",
    "merchant_id_tx_count",
    "device_id_tx_count",
    "device_missing"
]
missing = [c for c in required_cols if c not in test_fe.columns]
if missing:
    raise ValueError(f"Missing required columns for cold-start segmentation: {missing}")

# ----------------------------------------------------------------------
# 1) Define cold-start mask (risk-based, not volume-only)
# ----------------------------------------------------------------------
cold_mask = (
    (test_fe["user_id_cbk_rate"] < 0.05) &
    (test_fe["merchant_id_cbk_rate"] < 0.05) &
    (test_fe["device_id_cbk_rate"] < 0.05)
)

print(f"Cold-start share in test: {cold_mask.mean()*100:.2f}% "
      f"({cold_mask.sum()} / {len(cold_mask)})")


# ----------------------------------------------------------------------
# 2) Choose models
#    - main_model: your best general model (e.g., XGBoost)
#    - cold_model: lightweight model focusing on non-history features
# ----------------------------------------------------------------------
MAIN_MODEL_NAME = "xgb"     # change if you prefer
COLD_MODEL_NAME = "rf"  # good baseline; can be RF/MLP too

main_model = models[MAIN_MODEL_NAME]
cold_model = models[COLD_MODEL_NAME]

# ----------------------------------------------------------------------
# 3) Define cold-start feature set (avoid heavy reliance on historical rates)
#    Keep it simple and robust.
# ----------------------------------------------------------------------
COLD_FEATURES = [
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

cold_missing_feats = [c for c in COLD_FEATURES if c not in FEATURES]
if cold_missing_feats:
    raise ValueError(f"Cold-start features missing from FEATURES list: {cold_missing_feats}")

# Prepare segment datasets
X_test_main = X_test.loc[~cold_mask, FEATURES].fillna(0)
y_test_main = y_test.loc[~cold_mask]

X_test_cold = X_test.loc[cold_mask, COLD_FEATURES].fillna(0)
y_test_cold = y_test.loc[cold_mask]

# For training cold model, we train on TRAIN split using same cold features
X_train_cold = X_train.loc[:, COLD_FEATURES].fillna(0)

# Train cold model (safe: does not affect main model)
print("\nTraining cold-start model on cold feature set...")
cold_model.fit(X_train_cold, y_train)

# ----------------------------------------------------------------------
# 4) Score each segment
# ----------------------------------------------------------------------
print("\nScoring segments...")

probs_main = main_model.predict_proba(X_test_main)[:, 1] if len(X_test_main) else np.array([])
probs_cold = cold_model.predict_proba(X_test_cold)[:, 1] if len(X_test_cold) else np.array([])

preds_main = (probs_main >= THR_MAIN).astype(int) if len(probs_main) else np.array([], dtype=int)
preds_cold = (probs_cold >= THR_COLD).astype(int) if len(probs_cold) else np.array([], dtype=int)

# Reconstruct full predictions aligned with y_test index
y_pred_segmented = pd.Series(index=y_test.index, dtype=int)

if len(X_test_main):
    y_pred_segmented.loc[~cold_mask] = preds_main
if len(X_test_cold):
    y_pred_segmented.loc[cold_mask] = preds_cold

# Also full probabilities (useful for AUC/PR-AUC)
y_prob_segmented = pd.Series(index=y_test.index, dtype=float)

if len(X_test_main):
    y_prob_segmented.loc[~cold_mask] = probs_main
if len(X_test_cold):
    y_prob_segmented.loc[cold_mask] = probs_cold

# ----------------------------------------------------------------------
# 5) Metrics helper
# ----------------------------------------------------------------------
def compute_metrics(y_true, y_prob, y_pred, label):
    if len(y_true) == 0:
        return None

    auc = roc_auc_score(y_true, y_prob) if len(np.unique(y_true)) > 1 else np.nan
    pr_auc = average_precision_score(y_true, y_prob) if len(np.unique(y_true)) > 1 else np.nan
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

    return {
        "segment": label,
        "n": len(y_true),
        "fraud_rate": float(np.mean(y_true)),
        "AUC": auc,
        "PR_AUC": pr_auc,
        "Precision": prec,
        "Recall": rec,
        "F1": f1,
        "TP": tp, "FN": fn, "FP": fp, "TN": tn
    }

# Segment metrics
m_main = compute_metrics(y_test_main, probs_main, preds_main, "non_cold_start (main_model)")
m_cold = compute_metrics(y_test_cold, probs_cold, preds_cold, "cold_start (cold_model)")
m_all  = compute_metrics(y_test, y_prob_segmented.values, y_pred_segmented.values, "overall (segmented)")

metrics = [m for m in [m_main, m_cold, m_all] if m is not None]
metrics_df = pd.DataFrame(metrics)

print("\nSEGMENTED METRICS:")
print(metrics_df)

metrics_df.to_csv(OUT_TABLES / "segmented_coldstart_metrics.csv", index=False)
print("\n✅ Saved: outputs/tables/segmented_coldstart_metrics.csv")

# ----------------------------------------------------------------------
# 6) Export the actual cold-start and non-cold-start rows for inspection
# ----------------------------------------------------------------------
cold_rows = test_fe.loc[cold_mask].copy()
noncold_rows = test_fe.loc[~cold_mask].copy()

cold_rows["y_true"] = y_test.loc[cold_mask].values
cold_rows["y_prob_cold_model"] = probs_cold if len(probs_cold) else np.nan
cold_rows["y_pred_segmented"] = preds_cold if len(preds_cold) else np.nan

noncold_rows["y_true"] = y_test.loc[~cold_mask].values
noncold_rows["y_prob_main_model"] = probs_main if len(probs_main) else np.nan
noncold_rows["y_pred_segmented"] = preds_main if len(preds_main) else np.nan

cold_rows.to_csv(OUT_TABLES / "cold_start_rows_scored.csv", index=False)
noncold_rows.to_csv(OUT_TABLES / "non_cold_start_rows_scored.csv", index=False)

print("✅ Saved: outputs/tables/cold_start_rows_scored.csv")
print("✅ Saved: outputs/tables/non_cold_start_rows_scored.csv")

# ----------------------------------------------------------------------
# 7) Optional: show how many frauds are missed in each segment
# ----------------------------------------------------------------------
if len(y_test_cold):
    cold_fn = int(((y_test_cold == 1) & (preds_cold == 0)).sum())
    print(f"\nCold-start FN (frauds missed): {cold_fn} / {int((y_test_cold==1).sum())}")

if len(y_test_main):
    main_fn = int(((y_test_main == 1) & (preds_main == 0)).sum())
    print(f"Non-cold-start FN (frauds missed): {main_fn} / {int((y_test_main==1).sum())}")
