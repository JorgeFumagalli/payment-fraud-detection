# -*- coding: utf-8 -*-
"""
====================================================================
🚀 PAYMENT_FRAUD FRAUD DETECTION CASE - v4.0 (ENGLISH)
====================================================================
Explainable ML for chargeback prediction with clean visuals and exports
====================================================================
"""

#%% IMPORTS AND INITIAL SETTINGS
import os
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

import tensorflow as tf
from tensorflow.keras import layers, callbacks, regularizers

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import (
    roc_curve, precision_recall_curve, roc_auc_score,
    average_precision_score, confusion_matrix
)

from xgboost import XGBClassifier
import xgboost as xgb

from skopt import BayesSearchCV
from skopt.space import Integer

import shap

warnings.filterwarnings('ignore')
plt.style.use('seaborn-v0_8-darkgrid')

RANDOM_STATE = 42

# Output folders
OUT = Path("outputs")
DIRS = {
    "eda": OUT / "eda",
    "conf": OUT / "confusion_matrices",
    "curves": OUT / "curves",
    "shap": OUT / "shap",
}
for d in DIRS.values():
    d.mkdir(parents=True, exist_ok=True)

FEATURES = [
    'transaction_amount', 'hour', 'day_of_week',
    'merchant_id_cbk_rate', 'merchant_id_tx_count', 'merchant_id_avg_amount',
    'device_id_cbk_rate', 'device_id_tx_count', 'device_id_avg_amount',
    'user_id_cbk_rate', 'user_id_tx_count', 'user_id_avg_amount',
    'is_night', 'is_business_hour', 'is_high_value', 'device_missing',
    'amt_x_cbk_merchant', 'amt_x_cbk_device', 'amt_x_cbk_user',
    'user_last_tx_diff', 'user_tx_24h', 'user_device_div', 'user_merchant_div',
    'is_weekend', 'is_early_morning'
]

print("✅ Libraries loaded.")
print(f"Random State: {RANDOM_STATE}")

#%% BLOCK 1 — DATA LOADING & QUICK EDA
print("=" * 70)
print("BLOCK 1: DATA LOADING & QUICK EDA")
print("=" * 70)

# ► If needed, change the file name below
df = pd.read_excel('df.xlsx')

print(f"Dataset shape: {df.shape}")
print("\nDtypes:\n", df.dtypes)
print("\nMissing values:\n", df.isnull().sum())
print("\nTarget distribution:\n", df['has_cbk'].value_counts())
print(f"Fraud rate: {df['has_cbk'].mean()*100:.2f}%")

# Basic cleaning / coercions
df['device_missing'] = df['device_id'].isna().astype(int)
df['device_id'] = df['device_id'].fillna(-1)
df['transaction_date'] = pd.to_datetime(df['transaction_date'], errors='coerce')
df['has_cbk'] = df['has_cbk'].astype(int)

# Temporal basics
df['hour'] = df['transaction_date'].dt.hour
df['day_of_week'] = df['transaction_date'].dt.dayofweek

# EDA plots (saved)
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
sns.histplot(df['transaction_amount'], bins=50, kde=True, color='skyblue', ax=axes[0])
axes[0].set_title('Transaction Amount Distribution', fontsize=12, fontweight='bold')
axes[0].set_xlabel('Transaction Amount')
sns.boxplot(x=df['transaction_amount'], color='lightcoral', ax=axes[1])
axes[1].set_title('Transaction Amount Boxplot', fontsize=12, fontweight='bold')
axes[1].set_xlabel('Transaction Amount')
plt.tight_layout()
plt.savefig(DIRS["eda"] / "eda_transaction_amount.png", dpi=300)
plt.close()

# Train/test split (simple holdout, stratification preserved by randomness)
np.random.seed(RANDOM_STATE)
mask = np.random.rand(len(df)) < 0.8
train_df = df[mask].reset_index(drop=True)
test_df = df[~mask].reset_index(drop=True)

print(f"\n✅ Train: {train_df.shape}, Test: {test_df.shape}")
print(f"Fraud rate (train): {train_df['has_cbk'].mean()*100:.2f}%")
print(f"Fraud rate (test): {test_df['has_cbk'].mean()*100:.2f}%")

#%% BLOCK 2 — ADVANCED FEATURE ENGINEERING
print("\n" + "=" * 70)
print("BLOCK 2: ADVANCED FEATURE ENGINEERING")
print("=" * 70)

def make_entity_stats(train, key):
    """Aggregated stats per entity (rate, volume, avg amount)."""
    g = train.groupby(key).agg(
        cbk_rate=('has_cbk', 'mean'),
        tx_count=('has_cbk', 'size'),
        avg_amount=('transaction_amount', 'mean')
    ).reset_index()
    g.columns = [key, f'{key}_cbk_rate', f'{key}_tx_count', f'{key}_avg_amount']
    return g

def merge_stats(base, stats, key, global_rate, global_amt):
    """Left-join + NaN backfill with global priors."""
    out = base.merge(stats, on=key, how='left')
    out[f'{key}_cbk_rate'] = out[f'{key}_cbk_rate'].fillna(global_rate)
    out[f'{key}_tx_count'] = out[f'{key}_tx_count'].fillna(0)
    out[f'{key}_avg_amount'] = out[f'{key}_avg_amount'].fillna(global_amt)
    return out

def count_tx_last_24h(df_sorted):
    """For each user, count transactions in the last 24h (O(n^2) naive)."""
    df_sorted = df_sorted.sort_values(['user_id', 'transaction_date'])
    tx_24h = []
    for uid, g in df_sorted.groupby('user_id'):
        times = g['transaction_date'].values
        counts = []
        for i in range(len(times)):
            ref_time = times[i]
            counts.append(np.sum((ref_time - times) <= np.timedelta64(24, 'h')) - 1)
        tx_24h.extend(counts)
    return tx_24h

print("Computing entity-level statistics...")
merchant_stats = make_entity_stats(train_df, 'merchant_id')
device_stats   = make_entity_stats(train_df, 'device_id')
user_stats     = make_entity_stats(train_df, 'user_id')

global_rate = train_df['has_cbk'].mean()
global_amt  = train_df['transaction_amount'].mean()
print(f"Global chargeback rate: {global_rate*100:.2f}%")
print(f"Global average amount: ${global_amt:.2f}")

print("Merging stats into train/test...")
train_fe = merge_stats(train_df.copy(), merchant_stats, 'merchant_id', global_rate, global_amt)
train_fe = merge_stats(train_fe,         device_stats,   'device_id',   global_rate, global_amt)
train_fe = merge_stats(train_fe,         user_stats,     'user_id',     global_rate, global_amt)

test_fe  = merge_stats(test_df.copy(),  merchant_stats, 'merchant_id', global_rate, global_amt)
test_fe  = merge_stats(test_fe,         device_stats,   'device_id',   global_rate, global_amt)
test_fe  = merge_stats(test_fe,         user_stats,     'user_id',     global_rate, global_amt)

print("Creating temporal flags and cross-features...")
high_thr = train_fe['transaction_amount'].quantile(0.95)
for d in (train_fe, test_fe):
    d['is_night']         = ((d['hour'] < 7) | (d['hour'] > 22)).astype(int)
    d['is_business_hour'] = ((d['hour'] >= 8) & (d['hour'] <= 18)).astype(int)
    d['is_high_value']    = (d['transaction_amount'] >= high_thr).astype(int)
    d['amt_x_cbk_merchant'] = d['transaction_amount'] * d['merchant_id_cbk_rate']
    d['amt_x_cbk_device']   = d['transaction_amount'] * d['device_id_cbk_rate']
    d['amt_x_cbk_user']     = d['transaction_amount'] * d['user_id_cbk_rate']

print("Computing behavioral and diversity features...")
train_fe = train_fe.sort_values(['user_id','transaction_date']).reset_index(drop=True)
test_fe  = test_fe.sort_values(['user_id','transaction_date']).reset_index(drop=True)

train_fe['user_last_tx_diff'] = train_fe.groupby('user_id')['transaction_date'].diff().dt.total_seconds().fillna(999999)
test_fe['user_last_tx_diff']  =  test_fe.groupby('user_id')['transaction_date'].diff().dt.total_seconds().fillna(999999)

train_fe['user_tx_24h'] = count_tx_last_24h(train_fe)
test_fe['user_tx_24h']  = count_tx_last_24h(test_fe)

user_dev = train_fe.groupby('user_id')['device_id'].nunique().reset_index(name='user_device_div')
user_mer = train_fe.groupby('user_id')['merchant_id'].nunique().reset_index(name='user_merchant_div')

for name, d in [('train_fe', train_fe), ('test_fe', test_fe)]:
    d = d.merge(user_dev, on='user_id', how='left')
    d = d.merge(user_mer, on='user_id', how='left')
    d['is_weekend']       = d['day_of_week'].isin([5, 6]).astype(int)
    d['is_early_morning'] = ((d['hour'] >= 0) & (d['hour'] <= 5)).astype(int)
    if name == 'train_fe':
        train_fe = d
    else:
        test_fe = d

print(f"✅ Feature engineering complete. Total features engineered: {len(FEATURES)}")

#%% BLOCK 3 — MODEL TRAINING
print("\n" + "=" * 70)
print("BLOCK 3: MODEL TRAINING")
print("=" * 70)

TARGET = 'has_cbk'
X_train, y_train = train_fe[FEATURES].fillna(0), train_fe[TARGET]
X_test,  y_test  =  test_fe[FEATURES].fillna(0),  test_fe[TARGET]

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)

results, models, model_probs = {}, {}, {}

# 1) Logistic Regression
print("\n[1/5] Training Logistic Regression…")
lr = LogisticRegression(max_iter=1000, class_weight='balanced', random_state=RANDOM_STATE)
lr.fit(X_train_scaled, y_train)
lr_prob = lr.predict_proba(X_test_scaled)[:, 1]
results['Logistic Regression'] = {'AUC': roc_auc_score(y_test, lr_prob)}
model_probs['LogReg'] = lr_prob
models['lr'] = lr
print(f"   ✓ AUC={results['Logistic Regression']['AUC']:.4f}")

# 2) Random Forest (Bayesian Optimization)
print("[2/5] Training Random Forest (BayesSearchCV)…")
search_spaces = {'n_estimators': Integer(200, 800), 'max_depth': Integer(4, 20)}
bayes_rf = BayesSearchCV(
    RandomForestClassifier(class_weight='balanced', n_jobs=-1, random_state=RANDOM_STATE),
    search_spaces=search_spaces, n_iter=8, scoring='roc_auc', cv=3,
    random_state=RANDOM_STATE, verbose=0
)
bayes_rf.fit(X_train, y_train)
rf_best = bayes_rf.best_estimator_
rf_prob = rf_best.predict_proba(X_test)[:, 1]
results['Random Forest'] = {'AUC': roc_auc_score(y_test, rf_prob)}
model_probs['RF'] = rf_prob
models['rf'] = rf_best
print(f"   ✓ AUC={results['Random Forest']['AUC']:.4f}, Params={bayes_rf.best_params_}")

# 3) XGBoost
print("[3/5] Training XGBoost…")
xgb_model = XGBClassifier(
    n_estimators=500, max_depth=6, learning_rate=0.05,
    subsample=0.8, colsample_bytree=0.8,
    scale_pos_weight=(y_train.value_counts()[0] / y_train.value_counts()[1]),
    random_state=RANDOM_STATE, eval_metric='auc', verbosity=0
)
xgb_model.fit(X_train, y_train)
xgb_prob = xgb_model.predict_proba(X_test)[:, 1]
results['XGBoost'] = {'AUC': roc_auc_score(y_test, xgb_prob)}
model_probs['XGB'] = xgb_prob
models['xgb'] = xgb_model
print(f"   ✓ AUC={results['XGBoost']['AUC']:.4f}")

# 4) MLP
print("[4/5] Training MLP Classifier…")
mlp = MLPClassifier(hidden_layer_sizes=(128, 64), activation='relu', 
                    solver='adam', alpha=1e-4, batch_size=64, 
                    learning_rate_init=1e-3,   max_iter=500, 
                    early_stopping=True, validation_fraction=0.15,
                    n_iter_no_change=20, tol=1e-4, random_state=RANDOM_STATE)
mlp.fit(X_train_scaled, y_train)
mlp_prob = mlp.predict_proba(X_test_scaled)[:, 1]
results['MLP'] = {'AUC': roc_auc_score(y_test, mlp_prob)}
model_probs['MLP'] = mlp_prob
models['mlp'] = mlp
print(f"   ✓ AUC={results['MLP']['AUC']:.4f}")

print("[5/5] Training LSTM Classifier…")

X_train_seq = X_train_scaled.reshape(X_train_scaled.shape[0], 1, X_train_scaled.shape[1])
X_test_seq  = X_test_scaled.reshape(X_test_scaled.shape[0], 1, X_test_scaled.shape[1])

tf.keras.utils.set_random_seed(RANDOM_STATE)

model = tf.keras.Sequential([
    layers.Input(shape=(1, X_train_scaled.shape[1])),
    layers.LSTM(
        64,
        return_sequences=False,
        dropout=0.2,
        recurrent_dropout=0.0,
        kernel_regularizer=regularizers.l2(1e-5)
    ),
    layers.BatchNormalization(),
    layers.Dense(64, activation="relu", kernel_regularizer=regularizers.l2(1e-5)),
    layers.Dropout(0.3),
    layers.Dense(1, activation="sigmoid")
])

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
    loss="binary_crossentropy",
    metrics=[tf.keras.metrics.AUC(name="auc")]
)

cbs = [
    callbacks.EarlyStopping(monitor="val_auc", mode="max", patience=15, restore_best_weights=True),
    callbacks.ReduceLROnPlateau(monitor="val_auc", mode="max", patience=6, factor=0.5, min_lr=1e-5)
]

history = model.fit(
    X_train_seq, y_train,
    validation_split=0.15,
    epochs=120,
    batch_size=64,
    callbacks=cbs,
    verbose=0
)

lstm_prob = model.predict(X_test_seq, verbose=0).ravel()

results['LSTM'] = {'AUC': roc_auc_score(y_test, lstm_prob)}
model_probs['LSTM'] = lstm_prob
models['lstm'] = model

print(f"   ✓ AUC={results['LSTM']['AUC']:.4f}")

print("\n✅ All models trained successfully.")

#%% BLOCK 4 — ROC & PRECISION–RECALL CURVES (SAVE)
print("\n" + "=" * 70)
print("BLOCK 4: PERFORMANCE CURVES (ROC & PR)")
print("=" * 70)

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# ROC
for name, prob in model_probs.items():
    fpr, tpr, _ = roc_curve(y_test, prob)
    auc_val = roc_auc_score(y_test, prob)
    axes[0].plot(fpr, tpr, lw=2, label=f'{name} (AUC={auc_val:.3f})')
axes[0].plot([0,1], [0,1], '--', color='gray', label='Random')
axes[0].set_title('ROC Curves — Model Comparison')
axes[0].set_xlabel('False Positive Rate')
axes[0].set_ylabel('True Positive Rate')
axes[0].legend(fontsize=9)

# PR
for name, prob in model_probs.items():
    prec, rec, _ = precision_recall_curve(y_test, prob)
    ap = average_precision_score(y_test, prob)
    axes[1].plot(rec, prec, lw=2, label=f'{name} (AP={ap:.3f})')
axes[1].set_title('Precision–Recall Curves')
axes[1].set_xlabel('Recall')
axes[1].set_ylabel('Precision')
axes[1].legend(fontsize=9)

plt.tight_layout()
plt.savefig(DIRS["curves"] / "roc_pr_curves.png", dpi=300)
plt.close()
print("✅ ROC/PR curves saved.")

#%% BLOCK 5 — CONFUSION MATRICES (SAVE ONE PER MODEL)
print("\n" + "=" * 70)
print("BLOCK 5: CONFUSION MATRICES")
print("=" * 70)

model_list = [
    ('Logistic Regression', model_probs['LogReg']),
    ('Random Forest',        model_probs['RF']),
    ('XGBoost',              model_probs['XGB']),
    ('MLP',                  model_probs['MLP']),
    ('LSTM',                model_probs['LSTM'])
]

for name, probs in model_list:
    preds = (probs >= 0.5).astype(int)
    cm = confusion_matrix(y_test, preds)
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
                xticklabels=['Normal', 'Fraud'], yticklabels=['Normal', 'Fraud'])
    plt.title(f'{name} — Confusion Matrix')
    plt.xlabel('Predicted'); plt.ylabel('Actual')
    plt.tight_layout()
    plt.savefig(DIRS["conf"] / f"{name.replace(' ', '_')}_cm.png", dpi=300)
    plt.close()

print("✅ Confusion matrices saved to /outputs/confusion_matrices.")

#%% BLOCK 6 — COST SENSITIVITY & SWITCH POINTS
print("\n" + "=" * 70)
print("BLOCK 6: COST SENSITIVITY ANALYSIS & MODEL SWITCH POINTS")
print("=" * 70)

def calc_cost(cm, r):
    """Normalized total cost with cost ratio r = Cost(FN) / Cost(FP)."""
    tn, fp, fn, tp = cm.ravel()
    return fp + r * fn

# Build confusion matrices at threshold 0.5
cms = {name: confusion_matrix(y_test, (probs >= 0.5).astype(int))
       for name, probs in model_list}

# Cost curves
r_values = np.linspace(0, 10, 500)
cost_curves = {name: [calc_cost(cm, r) for r in r_values] for name, cm in cms.items()}

# Plot curves
plt.figure(figsize=(12, 7))
palette = {'Logistic Regression': '#1f77b4', 'Random Forest': '#ff7f0e',
           'XGBoost': '#2ca02c', 'MLP': '#d62728', 'LSTM':  '#993399'}
for name, costs in cost_curves.items():
    plt.plot(r_values, costs, label=name, lw=2.2, color=palette[name], alpha=0.9)
plt.axvline(0.74, color='k', ls=':', lw=1)
plt.axvline(1.32, color='k', ls=':', lw=1)
plt.axvline(1.98, color='k', ls=':', lw=1)
plt.axvline(2.99, color='k', ls=':', lw=1)
plt.text(1.02, plt.ylim()[1]*0.95, 'r=1 (equal costs)', fontsize=9)
plt.xlabel("r = Cost(False Negative) / Cost(False Positive)")
plt.ylabel("Normalized Total Cost")
plt.title("Cost Sensitivity — Optimal Model by Cost Ratio")
plt.legend(fontsize=10, loc='upper left')
plt.tight_layout()
plt.savefig(DIRS["curves"] / "cost_sensitivity.png", dpi=300)
plt.close()
print("✅ Cost sensitivity plot saved.")

# Find switching points (curve intersections)
print("\nSwitching points (where one model becomes cheaper than another):\n")
names = list(cost_curves.keys())
switch_points = []
for i in range(len(names)):
    for j in range(i+1, len(names)):
        a, b = names[i], names[j]
        diff = np.array(cost_curves[a]) - np.array(cost_curves[b])
        idx = np.where(np.diff(np.sign(diff)))[0]
        for k in idx:
            switch_points.append((r_values[k], a, b))
switch_points = sorted(switch_points, key=lambda x: x[0])
if switch_points:
    for r, m1, m2 in switch_points:
        print(f" • r ≈ {r:.2f}: {m1} ⇄ {m2}")
else:
    print(" • No intersections found — one model dominates across the range.")

# Quick guidance at representative r values
probe_r = [0.5, 1.0, 2.0, 5.0, 10.0]
print("\nRecommendations by cost ratio:")
for rv in probe_r:
    costs_at_r = {name: calc_cost(cms[name], rv) for name in names}
    best = min(costs_at_r.items(), key=lambda kv: kv[1])
    context = ("Blocked customer cost > missed fraud" if rv < 1 else
               "Equal costs (FP = FN)" if rv == 1 else
               f"Missed fraud costs {int(rv)}x more than blocked customer")
    print(f" r = {rv:4.1f} ({context}) → Best: {best[0]} (Cost={best[1]:.1f})")

#%% BLOCK 7 — FEATURE IMPORTANCE (XGBOOST, TOP-10)
print("\n" + "=" * 70)
print("BLOCK 7: FEATURE IMPORTANCE (XGBOOST, TOP-10)")
print("=" * 70)

plt.figure(figsize=(10, 7))
xgb.plot_importance(xgb_model.get_booster(), importance_type='gain',
                    max_num_features=10, height=0.6)
plt.title('Top-10 Features by Gain (XGBoost)')
plt.xlabel('Information Gain')
plt.tight_layout()
plt.savefig(DIRS["shap"] / "xgb_top10_gain.png", dpi=300)
plt.close()
print("✅ XGBoost top-10 gain plot saved.")

# Snapshot tree for quick rules
clf = DecisionTreeClassifier(max_depth=3, criterion='gini',
                             class_weight='balanced', random_state=RANDOM_STATE)
clf.fit(X_train, y_train)
plt.figure(figsize=(18, 11))
plot_tree(clf, filled=True, feature_names=FEATURES,
          class_names=["Normal", "Fraud"], rounded=True,
          proportion=True, fontsize=10, impurity=False)
plt.title("Representative Decision Tree (max_depth=3)")
plt.tight_layout()
plt.savefig(DIRS["shap"] / "decision_tree_snapshot.png", dpi=300)
plt.close()
print("✅ Decision tree snapshot saved.")

#%% BLOCK 8 — EXPLAINABILITY WITH SHAP (TOP-10)
print("\n" + "=" * 70)
print("BLOCK 8: EXPLAINABILITY WITH SHAP (TOP-10)")
print("=" * 70)

print("Computing SHAP values using Random Forest (stable path)…")
explainer = shap.TreeExplainer(rf_best)
sample_n = min(300, len(X_test))
X_sample = X_test.iloc[:sample_n].copy()
sv_raw = explainer.shap_values(X_sample)

# Normalize output shape across SHAP versions
if isinstance(sv_raw, list):              # [class0, class1]
    shap_values = sv_raw[1]
elif hasattr(sv_raw, "shape") and len(sv_raw.shape) == 3:  # (n, features, classes)
    shap_values = sv_raw[:, :, 1]
else:
    shap_values = sv_raw

# 1) Global bar (Top-10)
plt.figure(figsize=(10, 7))
shap.summary_plot(shap_values, X_sample, plot_type='bar', max_display=10, show=False)
plt.title('Global Feature Importance (SHAP) — Top-10')
plt.tight_layout()
plt.savefig(DIRS["shap"] / "shap_bar_top10.png", dpi=300)
plt.close()

# 2) Beeswarm (Top-10)
plt.figure(figsize=(12, 7))
shap.summary_plot(shap_values, X_sample, max_display=10, show=False)
plt.title('SHAP Beeswarm — Top-10 Features')
plt.tight_layout()
plt.savefig(DIRS["shap"] / "shap_beeswarm_top10.png", dpi=300)
plt.close()

# 3) Directional mean impact (Top-10)
sv_df = pd.DataFrame(shap_values, columns=FEATURES[:shap_values.shape[1]])
impact_means = sv_df.mean(axis=0).sort_values(ascending=True).tail(10)
plt.figure(figsize=(10, 6))
colors = ['#2ca02c' if v < 0 else '#d62728' for v in impact_means.values]
plt.barh(impact_means.index, impact_means.values, color=colors, alpha=0.85)
plt.title('Directional Mean Impact (SHAP) — Top-10')
plt.xlabel('Average Contribution to Fraud Probability')
plt.axvline(x=0, color='black', linestyle='-', linewidth=0.8)
plt.grid(axis='x', alpha=0.3)
plt.tight_layout()
plt.savefig(DIRS["shap"] / "shap_directional_top10.png", dpi=300)
plt.close()

print("✅ SHAP plots saved to /outputs/shap")

#%% BLOCK 9 — FINAL SUMMARY (PRINT-ONLY, FOR REPORT WRITING)
print("\n" + "=" * 70)
print("BLOCK 9: FINAL SUMMARY")
print("=" * 70)

print("\nModel AUC on test set:")
for k, v in results.items():
    print(f" • {k:20s} AUC={v['AUC']:.4f}")

best_model = max(results.items(), key=lambda kv: kv[1]['AUC'])[0]
print(f"\n✅ Best model by AUC: {best_model}")
print("\nNext steps recommended:")
print("  1) Deploy the best model with a business-calibrated threshold;")
print("  2) Monitor cost ratio r and switch model if crossing points are reached;")
print("  3) Retrain quarterly and track SHAP drift on the Top-10 features.")
