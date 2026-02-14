#!/usr/bin/env python3
"""Complete XAU/USD ML Pipeline"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import json
import warnings
from pathlib import Path
from datetime import datetime

from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import RobustScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import (classification_report, confusion_matrix, roc_curve, 
                             auc, accuracy_score, f1_score, precision_score, recall_score)

warnings.filterwarnings('ignore')
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

# Output directories
output_dir = Path('/mnt/okcomputer/output')
output_dir.mkdir(exist_ok=True)
viz_dir = output_dir / 'visualizations'
viz_dir.mkdir(exist_ok=True)

print("="*70)
print("XAU/USD GOLD PRICE PREDICTION - ML PIPELINE")
print("="*70)

# =============================================================================
# 1. LOAD AND PREPROCESS DATA
# =============================================================================
print("\n[1/5] Loading and preprocessing data...")
df = pd.read_csv('/mnt/okcomputer/upload/XAU_15m_data.csv', sep=';')
df['Date'] = pd.to_datetime(df['Date'], format='%Y.%m.%d %H:%M')
df = df.sort_values('Date').reset_index(drop=True)

# Filter 2022-2025
mask = (df['Date'].dt.year >= 2022) & (df['Date'].dt.year <= 2025)
df = df[mask].copy().reset_index(drop=True)
print(f"  Filtered data: {len(df):,} rows ({df['Date'].min().date()} to {df['Date'].max().date()})")

# =============================================================================
# 2. FEATURE ENGINEERING
# =============================================================================
print("\n[2/5] Engineering features...")

# Technical indicators
df['SMA_20'] = df['Close'].rolling(window=20, min_periods=1).mean()
df['SMA_50'] = df['Close'].rolling(window=50, min_periods=1).mean()
df['EMA_20'] = df['Close'].ewm(span=20, adjust=False, min_periods=1).mean()

# RSI
delta = df['Close'].diff()
gain = delta.where(delta > 0, 0).rolling(window=14, min_periods=1).mean()
loss = (-delta.where(delta < 0, 0)).rolling(window=14, min_periods=1).mean()
df['RSI'] = 100 - (100 / (1 + gain / (loss + 1e-10)))

# Volume features
df['Volume_MA'] = df['Volume'].rolling(window=20, min_periods=1).mean()
df['Volume_Ratio'] = df['Volume'] / df['Volume_MA']

# Volatility
df['ATR'] = (df['High'] - df['Low']).rolling(window=14, min_periods=1).mean()
df['ATR_Ratio'] = df['ATR'] / df['Close']

# Bollinger Bands
df['BB_Mid'] = df['Close'].rolling(window=20, min_periods=1).mean()
bb_std = df['Close'].rolling(window=20, min_periods=1).std()
df['BB_Upper'] = df['BB_Mid'] + 2 * bb_std
df['BB_Lower'] = df['BB_Mid'] - 2 * bb_std
df['BB_Position'] = (df['Close'] - df['BB_Lower']) / (df['BB_Upper'] - df['BB_Lower'] + 1e-10)

# MACD
ema_12 = df['Close'].ewm(span=12, adjust=False, min_periods=1).mean()
ema_26 = df['Close'].ewm(span=26, adjust=False, min_periods=1).mean()
df['MACD'] = ema_12 - ema_26
df['MACD_Signal'] = df['MACD'].ewm(span=9, adjust=False, min_periods=1).mean()
df['MACD_Hist'] = df['MACD'] - df['MACD_Signal']

# Momentum
df['Momentum_5'] = df['Close'].pct_change(periods=5)
df['Momentum_10'] = df['Close'].pct_change(periods=10)

# Fibonacci levels
df['Rolling_High'] = df['High'].rolling(window=100, min_periods=1).max()
df['Rolling_Low'] = df['Low'].rolling(window=100, min_periods=1).min()
df['Price_Range'] = df['Rolling_High'] - df['Rolling_Low']
df['Fib_618'] = df['Rolling_High'] - 0.618 * df['Price_Range']
df['Fib_382'] = df['Rolling_High'] - 0.382 * df['Price_Range']
df['Fib_Position'] = (df['Close'] - df['Rolling_Low']) / df['Price_Range']

# SMC signals
df['Higher_High'] = (df['High'] > df['High'].shift(1).rolling(window=10).max()).astype(int)
df['Lower_Low'] = (df['Low'] < df['Low'].shift(1).rolling(window=10).min()).astype(int)
df['BOS_Bullish'] = (df['Close'] > df['High'].shift(1).rolling(window=20).max().shift(1)).astype(int)
df['BOS_Bearish'] = (df['Close'] < df['Low'].shift(1).rolling(window=20).min().shift(1)).astype(int)

# Target: Price goes up > 0.1% in next 6 periods (1.5 hours)
df['Target'] = (df['Close'].shift(-6) / df['Close'] - 1 > 0.001).astype(int)
df = df.dropna(subset=['Target']).reset_index(drop=True)

print(f"  Dataset: {len(df):,} rows")
print(f"  Target distribution: {df['Target'].value_counts().to_dict()}")

# =============================================================================
# 3. PREPARE DATA FOR TRAINING
# =============================================================================
print("\n[3/5] Preparing data for training...")

feature_cols = ['Open', 'High', 'Low', 'Close', 'Volume', 
                'SMA_20', 'SMA_50', 'EMA_20', 'RSI', 
                'Volume_MA', 'Volume_Ratio', 'ATR', 'ATR_Ratio',
                'BB_Mid', 'BB_Position', 'MACD', 'MACD_Hist',
                'Momentum_5', 'Momentum_10',
                'Fib_Position', 'Higher_High', 'Lower_Low', 'BOS_Bullish', 'BOS_Bearish']

X = df[feature_cols].values
y = df['Target'].values

# Handle NaN
X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
)

# Scale
scaler = RobustScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s = scaler.transform(X_test)

print(f"  Features: {len(feature_cols)}")
print(f"  Train: {X_train_s.shape[0]:,}, Test: {X_test_s.shape[0]:,}")

# =============================================================================
# 4. TRAIN MODELS
# =============================================================================
print("\n[4/5] Training models...")

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
results = {}
models_dict = {}

# Model configurations
model_configs = {
    'Random Forest': RandomForestClassifier(
        n_estimators=150, max_depth=15, 
        class_weight='balanced', random_state=RANDOM_STATE, n_jobs=-1
    ),
    'Gradient Boosting': GradientBoostingClassifier(
        n_estimators=100, max_depth=4, learning_rate=0.1, random_state=RANDOM_STATE
    ),
    'Logistic Regression': LogisticRegression(
        C=1, max_iter=1000, class_weight='balanced', random_state=RANDOM_STATE
    ),
    'SVM': SVC(
        C=1, kernel='rbf', probability=True, class_weight='balanced', random_state=RANDOM_STATE
    )
}

for name, model in model_configs.items():
    print(f"  Training {name}...")
    model.fit(X_train_s, y_train)
    y_pred = model.predict(X_test_s)
    y_prob = model.predict_proba(X_test_s)[:, 1] if hasattr(model, 'predict_proba') else None
    
    cv_scores = cross_val_score(model, X_train_s, y_train, cv=cv, scoring='f1', n_jobs=-1)
    
    models_dict[name] = model
    results[name] = {
        'accuracy': accuracy_score(y_test, y_pred),
        'f1_score': f1_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred, zero_division=0),
        'recall': recall_score(y_test, y_pred, zero_division=0),
        'cv_f1_mean': cv_scores.mean(),
        'cv_f1_std': cv_scores.std(),
        'y_pred': y_pred,
        'y_prob': y_prob,
        'report': classification_report(y_test, y_pred, output_dict=True)
    }
    print(f"    Accuracy: {results[name]['accuracy']:.4f}, F1: {results[name]['f1_score']:.4f}")

# =============================================================================
# 5. SELECT BEST MODEL
# =============================================================================
print("\n[5/5] Selecting best model...")

# Sort by F1, then accuracy, then CV stability
sorted_models = sorted(
    results.items(),
    key=lambda x: (x[1]['f1_score'], x[1]['accuracy'], -x[1]['cv_f1_std']),
    reverse=True
)

best_model_name = sorted_models[0][0]
best_model = models_dict[best_model_name]

print(f"\n  Best Model: {best_model_name}")
print(f"  Accuracy: {results[best_model_name]['accuracy']:.2%}")
print(f"  F1-Score: {results[best_model_name]['f1_score']:.4f}")
print(f"  CV F1: {results[best_model_name]['cv_f1_mean']:.4f} (+/- {results[best_model_name]['cv_f1_std']:.4f})")

# =============================================================================
# 6. GENERATE VISUALIZATIONS
# =============================================================================
print("\nGenerating visualizations...")

# Model comparison
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

models = list(results.keys())
accuracies = [results[m]['accuracy'] for m in models]
f1_scores = [results[m]['f1_score'] for m in models]

x = np.arange(len(models))
width = 0.35

axes[0].bar(x - width/2, accuracies, width, label='Accuracy', color='steelblue')
axes[0].bar(x + width/2, f1_scores, width, label='F1-Score', color='coral')
axes[0].set_ylabel('Score')
axes[0].set_title('Model Comparison: Accuracy vs F1-Score')
axes[0].set_xticks(x)
axes[0].set_xticklabels(models, rotation=45, ha='right')
axes[0].legend()
axes[0].axhline(y=0.65, color='green', linestyle='--', alpha=0.7)
axes[0].axhline(y=0.70, color='green', linestyle='--', alpha=0.7)
axes[0].set_ylim(0, 1)

# ROC Curves
for name, metrics in results.items():
    if metrics['y_prob'] is not None:
        fpr, tpr, _ = roc_curve(y_test, metrics['y_prob'])
        roc_auc = auc(fpr, tpr)
        axes[1].plot(fpr, tpr, label=f'{name} (AUC = {roc_auc:.3f})')

axes[1].plot([0, 1], [0, 1], 'k--', label='Random')
axes[1].set_xlabel('False Positive Rate')
axes[1].set_ylabel('True Positive Rate')
axes[1].set_title('ROC Curves')
axes[1].legend(loc='lower right')

plt.tight_layout()
plt.savefig(viz_dir / 'model_comparison.png', dpi=150, bbox_inches='tight')
plt.close()

# Confusion matrices
fig, axes = plt.subplots(2, 2, figsize=(12, 10))
axes = axes.flatten()

for idx, (name, metrics) in enumerate(results.items()):
    cm = confusion_matrix(y_test, metrics['y_pred'])
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[idx],
                xticklabels=['Down', 'Up'], yticklabels=['Down', 'Up'])
    axes[idx].set_title(f'{name}\nAccuracy: {metrics["accuracy"]:.3f}')
    axes[idx].set_xlabel('Predicted')
    axes[idx].set_ylabel('Actual')

plt.tight_layout()
plt.savefig(viz_dir / 'confusion_matrices.png', dpi=150, bbox_inches='tight')
plt.close()

# Feature importance for Random Forest
if 'Random Forest' in models_dict:
    rf_model = models_dict['Random Forest']
    importances = rf_model.feature_importances_
    indices = np.argsort(importances)[-15:]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(range(15), importances[indices], color='steelblue')
    ax.set_yticks(range(15))
    ax.set_yticklabels([feature_cols[i] for i in indices])
    ax.set_xlabel('Importance')
    ax.set_title('Random Forest - Top 15 Feature Importances')
    plt.tight_layout()
    plt.savefig(viz_dir / 'feature_importance.png', dpi=150, bbox_inches='tight')
    plt.close()

print("  Visualizations saved!")

# =============================================================================
# 7. SAVE RESULTS
# =============================================================================
print("\nSaving results...")

# Save best model
with open(output_dir / 'best_model.pkl', 'wb') as f:
    pickle.dump({
        'model': best_model,
        'scaler': scaler,
        'feature_names': feature_cols,
        'model_name': best_model_name
    }, f)

# Save results CSV
results_df = pd.DataFrame([
    {
        'Model': name,
        'Accuracy': metrics['accuracy'],
        'F1_Score': metrics['f1_score'],
        'Precision': metrics['precision'],
        'Recall': metrics['recall'],
        'CV_F1_Mean': metrics['cv_f1_mean'],
        'CV_F1_Std': metrics['cv_f1_std']
    }
    for name, metrics in results.items()
])
results_df = results_df.sort_values('F1_Score', ascending=False)
results_df.to_csv(output_dir / 'model_evaluation_results.csv', index=False)

# Save detailed results
with open(output_dir / 'detailed_results.json', 'w') as f:
    json.dump({
        'best_model': best_model_name,
        'best_accuracy': results[best_model_name]['accuracy'],
        'best_f1': results[best_model_name]['f1_score'],
        'all_results': {k: {kk: float(vv) if isinstance(vv, (np.floating, np.integer)) else vv 
                           for kk, vv in v.items() if kk not in ['y_pred', 'y_prob', 'report']}
                       for k, v in results.items()}
    }, f, indent=2)

print("  Results saved!")

# =============================================================================
# 8. PRINT FINAL REPORT
# =============================================================================
print("\n" + "="*70)
print("FINAL RESULTS")
print("="*70)
print(f"Best Model: {best_model_name}")
print(f"Accuracy: {results[best_model_name]['accuracy']:.2%}")
print(f"F1-Score: {results[best_model_name]['f1_score']:.4f}")
print("="*70)

print("\nClassification Report - Best Model:")
print(classification_report(y_test, results[best_model_name]['y_pred']))

print("\nModel Comparison:")
print("-"*70)
print(f"{'Model':<20} {'Accuracy':>12} {'F1-Score':>12} {'CV F1':>15}")
print("-"*70)
for name, metrics in sorted(results.items(), key=lambda x: x[1]['f1_score'], reverse=True):
    print(f"{name:<20} {metrics['accuracy']:>12.4f} {metrics['f1_score']:>12.4f} {metrics['cv_f1_mean']:>15.4f}")
print("-"*70)

print("\nPipeline completed successfully!")
print(f"Output directory: {output_dir}")
