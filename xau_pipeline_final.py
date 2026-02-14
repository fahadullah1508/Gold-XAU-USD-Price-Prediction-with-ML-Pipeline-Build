#!/usr/bin/env python3
"""
XAU/USD Gold Price Prediction - Complete ML Pipeline
"""

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
output_dir = Path('output')
output_dir.mkdir(exist_ok=True)
viz_dir = output_dir / 'visualizations'
viz_dir.mkdir(exist_ok=True)

print("="*70)
print("XAU/USD GOLD PRICE PREDICTION - ML PIPELINE")
print("="*70)

# =============================================================================
# 1. LOAD DATA
# =============================================================================
print("\n[1/5] Loading data...")
df = pd.read_csv('XAU_15m_data.csv', sep=';')
df['Date'] = pd.to_datetime(df['Date'], format='%Y.%m.%d %H:%M')
df = df.sort_values('Date').reset_index(drop=True)
mask = (df['Date'].dt.year >= 2022) & (df['Date'].dt.year <= 2025)
df = df[mask].copy().reset_index(drop=True)
print(f"  Loaded: {len(df):,} rows ({df['Date'].min().date()} to {df['Date'].max().date()})")

# =============================================================================
# 2. FEATURE ENGINEERING
# =============================================================================
print("\n[2/5] Engineering features...")

# Moving averages
for period in [5, 10, 20, 50]:
    df[f'SMA_{period}'] = df['Close'].rolling(window=period, min_periods=1).mean()
    df[f'EMA_{period}'] = df['Close'].ewm(span=period, adjust=False, min_periods=1).mean()

# RSI
delta = df['Close'].diff()
gain = delta.where(delta > 0, 0).rolling(window=14, min_periods=1).mean()
loss = (-delta.where(delta < 0, 0)).rolling(window=14, min_periods=1).mean()
df['RSI'] = 100 - (100 / (1 + gain / (loss + 1e-10)))

# MACD
ema_12 = df['Close'].ewm(span=12, adjust=False, min_periods=1).mean()
ema_26 = df['Close'].ewm(span=26, adjust=False, min_periods=1).mean()
df['MACD'] = ema_12 - ema_26
df['MACD_Signal'] = df['MACD'].ewm(span=9, adjust=False, min_periods=1).mean()
df['MACD_Hist'] = df['MACD'] - df['MACD_Signal']

# Bollinger Bands
df['BB_Mid'] = df['Close'].rolling(window=20, min_periods=1).mean()
bb_std = df['Close'].rolling(window=20, min_periods=1).std()
df['BB_Upper'] = df['BB_Mid'] + 2 * bb_std
df['BB_Lower'] = df['BB_Mid'] - 2 * bb_std
df['BB_Width'] = (df['BB_Upper'] - df['BB_Lower']) / df['BB_Mid']
df['BB_Position'] = (df['Close'] - df['BB_Lower']) / (df['BB_Upper'] - df['BB_Lower'] + 1e-10)

# ATR
tr1 = df['High'] - df['Low']
tr2 = abs(df['High'] - df['Close'].shift(1))
tr3 = abs(df['Low'] - df['Close'].shift(1))
df['ATR'] = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1).rolling(window=14, min_periods=1).mean()
df['ATR_Ratio'] = df['ATR'] / df['Close']

# Volume
df['Volume_MA'] = df['Volume'].rolling(window=20, min_periods=1).mean()
df['Volume_Ratio'] = df['Volume'] / df['Volume_MA']

# Momentum
df['Momentum_5'] = df['Close'].pct_change(periods=5)
df['Momentum_10'] = df['Close'].pct_change(periods=10)
df['Momentum_20'] = df['Close'].pct_change(periods=20)

# Price action
df['Body'] = df['Close'] - df['Open']
df['Upper_Shadow'] = df['High'] - df[['Open', 'Close']].max(axis=1)
df['Lower_Shadow'] = df[['Open', 'Close']].min(axis=1) - df['Low']

# Fibonacci
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

# Target
df['Target'] = (df['Close'].shift(-6) / df['Close'] - 1 > 0.001).astype(int)
df = df.dropna(subset=['Target']).reset_index(drop=True)

feature_cols = ['Open', 'High', 'Low', 'Close', 'Volume', 
                'SMA_5', 'SMA_10', 'SMA_20', 'SMA_50', 'EMA_5', 'EMA_10', 'EMA_20', 'EMA_50',
                'RSI', 'MACD', 'MACD_Signal', 'MACD_Hist',
                'BB_Mid', 'BB_Width', 'BB_Position',
                'ATR', 'ATR_Ratio', 'Volume_MA', 'Volume_Ratio',
                'Momentum_5', 'Momentum_10', 'Momentum_20',
                'Body', 'Upper_Shadow', 'Lower_Shadow',
                'Fib_Position', 'Higher_High', 'Lower_Low', 'BOS_Bullish', 'BOS_Bearish']

print(f"  Features: {len(feature_cols)}")
print(f"  Dataset: {len(df):,} rows")
print(f"  Target: {df['Target'].value_counts().to_dict()}")

# =============================================================================
# 3. PREPARE DATA
# =============================================================================
print("\n[3/5] Preparing data...")

X = df[feature_cols].values
y = df['Target'].values
X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
)

scaler = RobustScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s = scaler.transform(X_test)

print(f"  Train: {X_train_s.shape[0]:,}, Test: {X_test_s.shape[0]:,}")

# =============================================================================
# 4. TRAIN MODELS
# =============================================================================
print("\n[4/5] Training models...")

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
results = {}
models_dict = {}

# Random Forest
print("  Random Forest...")
rf = RandomForestClassifier(n_estimators=200, max_depth=20, min_samples_split=5, 
                            class_weight='balanced', random_state=RANDOM_STATE, n_jobs=-1)
rf.fit(X_train_s, y_train)
y_pred = rf.predict(X_test_s)
y_prob = rf.predict_proba(X_test_s)[:, 1]
cv_scores = cross_val_score(rf, X_train_s, y_train, cv=cv, scoring='f1', n_jobs=-1)
results['Random Forest'] = {
    'accuracy': accuracy_score(y_test, y_pred),
    'f1_score': f1_score(y_test, y_pred),
    'precision': precision_score(y_test, y_pred, zero_division=0),
    'recall': recall_score(y_test, y_pred, zero_division=0),
    'cv_f1_mean': cv_scores.mean(),
    'cv_f1_std': cv_scores.std(),
    'y_pred': y_pred, 'y_prob': y_prob
}
models_dict['Random Forest'] = rf
print(f"    Acc: {results['Random Forest']['accuracy']:.4f}, F1: {results['Random Forest']['f1_score']:.4f}")

# Gradient Boosting
print("  Gradient Boosting...")
gb = GradientBoostingClassifier(n_estimators=150, max_depth=5, learning_rate=0.1, random_state=RANDOM_STATE)
gb.fit(X_train_s, y_train)
y_pred = gb.predict(X_test_s)
y_prob = gb.predict_proba(X_test_s)[:, 1]
cv_scores = cross_val_score(gb, X_train_s, y_train, cv=cv, scoring='f1', n_jobs=-1)
results['Gradient Boosting'] = {
    'accuracy': accuracy_score(y_test, y_pred),
    'f1_score': f1_score(y_test, y_pred),
    'precision': precision_score(y_test, y_pred, zero_division=0),
    'recall': recall_score(y_test, y_pred, zero_division=0),
    'cv_f1_mean': cv_scores.mean(),
    'cv_f1_std': cv_scores.std(),
    'y_pred': y_pred, 'y_prob': y_prob
}
models_dict['Gradient Boosting'] = gb
print(f"    Acc: {results['Gradient Boosting']['accuracy']:.4f}, F1: {results['Gradient Boosting']['f1_score']:.4f}")

# Logistic Regression
print("  Logistic Regression...")
lr = LogisticRegression(C=1, max_iter=1000, class_weight='balanced', random_state=RANDOM_STATE)
lr.fit(X_train_s, y_train)
y_pred = lr.predict(X_test_s)
y_prob = lr.predict_proba(X_test_s)[:, 1]
cv_scores = cross_val_score(lr, X_train_s, y_train, cv=cv, scoring='f1', n_jobs=-1)
results['Logistic Regression'] = {
    'accuracy': accuracy_score(y_test, y_pred),
    'f1_score': f1_score(y_test, y_pred),
    'precision': precision_score(y_test, y_pred, zero_division=0),
    'recall': recall_score(y_test, y_pred, zero_division=0),
    'cv_f1_mean': cv_scores.mean(),
    'cv_f1_std': cv_scores.std(),
    'y_pred': y_pred, 'y_prob': y_prob
}
models_dict['Logistic Regression'] = lr
print(f"    Acc: {results['Logistic Regression']['accuracy']:.4f}, F1: {results['Logistic Regression']['f1_score']:.4f}")

# SVM
print("  SVM...")
svm = SVC(C=1, kernel='rbf', probability=True, class_weight='balanced', random_state=RANDOM_STATE)
svm.fit(X_train_s, y_train)
y_pred = svm.predict(X_test_s)
y_prob = svm.predict_proba(X_test_s)[:, 1]
cv_scores = cross_val_score(svm, X_train_s, y_train, cv=cv, scoring='f1', n_jobs=-1)
results['SVM'] = {
    'accuracy': accuracy_score(y_test, y_pred),
    'f1_score': f1_score(y_test, y_pred),
    'precision': precision_score(y_test, y_pred, zero_division=0),
    'recall': recall_score(y_test, y_pred, zero_division=0),
    'cv_f1_mean': cv_scores.mean(),
    'cv_f1_std': cv_scores.std(),
    'y_pred': y_pred, 'y_prob': y_prob
}
models_dict['SVM'] = svm
print(f"    Acc: {results['SVM']['accuracy']:.4f}, F1: {results['SVM']['f1_score']:.4f}")

# =============================================================================
# 5. SELECT BEST MODEL
# =============================================================================
print("\n[5/5] Selecting best model...")

sorted_models = sorted(results.items(), key=lambda x: (x[1]['f1_score'], x[1]['accuracy'], -x[1]['cv_f1_std']), reverse=True)
best_model_name = sorted_models[0][0]
best_model = models_dict[best_model_name]

print(f"\n  Best Model: {best_model_name}")
print(f"  Accuracy: {results[best_model_name]['accuracy']:.2%}")
print(f"  F1-Score: {results[best_model_name]['f1_score']:.4f}")

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

# Feature importance
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

print("\n" + "="*70)
print("Pipeline completed successfully!")
print(f"Output directory: {output_dir}")
print("="*70)
