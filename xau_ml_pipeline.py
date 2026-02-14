#!/usr/bin/env python3
"""
XAU/USD Gold Price Prediction - Complete ML Pipeline with Enhanced Visualizations
Author: Fahad Ullah
Date: 2026-02-14
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.gridspec as gridspec
import seaborn as sns
from datetime import datetime
import warnings
import logging
import pickle
import json
from pathlib import Path
from typing import Dict, List, Tuple

# ML Libraries
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold, cross_val_score, learning_curve
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import (classification_report, confusion_matrix, roc_curve, 
                             auc, accuracy_score, f1_score, precision_score, recall_score,
                             precision_recall_curve, average_precision_score)
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

# XGBoost
try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    warnings.warn("XGBoost not available, will be skipped")

# PDF Generation
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter, A4
from reportlab.platypus import (SimpleDocTemplate, Table, TableStyle, Paragraph, 
                                Spacer, Image, PageBreak, ListFlowable, ListItem)
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_JUSTIFY

# Suppress warnings
warnings.filterwarnings('ignore')

# Set random seeds for reproducibility
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('xau_ml_pipeline.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Set style for better-looking plots
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")


class XAUPipeline:
    """Complete ML Pipeline for XAU/USD Price Prediction with Enhanced Visualizations"""
    
    def __init__(self, data_path: str, output_dir: str = 'output'):
        self.data_path = data_path
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.viz_dir = self.output_dir / 'visualizations'
        self.viz_dir.mkdir(exist_ok=True)
        self.df = None
        self.df_filtered = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.scaler = None
        self.models = {}
        self.results = {}
        self.best_model = None
        self.best_model_name = None
        self.feature_names = None
        self.feature_importance_df = None
        
    def load_data(self):
        """Load and parse the XAU/USD dataset"""
        logger.info("Loading data from %s", self.data_path)
        self.df = pd.read_csv(self.data_path, sep=';')
        self.df['Date'] = pd.to_datetime(self.df['Date'], format='%Y.%m.%d %H:%M')
        self.df = self.df.sort_values('Date').reset_index(drop=True)
        logger.info("Loaded %d rows spanning %s to %s", 
                   len(self.df), self.df['Date'].min(), self.df['Date'].max())
        return self
    
    def filter_data(self, start_year: int = 2022, end_year: int = 2025):
        """Filter data to specified year range"""
        logger.info("Filtering data for years %d-%d", start_year, end_year)
        mask = (self.df['Date'].dt.year >= start_year) & (self.df['Date'].dt.year <= end_year)
        self.df_filtered = self.df[mask].copy().reset_index(drop=True)
        logger.info("Filtered dataset: %d rows from %s to %s",
                   len(self.df_filtered), 
                   self.df_filtered['Date'].min(), 
                   self.df_filtered['Date'].max())
        return self
    
    def preprocess_data(self):
        """Handle missing values and outliers"""
        logger.info("Preprocessing data...")
        
        # Check for missing values
        missing = self.df_filtered.isnull().sum()
        if missing.sum() > 0:
            logger.info("Handling missing values: %s", missing[missing > 0].to_dict())
            self.df_filtered = self.df_filtered.fillna(method='ffill').fillna(method='bfill')
        
        # Remove price anomalies (prices outside 3 standard deviations)
        for col in ['Open', 'High', 'Low', 'Close']:
            mean = self.df_filtered[col].mean()
            std = self.df_filtered[col].std()
            mask = (self.df_filtered[col] >= mean - 3*std) & (self.df_filtered[col] <= mean + 3*std)
            outliers = (~mask).sum()
            if outliers > 0:
                logger.info("Removing %d outliers from %s", outliers, col)
                self.df_filtered = self.df_filtered[mask].reset_index(drop=True)
        
        # Remove zero volume records
        zero_volume = (self.df_filtered['Volume'] == 0).sum()
        if zero_volume > 0:
            logger.info("Removing %d zero-volume records", zero_volume)
            self.df_filtered = self.df_filtered[self.df_filtered['Volume'] > 0].reset_index(drop=True)
        
        logger.info("After preprocessing: %d rows", len(self.df_filtered))
        return self
    
    def calculate_fibonacci_levels(self, window: int = 100):
        """Calculate Fibonacci retracement levels"""
        logger.info("Calculating Fibonacci retracement levels...")
        
        df = self.df_filtered
        
        # Rolling high and low for Fibonacci calculation
        df['Rolling_High'] = df['High'].rolling(window=window, min_periods=1).max()
        df['Rolling_Low'] = df['Low'].rolling(window=window, min_periods=1).min()
        df['Price_Range'] = df['Rolling_High'] - df['Rolling_Low']
        
        # Fibonacci levels
        fib_ratios = [0.0, 0.236, 0.382, 0.5, 0.618, 0.786, 1.0]
        for ratio in fib_ratios:
            col_name = f'Fib_{int(ratio*1000)}'
            df[col_name] = df['Rolling_High'] - ratio * df['Price_Range']
        
        # Distance from current price to each Fibonacci level
        for ratio in [0.236, 0.382, 0.5, 0.618, 0.786]:
            col_name = f'Fib_{int(ratio*1000)}'
            df[f'Dist_to_{col_name}'] = (df['Close'] - df[col_name]) / df['Price_Range']
        
        # Price position within Fibonacci range (0 = at support, 1 = at resistance)
        df['Fib_Position'] = (df['Close'] - df['Fib_0']) / df['Price_Range']
        
        logger.info("Fibonacci levels calculated")
        return self
    
    def calculate_smc_signals(self):
        """Calculate Smart Money Concepts (SMC) signals"""
        logger.info("Calculating SMC signals (BOS, CHoCH, Liquidity)...")
        
        df = self.df_filtered
        
        # Swing Highs and Lows detection
        window = 5
        df['Swing_High'] = (df['High'] == df['High'].rolling(window=window*2+1, center=True).max()).astype(int)
        df['Swing_Low'] = (df['Low'] == df['Low'].rolling(window=window*2+1, center=True).min()).astype(int)
        
        # Market Structure - Track highs and lows
        df['Higher_High'] = (df['High'] > df['High'].shift(1).rolling(window=10).max()).astype(int)
        df['Lower_Low'] = (df['Low'] < df['Low'].shift(1).rolling(window=10).min()).astype(int)
        df['Higher_Low'] = (df['Low'] > df['Low'].shift(1).rolling(window=10).min()).astype(int)
        df['Lower_High'] = (df['High'] < df['High'].shift(1).rolling(window=10).max()).astype(int)
        
        # BOS (Break of Structure) - price breaks previous high/low
        df['BOS_Bullish'] = (df['Close'] > df['High'].shift(1).rolling(window=20).max().shift(1)).astype(int)
        df['BOS_Bearish'] = (df['Close'] < df['Low'].shift(1).rolling(window=20).min().shift(1)).astype(int)
        
        # CHoCH (Change of Character) - trend reversal signal
        hh_shift = df['Higher_High'].shift(1).fillna(0).astype(int)
        ll_shift = df['Lower_Low'].shift(1).fillna(0).astype(int)
        df['CHoCH'] = ((hh_shift & df['Lower_Low']) | (ll_shift & df['Higher_High'])).astype(int)
        
        # Liquidity zones (recent support/resistance)
        df['Resistance_Level'] = df['High'].rolling(window=50, min_periods=1).max()
        df['Support_Level'] = df['Low'].rolling(window=50, min_periods=1).min()
        df['Near_Resistance'] = (df['Close'] > df['Resistance_Level'] * 0.995).astype(int)
        df['Near_Support'] = (df['Close'] < df['Support_Level'] * 1.005).astype(int)
        
        # Liquidity sweep detection
        df['Liquidity_Sweep_Up'] = ((df['High'] > df['Resistance_Level'].shift(1)) & 
                                    (df['Close'] < df['Resistance_Level'].shift(1))).astype(int)
        df['Liquidity_Sweep_Down'] = ((df['Low'] < df['Support_Level'].shift(1)) & 
                                      (df['Close'] > df['Support_Level'].shift(1))).astype(int)
        
        # Order blocks (recent strong candles)
        df['Bullish_OB'] = ((df['Close'] > df['Open']) & 
                            (df['Close'] - df['Open'] > df['Close'].std() * 0.5)).astype(int)
        df['Bearish_OB'] = ((df['Close'] < df['Open']) & 
                            (df['Open'] - df['Close'] > df['Close'].std() * 0.5)).astype(int)
        
        # Fair Value Gap (FVG)
        df['FVG_Bullish'] = (df['Low'] > df['High'].shift(2)).astype(int)
        df['FVG_Bearish'] = (df['High'] < df['Low'].shift(2)).astype(int)
        
        logger.info("SMC signals calculated")
        return self
    
    def calculate_volume_features(self):
        """Calculate volume-based features"""
        logger.info("Calculating volume-based features...")
        
        df = self.df_filtered
        
        # Basic volume metrics
        df['Volume_MA_10'] = df['Volume'].rolling(window=10, min_periods=1).mean()
        df['Volume_MA_20'] = df['Volume'].rolling(window=20, min_periods=1).mean()
        df['Volume_Ratio'] = df['Volume'] / df['Volume_MA_20']
        df['Volume_Trend'] = np.where(df['Volume'] > df['Volume_MA_10'], 1, 0)
        
        # Volume-Price relationship
        df['Price_Change'] = df['Close'].pct_change()
        df['Volume_Price_Corr'] = df['Volume'].rolling(window=20, min_periods=1).corr(df['Price_Change'].abs())
        df['Volume_Price_Corr'] = df['Volume_Price_Corr'].fillna(0)
        
        # On-Balance Volume (OBV)
        df['OBV'] = (np.sign(df['Close'].diff()) * df['Volume']).cumsum()
        df['OBV_MA'] = df['OBV'].rolling(window=20, min_periods=1).mean()
        df['OBV_Trend'] = np.where(df['OBV'] > df['OBV_MA'], 1, 0)
        
        # Volume-weighted features
        df['VWAP'] = (df['Close'] * df['Volume']).cumsum() / df['Volume'].cumsum()
        df['Dist_to_VWAP'] = (df['Close'] - df['VWAP']) / df['VWAP']
        
        logger.info("Volume features calculated")
        return self
    
    def calculate_technical_indicators(self):
        """Calculate additional technical indicators"""
        logger.info("Calculating technical indicators...")
        
        df = self.df_filtered
        
        # Moving Averages
        for period in [5, 10, 20, 50, 100]:
            df[f'SMA_{period}'] = df['Close'].rolling(window=period, min_periods=1).mean()
            df[f'EMA_{period}'] = df['Close'].ewm(span=period, adjust=False, min_periods=1).mean()
        
        # Price relative to MAs
        df['Price_above_SMA20'] = (df['Close'] > df['SMA_20']).astype(int)
        df['Price_above_SMA50'] = (df['Close'] > df['SMA_50']).astype(int)
        df['MA_Cross_10_20'] = (df['SMA_10'] > df['SMA_20']).astype(int)
        df['MA_Cross_20_50'] = (df['SMA_20'] > df['SMA_50']).astype(int)
        
        # RSI
        delta = df['Close'].diff()
        gain = delta.where(delta > 0, 0).rolling(window=14, min_periods=1).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14, min_periods=1).mean()
        rs = gain / (loss + 1e-10)
        df['RSI'] = 100 - (100 / (1 + rs))
        df['RSI_Overbought'] = (df['RSI'] > 70).astype(int)
        df['RSI_Oversold'] = (df['RSI'] < 30).astype(int)
        
        # MACD
        ema_12 = df['Close'].ewm(span=12, adjust=False, min_periods=1).mean()
        ema_26 = df['Close'].ewm(span=26, adjust=False, min_periods=1).mean()
        df['MACD'] = ema_12 - ema_26
        df['MACD_Signal'] = df['MACD'].ewm(span=9, adjust=False, min_periods=1).mean()
        df['MACD_Histogram'] = df['MACD'] - df['MACD_Signal']
        df['MACD_Bullish'] = (df['MACD'] > df['MACD_Signal']).astype(int)
        
        # Bollinger Bands
        df['BB_Middle'] = df['Close'].rolling(window=20, min_periods=1).mean()
        bb_std = df['Close'].rolling(window=20, min_periods=1).std()
        df['BB_Upper'] = df['BB_Middle'] + 2 * bb_std
        df['BB_Lower'] = df['BB_Middle'] - 2 * bb_std
        df['BB_Width'] = (df['BB_Upper'] - df['BB_Lower']) / df['BB_Middle']
        df['BB_Position'] = (df['Close'] - df['BB_Lower']) / (df['BB_Upper'] - df['BB_Lower'] + 1e-10)
        df['BB_Squeeze'] = (df['BB_Width'] < df['BB_Width'].rolling(window=50, min_periods=1).mean() * 0.8).astype(int)
        
        # ATR (Average True Range)
        tr1 = df['High'] - df['Low']
        tr2 = abs(df['High'] - df['Close'].shift(1))
        tr3 = abs(df['Low'] - df['Close'].shift(1))
        df['True_Range'] = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        df['ATR'] = df['True_Range'].rolling(window=14, min_periods=1).mean()
        df['ATR_Ratio'] = df['ATR'] / df['Close']
        
        # Stochastic Oscillator
        low_14 = df['Low'].rolling(window=14, min_periods=1).min()
        high_14 = df['High'].rolling(window=14, min_periods=1).max()
        df['Stoch_K'] = 100 * (df['Close'] - low_14) / (high_14 - low_14 + 1e-10)
        df['Stoch_D'] = df['Stoch_K'].rolling(window=3, min_periods=1).mean()
        
        # Price momentum
        df['Momentum_5'] = df['Close'].pct_change(periods=5)
        df['Momentum_10'] = df['Close'].pct_change(periods=10)
        df['Momentum_20'] = df['Close'].pct_change(periods=20)
        
        # Volatility
        df['Volatility_20'] = df['Close'].pct_change().rolling(window=20, min_periods=1).std()
        
        logger.info("Technical indicators calculated")
        return self
    
    def create_target_variable(self, lookahead: int = 6):
        """Create target variable: 1 if price goes up in next N periods, 0 otherwise"""
        logger.info("Creating target variable with lookahead=%d periods", lookahead)
        
        df = self.df_filtered
        
        # Future price change
        future_return = df['Close'].shift(-lookahead) / df['Close'] - 1
        
        # Binary target: 1 if price increases by at least 0.1%, 0 otherwise
        threshold = 0.001
        df['Target'] = (future_return > threshold).astype(int)
        
        # Remove rows with NaN target
        self.df_filtered = df.dropna(subset=['Target']).reset_index(drop=True)
        
        class_dist = self.df_filtered['Target'].value_counts()
        logger.info("Target distribution: %s", class_dist.to_dict())
        return self
    
    def prepare_features(self):
        """Prepare feature matrix and target vector"""
        logger.info("Preparing features...")
        
        df = self.df_filtered
        
        # Select feature columns (exclude Date, Target, and intermediate calculations)
        exclude_cols = ['Date', 'Target', 'Rolling_High', 'Rolling_Low', 'Price_Range',
                       'Swing_High', 'Swing_Low', 'Resistance_Level', 'Support_Level',
                       'VWAP', 'OBV', 'True_Range']
        
        feature_cols = [c for c in df.columns if c not in exclude_cols and 
                       c not in ['Open', 'High', 'Low', 'Close']]
        
        # Add raw prices
        feature_cols.extend(['Open', 'High', 'Low', 'Close'])
        
        self.feature_names = feature_cols
        
        X = df[feature_cols].values
        y = df['Target'].values
        
        # Remove any remaining NaN/Inf
        valid_mask = np.all(np.isfinite(X), axis=1)
        X = X[valid_mask]
        y = y[valid_mask]
        
        logger.info("Feature matrix shape: %s", X.shape)
        logger.info("Number of features: %d", len(feature_cols))
        
        # Split data
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
        )
        
        logger.info("Train set: %s, Test set: %s", self.X_train.shape, self.X_test.shape)
        
        # Scale features
        self.scaler = RobustScaler()
        self.X_train = self.scaler.fit_transform(self.X_train)
        self.X_test = self.scaler.transform(self.X_test)
        
        return self
    
    def train_models(self):
        """Train multiple ML models with hyperparameter tuning"""
        logger.info("Training models...")
        
        # Define models and hyperparameter grids
        model_configs = {
            'Random Forest': {
                'model': RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=-1),
                'params': {
                    'n_estimators': [100, 200],
                    'max_depth': [10, 20, None],
                    'min_samples_split': [2, 5],
                    'min_samples_leaf': [1, 2]
                }
            },
            'Gradient Boosting': {
                'model': GradientBoostingClassifier(random_state=RANDOM_STATE),
                'params': {
                    'n_estimators': [100, 200],
                    'max_depth': [3, 5, 7],
                    'learning_rate': [0.05, 0.1],
                    'min_samples_split': [2, 5]
                }
            },
            'Logistic Regression': {
                'model': LogisticRegression(random_state=RANDOM_STATE, max_iter=1000),
                'params': {
                    'C': [0.1, 1, 10],
                    'penalty': ['l2'],
                    'solver': ['lbfgs', 'liblinear']
                }
            },
            'SVM': {
                'model': SVC(random_state=RANDOM_STATE, probability=True),
                'params': {
                    'C': [0.1, 1, 10],
                    'kernel': ['rbf', 'linear'],
                    'gamma': ['scale', 'auto']
                }
            }
        }
        
        if XGBOOST_AVAILABLE:
            model_configs['XGBoost'] = {
                'model': xgb.XGBClassifier(random_state=RANDOM_STATE, use_label_encoder=False, eval_metric='logloss'),
                'params': {
                    'n_estimators': [100, 200],
                    'max_depth': [3, 5, 7],
                    'learning_rate': [0.05, 0.1],
                    'subsample': [0.8, 1.0]
                }
            }
        
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
        
        for name, config in model_configs.items():
            logger.info("Training %s...", name)
            
            grid_search = GridSearchCV(
                config['model'], 
                config['params'], 
                cv=cv, 
                scoring='f1',
                n_jobs=-1,
                verbose=0
            )
            
            grid_search.fit(self.X_train, self.y_train)
            
            best_model = grid_search.best_estimator_
            self.models[name] = best_model
            
            # Cross-validation scores
            cv_scores = cross_val_score(best_model, self.X_train, self.y_train, cv=cv, scoring='f1')
            
            # Predictions
            y_pred = best_model.predict(self.X_test)
            y_prob = best_model.predict_proba(self.X_test)[:, 1] if hasattr(best_model, 'predict_proba') else None
            
            # Metrics
            self.results[name] = {
                'model': best_model,
                'best_params': grid_search.best_params_,
                'cv_f1_mean': cv_scores.mean(),
                'cv_f1_std': cv_scores.std(),
                'accuracy': accuracy_score(self.y_test, y_pred),
                'f1_score': f1_score(self.y_test, y_pred),
                'precision': precision_score(self.y_test, y_pred, zero_division=0),
                'recall': recall_score(self.y_test, y_pred, zero_division=0),
                'y_pred': y_pred,
                'y_prob': y_prob,
                'classification_report': classification_report(self.y_test, y_pred, output_dict=True)
            }
            
            logger.info("%s - Accuracy: %.4f, F1: %.4f", name, 
                       self.results[name]['accuracy'], self.results[name]['f1_score'])
        
        return self
    
    def select_best_model(self):
        """Select best model based on F1-score, accuracy, and CV stability"""
        logger.info("Selecting best model...")
        
        # Sort by F1-score (descending), then accuracy (descending), then CV std (ascending)
        sorted_models = sorted(
            self.results.items(),
            key=lambda x: (x[1]['f1_score'], x[1]['accuracy'], -x[1]['cv_f1_std']),
            reverse=True
        )
        
        # Check if any model meets the 65-70% accuracy target
        target_met = False
        for name, metrics in sorted_models:
            if 0.65 <= metrics['accuracy'] <= 0.70:
                self.best_model_name = name
                self.best_model = metrics['model']
                target_met = True
                break
        
        if not target_met:
            logger.warning("No model achieved 65-70% accuracy. Selecting best available.")
            self.best_model_name = sorted_models[0][0]
            self.best_model = sorted_models[0][1]['model']
        
        best_acc = self.results[self.best_model_name]['accuracy']
        logger.info("Best model: %s (Accuracy: %.2f%%)", self.best_model_name, best_acc * 100)
        
        return self
    
    # ============================================================================
    # ENHANCED VISUALIZATION METHODS
    # ============================================================================
    
    def plot_price_overview(self):
        """1. Plot comprehensive price overview with volume and technical indicators"""
        logger.info("Generating price overview visualization...")
        
        df = self.df_filtered.copy()
        
        fig = plt.figure(figsize=(16, 12))
        gs = gridspec.GridSpec(4, 1, height_ratios=[3, 1, 1, 1], hspace=0.05)
        
        # Price and Moving Averages
        ax1 = fig.add_subplot(gs[0])
        ax1.plot(df['Date'], df['Close'], label='Close Price', color='black', linewidth=1)
        ax1.plot(df['Date'], df['SMA_20'], label='SMA 20', color='blue', alpha=0.7)
        ax1.plot(df['Date'], df['SMA_50'], label='SMA 50', color='red', alpha=0.7)
        ax1.plot(df['Date'], df['SMA_100'], label='SMA 100', color='green', alpha=0.7)
        
        # Bollinger Bands
        ax1.fill_between(df['Date'], df['BB_Upper'], df['BB_Lower'], alpha=0.1, color='gray', label='Bollinger Bands')
        
        # Highlight target variable periods
        up_periods = df[df['Target'] == 1]
        ax1.scatter(up_periods['Date'], up_periods['Close'], color='green', alpha=0.3, s=10, label='Target=Up')
        
        ax1.set_ylabel('Price (USD)')
        ax1.set_title('XAU/USD Price Overview with Technical Indicators', fontsize=14, fontweight='bold')
        ax1.legend(loc='upper left', fontsize=8)
        ax1.grid(True, alpha=0.3)
        
        # Volume
        ax2 = fig.add_subplot(gs[1], sharex=ax1)
        colors = ['green' if df['Close'].iloc[i] >= df['Open'].iloc[i] else 'red' for i in range(len(df))]
        ax2.bar(df['Date'], df['Volume'], color=colors, alpha=0.6, width=0.8)
        ax2.plot(df['Date'], df['Volume_MA_20'], color='blue', linewidth=1, label='Volume MA20')
        ax2.set_ylabel('Volume')
        ax2.legend(loc='upper left', fontsize=8)
        ax2.grid(True, alpha=0.3)
        
        # RSI
        ax3 = fig.add_subplot(gs[2], sharex=ax1)
        ax3.plot(df['Date'], df['RSI'], color='purple', linewidth=1)
        ax3.axhline(y=70, color='r', linestyle='--', alpha=0.5, label='Overbought')
        ax3.axhline(y=30, color='g', linestyle='--', alpha=0.5, label='Oversold')
        ax3.fill_between(df['Date'], 30, 70, alpha=0.1, color='blue')
        ax3.set_ylabel('RSI')
        ax3.set_ylim(0, 100)
        ax3.legend(loc='upper left', fontsize=8)
        ax3.grid(True, alpha=0.3)
        
        # MACD
        ax4 = fig.add_subplot(gs[3], sharex=ax1)
        ax4.plot(df['Date'], df['MACD'], label='MACD', color='blue', linewidth=1)
        ax4.plot(df['Date'], df['MACD_Signal'], label='Signal', color='red', linewidth=1)
        ax4.bar(df['Date'], df['MACD_Histogram'], label='Histogram', 
                color=['green' if h >= 0 else 'red' for h in df['MACD_Histogram']], alpha=0.6)
        ax4.set_ylabel('MACD')
        ax4.set_xlabel('Date')
        ax4.legend(loc='upper left', fontsize=8)
        ax4.grid(True, alpha=0.3)
        
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(self.viz_dir / '01_price_overview.png', dpi=150, bbox_inches='tight')
        plt.close()
        
    def plot_fibonacci_analysis(self):
        """2. Plot Fibonacci retracement analysis"""
        logger.info("Generating Fibonacci analysis visualization...")
        
        df = self.df_filtered.copy()
        
        # Select recent 500 periods for clarity
        sample_df = df.tail(500).reset_index(drop=True)
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 10), gridspec_kw={'height_ratios': [3, 1]})
        
        # Price with Fibonacci levels
        ax1.plot(sample_df['Date'], sample_df['Close'], label='Close', color='black', linewidth=1.5)
        
        # Plot Fibonacci levels
        fib_colors = ['red', 'orange', 'yellow', 'green', 'blue', 'purple', 'gray']
        fib_ratios = [0.0, 0.236, 0.382, 0.5, 0.618, 0.786, 1.0]
        
        for ratio, color in zip(fib_ratios, fib_colors):
            col_name = f'Fib_{int(ratio*1000)}'
            ax1.plot(sample_df['Date'], sample_df[col_name], 
                    label=f'Fib {ratio:.1%}', color=color, linestyle='--', alpha=0.7)
        
        # Highlight current Fibonacci position
        ax1_twin = ax1.twinx()
        ax1_twin.fill_between(sample_df['Date'], 0, sample_df['Fib_Position'], 
                             alpha=0.2, color='green', label='Fib Position')
        ax1_twin.set_ylabel('Fib Position (0=Support, 1=Resistance)', color='green')
        ax1_twin.set_ylim(0, 1)
        
        ax1.set_ylabel('Price (USD)')
        ax1.set_title('Fibonacci Retracement Analysis', fontsize=14, fontweight='bold')
        ax1.legend(loc='upper left', fontsize=8)
        ax1.grid(True, alpha=0.3)
        
        # Distance to key Fibonacci levels
        ax2.plot(sample_df['Date'], sample_df['Dist_to_Fib_236'], label='Dist to 23.6%', alpha=0.7)
        ax2.plot(sample_df['Date'], sample_df['Dist_to_Fib_382'], label='Dist to 38.2%', alpha=0.7)
        ax2.plot(sample_df['Date'], sample_df['Dist_to_Fib_500'], label='Dist to 50%', alpha=0.7)
        ax2.plot(sample_df['Date'], sample_df['Dist_to_Fib_618'], label='Dist to 61.8%', alpha=0.7)
        ax2.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
        ax2.set_ylabel('Normalized Distance')
        ax2.set_xlabel('Date')
        ax2.set_title('Distance to Key Fibonacci Levels')
        ax2.legend(loc='upper left', fontsize=8)
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.viz_dir / '02_fibonacci_analysis.png', dpi=150, bbox_inches='tight')
        plt.close()
        
    def plot_smc_analysis(self):
        """3. Plot Smart Money Concepts (SMC) analysis"""
        logger.info("Generating SMC analysis visualization...")
        
        df = self.df_filtered.copy()
        sample_df = df.tail(300).reset_index(drop=True)
        
        fig, axes = plt.subplots(3, 1, figsize=(16, 12), sharex=True)
        
        # Price with BOS/CHoCH signals
        ax1 = axes[0]
        ax1.plot(sample_df['Date'], sample_df['Close'], color='black', linewidth=1.5, label='Close')
        
        # Mark BOS signals
        bos_bull = sample_df[sample_df['BOS_Bullish'] == 1]
        bos_bear = sample_df[sample_df['BOS_Bearish'] == 1]
        choch = sample_df[sample_df['CHoCH'] == 1]
        
        ax1.scatter(bos_bull['Date'], bos_bull['Close'], color='green', marker='^', 
                   s=100, label='BOS Bullish', zorder=5)
        ax1.scatter(bos_bear['Date'], bos_bear['Close'], color='red', marker='v', 
                   s=100, label='BOS Bearish', zorder=5)
        ax1.scatter(choch['Date'], choch['Close'], color='orange', marker='o', 
                   s=50, label='CHoCH', zorder=5)
        
        # Support and Resistance levels
        ax1.plot(sample_df['Date'], sample_df['Resistance_Level'], 'r--', alpha=0.5, label='Resistance')
        ax1.plot(sample_df['Date'], sample_df['Support_Level'], 'g--', alpha=0.5, label='Support')
        
        ax1.set_ylabel('Price (USD)')
        ax1.set_title('Smart Money Concepts (SMC) - Market Structure', fontsize=14, fontweight='bold')
        ax1.legend(loc='upper left', fontsize=8)
        ax1.grid(True, alpha=0.3)
        
        # Order Blocks and Liquidity Sweeps
        ax2 = axes[1]
        ax2.plot(sample_df['Date'], sample_df['Close'], color='gray', alpha=0.5, linewidth=1)
        
        # Mark Order Blocks
        bull_ob = sample_df[sample_df['Bullish_OB'] == 1]
        bear_ob = sample_df[sample_df['Bearish_OB'] == 1]
        
        ax2.bar(bull_ob['Date'], bull_ob['Close'] - bull_ob['Open'], 
               bottom=bull_ob['Open'], color='green', alpha=0.7, width=0.8, label='Bullish OB')
        ax2.bar(bear_ob['Date'], bear_ob['Close'] - bear_ob['Open'], 
               bottom=bear_ob['Open'], color='red', alpha=0.7, width=0.8, label='Bearish OB')
        
        # Liquidity Sweeps
        sweep_up = sample_df[sample_df['Liquidity_Sweep_Up'] == 1]
        sweep_down = sample_df[sample_df['Liquidity_Sweep_Down'] == 1]
        ax2.scatter(sweep_up['Date'], sweep_up['High'], color='purple', marker='*', 
                   s=150, label='Liquidity Sweep Up', zorder=5)
        ax2.scatter(sweep_down['Date'], sweep_down['Low'], color='brown', marker='*', 
                   s=150, label='Liquidity Sweep Down', zorder=5)
        
        ax2.set_ylabel('Price (USD)')
        ax2.set_title('Order Blocks and Liquidity Sweeps')
        ax2.legend(loc='upper left', fontsize=8)
        ax2.grid(True, alpha=0.3)
        
        # Fair Value Gaps
        ax3 = axes[2]
        fvg_bull = sample_df[sample_df['FVG_Bullish'] == 1]
        fvg_bear = sample_df[sample_df['FVG_Bearish'] == 1]
        
        ax3.plot(sample_df['Date'], sample_df['Close'], color='black', linewidth=1)
        ax3.scatter(fvg_bull['Date'], fvg_bull['Close'], color='lime', marker='s', 
                   s=50, label='FVG Bullish', alpha=0.7)
        ax3.scatter(fvg_bear['Date'], fvg_bear['Close'], color='magenta', marker='s', 
                   s=50, label='FVG Bearish', alpha=0.7)
        
        ax3.set_ylabel('Price (USD)')
        ax3.set_xlabel('Date')
        ax3.set_title('Fair Value Gaps (FVG)')
        ax3.legend(loc='upper left', fontsize=8)
        ax3.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.viz_dir / '03_smc_analysis.png', dpi=150, bbox_inches='tight')
        plt.close()
        
    def plot_volume_analysis(self):
        """4. Comprehensive volume analysis"""
        logger.info("Generating volume analysis visualization...")
        
        df = self.df_filtered.copy()
        
        fig, axes = plt.subplots(3, 2, figsize=(16, 12))
        
        # Volume distribution
        ax1 = axes[0, 0]
        ax1.hist(df['Volume'], bins=50, color='steelblue', alpha=0.7, edgecolor='black')
        ax1.axvline(df['Volume'].mean(), color='red', linestyle='--', label=f'Mean: {df["Volume"].mean():.0f}')
        ax1.axvline(df['Volume'].median(), color='green', linestyle='--', label=f'Median: {df["Volume"].median():.0f}')
        ax1.set_xlabel('Volume')
        ax1.set_ylabel('Frequency')
        ax1.set_title('Volume Distribution')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Volume vs Price Change
        ax2 = axes[0, 1]
        scatter = ax2.scatter(df['Volume_Ratio'], df['Price_Change'] * 100, 
                            c=df['Target'], cmap='RdYlGn', alpha=0.5, s=10)
        ax2.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
        ax2.axvline(x=1, color='black', linestyle='--', linewidth=0.5)
        ax2.set_xlabel('Volume Ratio (Vol/MA20)')
        ax2.set_ylabel('Price Change (%)')
        ax2.set_title('Volume Ratio vs Price Change (colored by Target)')
        plt.colorbar(scatter, ax=ax2, label='Target')
        ax2.grid(True, alpha=0.3)
        
        # OBV Trend
        ax3 = axes[1, 0]
        ax3.plot(df['Date'], df['OBV'], label='OBV', color='blue', linewidth=1)
        ax3.plot(df['Date'], df['OBV_MA'], label='OBV MA20', color='red', linewidth=1)
        ax3_twin = ax3.twinx()
        ax3_twin.plot(df['Date'], df['Close'], color='gray', alpha=0.5, linewidth=1, label='Close')
        ax3.set_ylabel('OBV')
        ax3_twin.set_ylabel('Price')
        ax3.set_title('On-Balance Volume (OBV)')
        ax3.legend(loc='upper left')
        ax3_twin.legend(loc='upper right')
        ax3.grid(True, alpha=0.3)
        
        # Volume-Price Correlation over time
        ax4 = axes[1, 1]
        ax4.plot(df['Date'], df['Volume_Price_Corr'], color='purple', linewidth=1)
        ax4.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
        ax4.fill_between(df['Date'], 0, df['Volume_Price_Corr'], 
                        where=(df['Volume_Price_Corr'] > 0), alpha=0.3, color='green')
        ax4.fill_between(df['Date'], 0, df['Volume_Price_Corr'], 
                        where=(df['Volume_Price_Corr'] < 0), alpha=0.3, color='red')
        ax4.set_ylabel('Correlation')
        ax4.set_title('Volume-Price Correlation (20-period)')
        ax4.grid(True, alpha=0.3)
        
        # VWAP Analysis
        ax5 = axes[2, 0]
        ax5.plot(df['Date'], df['Close'], label='Close', color='black', linewidth=1)
        ax5.plot(df['Date'], df['VWAP'], label='VWAP', color='orange', linewidth=1)
        ax5.fill_between(df['Date'], df['Close'], df['VWAP'], 
                        where=(df['Close'] > df['VWAP']), alpha=0.3, color='green')
        ax5.fill_between(df['Date'], df['Close'], df['VWAP'], 
                        where=(df['Close'] < df['VWAP']), alpha=0.3, color='red')
        ax5.set_ylabel('Price (USD)')
        ax5.set_xlabel('Date')
        ax5.set_title('Price vs VWAP')
        ax5.legend()
        ax5.grid(True, alpha=0.3)
        
        # Volume Profile (Price vs Volume heatmap)
        ax6 = axes[2, 1]
        price_bins = np.linspace(df['Close'].min(), df['Close'].max(), 50)
        volume_profile = []
        for i in range(len(price_bins)-1):
            mask = (df['Close'] >= price_bins[i]) & (df['Close'] < price_bins[i+1])
            volume_profile.append(df[mask]['Volume'].sum())
        
        ax6.barh(price_bins[:-1], volume_profile, height=np.diff(price_bins), color='steelblue', alpha=0.7)
        ax6.axhline(y=df['Close'].iloc[-1], color='red', linestyle='--', linewidth=2, label='Current Price')
        ax6.set_xlabel('Total Volume')
        ax6.set_ylabel('Price Level')
        ax6.set_title('Volume Profile')
        ax6.legend()
        ax6.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.viz_dir / '04_volume_analysis.png', dpi=150, bbox_inches='tight')
        plt.close()
        
    def plot_correlation_analysis(self):
        """5. Feature correlation analysis"""
        logger.info("Generating correlation analysis visualization...")
        
        df = self.df_filtered.copy()
        
        # Select key features for correlation
        key_features = ['Close', 'Volume', 'RSI', 'MACD', 'ATR', 'BB_Width', 
                       'Momentum_10', 'Volatility_20', 'Fib_Position', 'Volume_Ratio',
                       'BOS_Bullish', 'BOS_Bearish', 'CHoCH', 'Target']
        
        corr_df = df[key_features].corr()
        
        fig, axes = plt.subplots(1, 2, figsize=(18, 8))
        
        # Correlation heatmap
        ax1 = axes[0]
        mask = np.triu(np.ones_like(corr_df, dtype=bool))
        sns.heatmap(corr_df, mask=mask, annot=True, fmt='.2f', cmap='RdBu_r', 
                   center=0, square=True, linewidths=0.5, cbar_kws={"shrink": 0.8}, ax=ax1)
        ax1.set_title('Feature Correlation Matrix', fontsize=14, fontweight='bold')
        
        # Correlation with target
        ax2 = axes[1]
        target_corr = corr_df['Target'].drop('Target').sort_values(key=abs, ascending=True)
        colors = ['green' if x > 0 else 'red' for x in target_corr.values]
        ax2.barh(range(len(target_corr)), target_corr.values, color=colors, alpha=0.7)
        ax2.set_yticks(range(len(target_corr)))
        ax2.set_yticklabels(target_corr.index)
        ax2.set_xlabel('Correlation with Target')
        ax2.set_title('Feature Correlation with Target Variable', fontsize=14, fontweight='bold')
        ax2.axvline(x=0, color='black', linestyle='-', linewidth=0.5)
        ax2.grid(True, alpha=0.3, axis='x')
        
        plt.tight_layout()
        plt.savefig(self.viz_dir / '05_correlation_analysis.png', dpi=150, bbox_inches='tight')
        plt.close()
        
    def plot_feature_distributions(self):
        """6. Feature distributions by target class"""
        logger.info("Generating feature distribution visualization...")
        
        df = self.df_filtered.copy()
        
        # Select top features based on correlation with target
        features_to_plot = ['RSI', 'MACD_Histogram', 'ATR_Ratio', 'Momentum_10', 
                           'BB_Position', 'Volume_Ratio', 'Fib_Position', 'Dist_to_VWAP']
        
        fig, axes = plt.subplots(2, 4, figsize=(20, 10))
        axes = axes.flatten()
        
        for idx, feature in enumerate(features_to_plot):
            ax = axes[idx]
            
            # Split by target
            up_data = df[df['Target'] == 1][feature]
            down_data = df[df['Target'] == 0][feature]
            
            ax.hist(down_data, bins=30, alpha=0.5, label='Down (0)', color='red', density=True)
            ax.hist(up_data, bins=30, alpha=0.5, label='Up (1)', color='green', density=True)
            
            ax.set_xlabel(feature)
            ax.set_ylabel('Density')
            ax.set_title(f'Distribution of {feature}')
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            # Add statistics
            up_mean = up_data.mean()
            down_mean = down_data.mean()
            ax.axvline(up_mean, color='green', linestyle='--', alpha=0.7, label=f'Up Mean: {up_mean:.2f}')
            ax.axvline(down_mean, color='red', linestyle='--', alpha=0.7, label=f'Down Mean: {down_mean:.2f}')
        
        plt.tight_layout()
        plt.savefig(self.viz_dir / '06_feature_distributions.png', dpi=150, bbox_inches='tight')
        plt.close()
        
    def plot_dimensionality_reduction(self):
        """7. PCA and t-SNE visualization"""
        logger.info("Generating dimensionality reduction visualization...")
        
        # Use a subset for t-SNE (computationally expensive)
        sample_size = min(5000, len(self.X_train))
        indices = np.random.choice(len(self.X_train), sample_size, replace=False)
        X_sample = self.X_train[indices]
        y_sample = self.y_train[indices]
        
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        
        # PCA
        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(X_sample)
        
        ax1 = axes[0]
        scatter1 = ax1.scatter(X_pca[:, 0], X_pca[:, 1], c=y_sample, cmap='RdYlGn', 
                              alpha=0.5, s=10)
        ax1.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)')
        ax1.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)')
        ax1.set_title('PCA: First Two Principal Components')
        plt.colorbar(scatter1, ax=ax1, label='Target')
        ax1.grid(True, alpha=0.3)
        
        # t-SNE
        tsne = TSNE(n_components=2, random_state=RANDOM_STATE, perplexity=30, n_iter=1000)
        X_tsne = tsne.fit_transform(X_sample)
        
        ax2 = axes[1]
        scatter2 = ax2.scatter(X_tsne[:, 0], X_tsne[:, 1], c=y_sample, cmap='RdYlGn', 
                              alpha=0.5, s=10)
        ax2.set_xlabel('t-SNE 1')
        ax2.set_ylabel('t-SNE 2')
        ax2.set_title('t-SNE: 2D Embedding of Feature Space')
        plt.colorbar(scatter2, ax=ax2, label='Target')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.viz_dir / '07_dimensionality_reduction.png', dpi=150, bbox_inches='tight')
        plt.close()
        
        # Explained variance plot for PCA
        pca_full = PCA()
        pca_full.fit(self.X_train)
        
        plt.figure(figsize=(12, 5))
        plt.subplot(1, 2, 1)
        plt.plot(range(1, 21), pca_full.explained_variance_ratio_[:20], 'bo-')
        plt.xlabel('Principal Component')
        plt.ylabel('Explained Variance Ratio')
        plt.title('PCA Explained Variance by Component')
        plt.grid(True, alpha=0.3)
        
        plt.subplot(1, 2, 2)
        cumsum = np.cumsum(pca_full.explained_variance_ratio_)
        plt.plot(range(1, 51), cumsum[:50], 'ro-')
        plt.axhline(y=0.95, color='k', linestyle='--', label='95% variance')
        plt.xlabel('Number of Components')
        plt.ylabel('Cumulative Explained Variance')
        plt.title('PCA Cumulative Explained Variance')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.viz_dir / '07b_pca_variance.png', dpi=150, bbox_inches='tight')
        plt.close()
        
    def plot_learning_curves(self):
        """8. Learning curves for all models"""
        logger.info("Generating learning curves visualization...")
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        axes = axes.flatten()
        
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
        
        for idx, (name, model) in enumerate(self.models.items()):
            if idx >= 6:
                break
                
            ax = axes[idx]
            
            train_sizes, train_scores, val_scores = learning_curve(
                model, self.X_train, self.y_train, cv=cv, n_jobs=-1,
                train_sizes=np.linspace(0.1, 1.0, 10), scoring='f1'
            )
            
            train_mean = np.mean(train_scores, axis=1)
            train_std = np.std(train_scores, axis=1)
            val_mean = np.mean(val_scores, axis=1)
            val_std = np.std(val_scores, axis=1)
            
            ax.plot(train_sizes, train_mean, 'o-', color='blue', label='Training Score')
            ax.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, alpha=0.1, color='blue')
            ax.plot(train_sizes, val_mean, 'o-', color='red', label='Validation Score')
            ax.fill_between(train_sizes, val_mean - val_std, val_mean + val_std, alpha=0.1, color='red')
            
            ax.set_xlabel('Training Set Size')
            ax.set_ylabel('F1 Score')
            ax.set_title(f'Learning Curve: {name}')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        # Hide unused subplots
        for idx in range(len(self.models), 6):
            axes[idx].axis('off')
        
        plt.tight_layout()
        plt.savefig(self.viz_dir / '08_learning_curves.png', dpi=150, bbox_inches='tight')
        plt.close()
        
    def plot_model_comparison(self):
        """9. Comprehensive model comparison"""
        logger.info("Generating model comparison visualization...")
        
        fig = plt.figure(figsize=(20, 12))
        gs = gridspec.GridSpec(3, 3, figure=fig)
        
        # Metrics comparison bar chart
        ax1 = fig.add_subplot(gs[0, :2])
        models = list(self.results.keys())
        metrics = ['accuracy', 'f1_score', 'precision', 'recall']
        x = np.arange(len(models))
        width = 0.2
        
        for i, metric in enumerate(metrics):
            values = [self.results[m][metric] for m in models]
            ax1.bar(x + i*width, values, width, label=metric.replace('_', ' ').title())
        
        ax1.set_ylabel('Score')
        ax1.set_title('Model Performance Comparison', fontsize=14, fontweight='bold')
        ax1.set_xticks(x + width * 1.5)
        ax1.set_xticklabels(models, rotation=45, ha='right')
        ax1.legend()
        ax1.axhline(y=0.65, color='green', linestyle='--', alpha=0.5, label='Target Min')
        ax1.axhline(y=0.70, color='green', linestyle='--', alpha=0.5, label='Target Max')
        ax1.set_ylim(0, 1)
        ax1.grid(True, alpha=0.3, axis='y')
        
        # Cross-validation scores distribution
        ax2 = fig.add_subplot(gs[0, 2])
        cv_data = [[self.results[m]['cv_f1_mean']] for m in models]
        bp = ax2.boxplot(cv_data, labels=models, patch_artist=True)
        for patch, color in zip(bp['boxes'], plt.cm.Set3(np.linspace(0, 1, len(models)))):
            patch.set_facecolor(color)
        ax2.set_ylabel('CV F1 Score')
        ax2.set_title('Cross-Validation F1 Scores')
        ax2.tick_params(axis='x', rotation=45)
        ax2.grid(True, alpha=0.3, axis='y')
        
        # ROC Curves
        ax3 = fig.add_subplot(gs[1, :])
        for name, metrics in self.results.items():
            if metrics['y_prob'] is not None:
                fpr, tpr, _ = roc_curve(self.y_test, metrics['y_prob'])
                roc_auc = auc(fpr, tpr)
                ax3.plot(fpr, tpr, label=f'{name} (AUC = {roc_auc:.3f})', linewidth=2)
        
        ax3.plot([0, 1], [0, 1], 'k--', label='Random Classifier', linewidth=1)
        ax3.set_xlabel('False Positive Rate')
        ax3.set_ylabel('True Positive Rate')
        ax3.set_title('ROC Curves Comparison', fontsize=14, fontweight='bold')
        ax3.legend(loc='lower right')
        ax3.grid(True, alpha=0.3)
        
        # Precision-Recall Curves
        ax4 = fig.add_subplot(gs[2, :2])
        for name, metrics in self.results.items():
            if metrics['y_prob'] is not None:
                precision, recall, _ = precision_recall_curve(self.y_test, metrics['y_prob'])
                avg_precision = average_precision_score(self.y_test, metrics['y_prob'])
                ax4.plot(recall, precision, label=f'{name} (AP = {avg_precision:.3f})', linewidth=2)
        
        ax4.set_xlabel('Recall')
        ax4.set_ylabel('Precision')
        ax4.set_title('Precision-Recall Curves', fontsize=14, fontweight='bold')
        ax4.legend(loc='lower left')
        ax4.grid(True, alpha=0.3)
        
        # Calibration plot
        ax5 = fig.add_subplot(gs[2, 2])
        from sklearn.calibration import calibration_curve
        
        for name, metrics in self.results.items():
            if metrics['y_prob'] is not None:
                prob_true, prob_pred = calibration_curve(self.y_test, metrics['y_prob'], n_bins=10)
                ax5.plot(prob_pred, prob_true, 's-', label=name, alpha=0.7)
        
        ax5.plot([0, 1], [0, 1], 'k--', label='Perfectly Calibrated')
        ax5.set_xlabel('Mean Predicted Probability')
        ax5.set_ylabel('Fraction of Positives')
        ax5.set_title('Calibration Plot')
        ax5.legend(loc='lower right', fontsize=8)
        ax5.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.viz_dir / '09_model_comparison.png', dpi=150, bbox_inches='tight')
        plt.close()
        
    def plot_confusion_matrices_detailed(self):
        """10. Detailed confusion matrices with additional metrics"""
        logger.info("Generating detailed confusion matrices...")
        
        n_models = len(self.results)
        n_cols = 3
        n_rows = (n_models + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5*n_rows))
        axes = axes.flatten() if n_models > 1 else [axes]
        
        for idx, (name, metrics) in enumerate(self.results.items()):
            ax = axes[idx]
            cm = confusion_matrix(self.y_test, metrics['y_pred'])
            
            # Normalize confusion matrix
            cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            
            # Create heatmap
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                       xticklabels=['Down', 'Up'], yticklabels=['Down', 'Up'],
                       cbar_kws={'label': 'Count'})
            
            # Add percentage annotations
            for i in range(2):
                for j in range(2):
                    text = ax.texts[i*2 + j]
                    text.set_text(f'{cm[i, j]}\\n({cm_norm[i, j]:.1%})')
            
            ax.set_title(f'{name}\\nAccuracy: {metrics["accuracy"]:.3f} | F1: {metrics["f1_score"]:.3f}')
            ax.set_xlabel('Predicted')
            ax.set_ylabel('Actual')
        
        # Hide unused subplots
        for idx in range(n_models, len(axes)):
            axes[idx].axis('off')
        
        plt.tight_layout()
        plt.savefig(self.viz_dir / '10_confusion_matrices.png', dpi=150, bbox_inches='tight')
        plt.close()
        
    def plot_feature_importance_detailed(self):
        """11. Detailed feature importance analysis"""
        logger.info("Generating detailed feature importance visualization...")
        
        tree_models = [n for n in self.models.keys() if hasattr(self.models[n], 'feature_importances_')]
        
        if not tree_models:
            logger.warning("No tree-based models available for feature importance")
            return
        
        fig = plt.figure(figsize=(20, 12))
        gs = gridspec.GridSpec(2, 2, figure=fig)
        
        # Feature importance comparison across models
        ax1 = fig.add_subplot(gs[0, :])
        importance_df = pd.DataFrame(index=self.feature_names)
        
        for name in tree_models:
            importance_df[name] = self.models[name].feature_importances_
        
        # Select top 20 features by average importance
        importance_df['Mean'] = importance_df.mean(axis=1)
        top_features = importance_df.nlargest(20, 'Mean')
        
        x = np.arange(len(top_features))
        width = 0.8 / len(tree_models)
        
        for i, model in enumerate(tree_models):
            ax1.bar(x + i*width, top_features[model], width, label=model, alpha=0.8)
        
        ax1.set_xticks(x + width * (len(tree_models)-1) / 2)
        ax1.set_xticklabels(top_features.index, rotation=45, ha='right')
        ax1.set_ylabel('Feature Importance')
        ax1.set_title('Top 20 Feature Importance Comparison', fontsize=14, fontweight='bold')
        ax1.legend()
        ax1.grid(True, alpha=0.3, axis='y')
        
        # Cumulative importance
        ax2 = fig.add_subplot(gs[1, 0])
        sorted_importance = top_features['Mean'].sort_values(ascending=False)
        cumsum = np.cumsum(sorted_importance)
        ax2.plot(range(1, len(cumsum)+1), cumsum, 'bo-', markersize=4)
        ax2.axhline(y=0.9, color='r', linestyle='--', label='90% importance')
        ax2.set_xlabel('Number of Features')
        ax2.set_ylabel('Cumulative Importance')
        ax2.set_title('Cumulative Feature Importance')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Feature importance heatmap
        ax3 = fig.add_subplot(gs[1, 1])
        importance_matrix = top_features[tree_models].T
        sns.heatmap(importance_matrix, cmap='YlOrRd', cbar_kws={'label': 'Importance'}, ax=ax3)
        ax3.set_title('Feature Importance Heatmap')
        ax3.set_xlabel('Features')
        ax3.set_ylabel('Models')
        
        plt.tight_layout()
        plt.savefig(self.viz_dir / '11_feature_importance.png', dpi=150, bbox_inches='tight')
        plt.close()
        
        # Store for later use
        self.feature_importance_df = importance_df
        
    def plot_prediction_analysis(self):
        """12. Analysis of model predictions over time"""
        logger.info("Generating prediction analysis visualization...")
        
        # Get predictions from best model
        best_model = self.best_model
        y_pred = best_model.predict(self.X_test)
        y_prob = best_model.predict_proba(self.X_test)[:, 1] if hasattr(best_model, 'predict_proba') else None
        
        # Create a dataframe with test data
        test_df = self.df_filtered.iloc[-len(self.y_test):].copy()
        test_df['Predicted'] = y_pred
        test_df['Actual'] = self.y_test
        test_df['Correct'] = (y_pred == self.y_test).astype(int)
        
        if y_prob is not None:
            test_df['Probability'] = y_prob
        
        fig, axes = plt.subplots(4, 1, figsize=(16, 14), sharex=True)
        
        # Price with predictions
        ax1 = axes[0]
        ax1.plot(test_df['Date'], test_df['Close'], color='black', linewidth=1, label='Close')
        
        correct_up = test_df[(test_df['Correct'] == 1) & (test_df['Predicted'] == 1)]
        correct_down = test_df[(test_df['Correct'] == 1) & (test_df['Predicted'] == 0)]
        wrong_up = test_df[(test_df['Correct'] == 0) & (test_df['Predicted'] == 1)]
        wrong_down = test_df[(test_df['Correct'] == 0) & (test_df['Predicted'] == 0)]
        
        ax1.scatter(correct_up['Date'], correct_up['Close'], color='green', marker='^', 
                   s=50, label='Correct Up', alpha=0.7, zorder=5)
        ax1.scatter(correct_down['Date'], correct_down['Close'], color='blue', marker='v', 
                   s=50, label='Correct Down', alpha=0.7, zorder=5)
        ax1.scatter(wrong_up['Date'], wrong_up['Close'], color='orange', marker='^', 
                   s=50, label='False Up', alpha=0.7, zorder=5)
        ax1.scatter(wrong_down['Date'], wrong_down['Close'], color='red', marker='v', 
                   s=50, label='False Down', alpha=0.7, zorder=5)
        
        ax1.set_ylabel('Price (USD)')
        ax1.set_title(f'Prediction Analysis - {self.best_model_name}', fontsize=14, fontweight='bold')
        ax1.legend(loc='upper left', fontsize=8)
        ax1.grid(True, alpha=0.3)
        
        # Prediction confidence over time
        if y_prob is not None:
            ax2 = axes[1]
            ax2.fill_between(test_df['Date'], 0.5, test_df['Probability'], 
                           where=(test_df['Probability'] > 0.5), alpha=0.3, color='green', label='Bullish Confidence')
            ax2.fill_between(test_df['Date'], test_df['Probability'], 0.5, 
                           where=(test_df['Probability'] <= 0.5), alpha=0.3, color='red', label='Bearish Confidence')
            ax2.axhline(y=0.5, color='black', linestyle='-', linewidth=0.5)
            ax2.set_ylabel('Probability')
            ax2.set_title('Prediction Confidence Over Time')
            ax2.legend(loc='upper left', fontsize=8)
            ax2.set_ylim(0, 1)
            ax2.grid(True, alpha=0.3)
        
        # Rolling accuracy
        ax3 = axes[2]
        window = 50
        rolling_accuracy = test_df['Correct'].rolling(window=window, min_periods=1).mean()
        ax3.plot(test_df['Date'], rolling_accuracy, color='purple', linewidth=2)
        ax3.axhline(y=0.5, color='red', linestyle='--', alpha=0.5, label='Random')
        ax3.axhline(y=self.results[self.best_model_name]['accuracy'], color='green', 
                   linestyle='--', alpha=0.5, label='Overall Accuracy')
        ax3.set_ylabel('Accuracy')
        ax3.set_title(f'Rolling Accuracy ({window}-period window)')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # Prediction distribution by hour/day
        ax4 = axes[3]
        test_df['Hour'] = test_df['Date'].dt.hour
        hourly_accuracy = test_df.groupby('Hour')['Correct'].mean()
        
        ax4.bar(hourly_accuracy.index, hourly_accuracy.values, color='steelblue', alpha=0.7)
        ax4.axhline(y=self.results[self.best_model_name]['accuracy'], color='red', 
                   linestyle='--', label='Overall Accuracy')
        ax4.set_xlabel('Hour of Day')
        ax4.set_ylabel('Accuracy')
        ax4.set_title('Prediction Accuracy by Hour of Day')
        ax4.legend()
        ax4.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        plt.savefig(self.viz_dir / '12_prediction_analysis.png', dpi=150, bbox_inches='tight')
        plt.close()
        
    def plot_residual_analysis(self):
        """13. Residual and error analysis"""
        logger.info("Generating residual analysis visualization...")
        
        y_pred = self.best_model.predict(self.X_test)
        y_prob = self.best_model.predict_proba(self.X_test)[:, 1] if hasattr(self.best_model, 'predict_proba') else None
        
        errors = (y_pred != self.y_test).astype(int)
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        
        # Error distribution by predicted probability
        if y_prob is not None:
            ax1 = axes[0, 0]
            correct_prob = y_prob[y_pred == self.y_test]
            wrong_prob = y_prob[y_pred != self.y_test]
            
            ax1.hist(correct_prob, bins=20, alpha=0.5, label='Correct', color='green', density=True)
            ax1.hist(wrong_prob, bins=20, alpha=0.5, label='Wrong', color='red', density=True)
            ax1.set_xlabel('Predicted Probability')
            ax1.set_ylabel('Density')
            ax1.set_title('Prediction Confidence Distribution')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            
            # Calibration curve
            ax2 = axes[0, 1]
            from sklearn.calibration import calibration_curve
            prob_true, prob_pred = calibration_curve(self.y_test, y_prob, n_bins=10)
            ax2.plot(prob_pred, prob_true, 's-', label=self.best_model_name, linewidth=2, markersize=8)
            ax2.plot([0, 1], [0, 1], 'k--', label='Perfect Calibration')
            ax2.set_xlabel('Mean Predicted Probability')
            ax2.set_ylabel('Fraction of Positives')
            ax2.set_title('Calibration Curve')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
            
            # Prediction confidence vs accuracy
            ax3 = axes[0, 2]
            bin_boundaries = np.linspace(0, 1, 11)
            bin_lowers = bin_boundaries[:-1]
            bin_uppers = bin_boundaries[1:]
            accuracies = []
            confidences = []
            
            for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
                in_bin = (y_prob > bin_lower) & (y_prob <= bin_upper)
                prop_in_bin = in_bin.mean()
                if prop_in_bin > 0:
                    accuracy_in_bin = (y_pred[in_bin] == self.y_test[in_bin]).mean()
                    avg_confidence_in_bin = y_prob[in_bin].mean()
                    accuracies.append(accuracy_in_bin)
                    confidences.append(avg_confidence_in_bin)
            
            ax3.bar(range(len(accuracies)), accuracies, alpha=0.7, color='steelblue', label='Accuracy')
            ax3.plot(range(len(confidences)), confidences, 'ro-', label='Avg Confidence')
            ax3.set_xlabel('Probability Bin')
            ax3.set_ylabel('Score')
            ax3.set_title('Accuracy vs Confidence by Bin')
            ax3.legend()
            ax3.grid(True, alpha=0.3)
        
        # Error analysis by feature
        ax4 = axes[1, 0]
        test_df = self.df_filtered.iloc[-len(self.y_test):].copy()
        test_df['Error'] = errors
        
        # Select a few key features
        for feature in ['RSI', 'Volatility_20', 'Volume_Ratio']:
            feature_bins = pd.qcut(test_df[feature], q=5, duplicates='drop')
            error_by_feature = test_df.groupby(feature_bins)['Error'].mean()
            ax4.plot(range(len(error_by_feature)), error_by_feature.values, 'o-', label=feature, linewidth=2)
        
        ax4.set_xlabel('Feature Quintile')
        ax4.set_ylabel('Error Rate')
        ax4.set_title('Error Rate by Feature Quintile')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        # Confusion matrix normalized
        ax5 = axes[1, 1]
        cm = confusion_matrix(self.y_test, y_pred)
        cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        sns.heatmap(cm_norm, annot=True, fmt='.2%', cmap='Blues', ax=ax5,
                   xticklabels=['Down', 'Up'], yticklabels=['Down', 'Up'])
        ax5.set_title('Normalized Confusion Matrix')
        ax5.set_xlabel('Predicted')
        ax5.set_ylabel('Actual')
        
        # Class-wise metrics
        ax6 = axes[1, 2]
        report = self.results[self.best_model_name]['classification_report']
        classes = ['0', '1']
        metrics = ['precision', 'recall', 'f1-score']
        
        x = np.arange(len(classes))
        width = 0.25
        
        for i, metric in enumerate(metrics):
            values = [report[cls][metric] for cls in classes]
            ax6.bar(x + i*width, values, width, label=metric.title())
        
        ax6.set_ylabel('Score')
        ax6.set_title('Class-wise Performance Metrics')
        ax6.set_xticks(x + width)
        ax6.set_xticklabels(['Down (0)', 'Up (1)'])
        ax6.legend()
        ax6.set_ylim(0, 1)
        ax6.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        plt.savefig(self.viz_dir / '13_residual_analysis.png', dpi=150, bbox_inches='tight')
        plt.close()
        
    def plot_trading_simulation(self):
        """14. Simple trading strategy simulation based on predictions"""
        logger.info("Generating trading simulation visualization...")
        
        # Get test data
        test_df = self.df_filtered.iloc[-len(self.y_test):].copy()
        test_df['Predicted'] = self.results[self.best_model_name]['y_pred']
        test_df['Actual'] = self.y_test
        
        # Simple strategy: Buy when predicted up, Sell when predicted down
        test_df['Position'] = test_df['Predicted'].shift(1)  # Position for next period
        test_df['Strategy_Return'] = test_df['Position'] * test_df['Close'].pct_change()
        test_df['Cumulative_Return'] = (1 + test_df['Strategy_Return']).cumprod()
        test_df['Buy_Hold_Return'] = (1 + test_df['Close'].pct_change()).cumprod()
        
        fig, axes = plt.subplots(3, 1, figsize=(16, 12), sharex=True)
        
        # Price and positions
        ax1 = axes[0]
        ax1.plot(test_df['Date'], test_df['Close'], color='black', linewidth=1, label='Close')
        
        long_positions = test_df[test_df['Position'] == 1]
        short_positions = test_df[test_df['Position'] == 0]
        
        ax1.scatter(long_positions['Date'], long_positions['Close'], color='green', 
                   marker='^', s=30, alpha=0.6, label='Long Signal')
        ax1.scatter(short_positions['Date'], short_positions['Close'], color='red', 
                   marker='v', s=30, alpha=0.6, label='Short Signal')
        
        ax1.set_ylabel('Price (USD)')
        ax1.set_title('Trading Strategy Signals', fontsize=14, fontweight='bold')
        ax1.legend(loc='upper left')
        ax1.grid(True, alpha=0.3)
        
        # Cumulative returns
        ax2 = axes[1]
        ax2.plot(test_df['Date'], test_df['Cumulative_Return'], 
                color='blue', linewidth=2, label='Strategy')
        ax2.plot(test_df['Date'], test_df['Buy_Hold_Return'], 
                color='gray', linewidth=2, label='Buy & Hold', linestyle='--')
        ax2.set_ylabel('Cumulative Return')
        ax2.set_title('Strategy Performance vs Buy & Hold')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Rolling Sharpe ratio
        ax3 = axes[2]
        window = 50
        rolling_returns = test_df['Strategy_Return'].rolling(window=window)
        rolling_sharpe = (rolling_returns.mean() / rolling_returns.std()) * np.sqrt(252 * 24 * 4)  # Annualized for 15min data
        
        ax3.plot(test_df['Date'], rolling_sharpe, color='purple', linewidth=2)
        ax3.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
        ax3.axhline(y=1, color='green', linestyle='--', alpha=0.5, label='Sharpe = 1')
        ax3.set_ylabel('Sharpe Ratio')
        ax3.set_xlabel('Date')
        ax3.set_title(f'Rolling Sharpe Ratio ({window}-period)')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.viz_dir / '14_trading_simulation.png', dpi=150, bbox_inches='tight')
        plt.close()
        
        # Print trading statistics
        total_return = test_df['Cumulative_Return'].iloc[-1] - 1
        buy_hold_return = test_df['Buy_Hold_Return'].iloc[-1] - 1
        sharpe = (test_df['Strategy_Return'].mean() / test_df['Strategy_Return'].std()) * np.sqrt(252 * 24 * 4)
        max_drawdown = (test_df['Cumulative_Return'] / test_df['Cumulative_Return'].cummax() - 1).min()
        
        logger.info("Trading Simulation Results:")
        logger.info(f"  Strategy Return: {total_return:.2%}")
        logger.info(f"  Buy & Hold Return: {buy_hold_return:.2%}")
        logger.info(f"  Sharpe Ratio: {sharpe:.2f}")
        logger.info(f"  Max Drawdown: {max_drawdown:.2%}")
        
    def plot_hyperparameter_analysis(self):
        """15. Hyperparameter tuning results visualization"""
        logger.info("Generating hyperparameter analysis visualization...")
        
        # This would require storing all CV results - simplified version
        fig, axes = plt.subplots(2, 2, figsize=(16, 10))
        
        # Model complexity vs performance (using max_depth as proxy for tree models)
        ax1 = axes[0, 0]
        tree_models = ['Random Forest', 'Gradient Boosting']
        if 'XGBoost' in self.models:
            tree_models.append('XGBoost')
        
        for model_name in tree_models:
            if model_name in self.results:
                params = self.results[model_name]['best_params']
                score = self.results[model_name]['f1_score']
                if 'max_depth' in params:
                    ax1.scatter(params['max_depth'], score, s=200, label=model_name, alpha=0.7)
        
        ax1.set_xlabel('Max Depth')
        ax1.set_ylabel('F1 Score')
        ax1.set_title('Model Complexity vs Performance')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Learning rate analysis for boosting models
        ax2 = axes[0, 1]
        for model_name in ['Gradient Boosting', 'XGBoost']:
            if model_name in self.results and 'learning_rate' in self.results[model_name]['best_params']:
                lr = self.results[model_name]['best_params']['learning_rate']
                score = self.results[model_name]['f1_score']
                ax2.scatter(lr, score, s=200, label=model_name, alpha=0.7)
        
        ax2.set_xlabel('Learning Rate')
        ax2.set_ylabel('F1 Score')
        ax2.set_title('Learning Rate vs Performance')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Regularization analysis (C parameter for SVM and LR)
        ax3 = axes[1, 0]
        for model_name in ['Logistic Regression', 'SVM']:
            if model_name in self.results and 'C' in self.results[model_name]['best_params']:
                c = self.results[model_name]['best_params']['C']
                score = self.results[model_name]['f1_score']
                ax3.scatter(c, score, s=200, label=model_name, alpha=0.7)
        
        ax3.set_xlabel('C (Regularization Strength)')
        ax3.set_ylabel('F1 Score')
        ax3.set_title('Regularization vs Performance')
        ax3.set_xscale('log')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # CV Score distribution
        ax4 = axes[1, 1]
        cv_means = [self.results[m]['cv_f1_mean'] for m in self.results.keys()]
        cv_stds = [self.results[m]['cv_f1_std'] for m in self.results.keys()]
        
        ax4.errorbar(range(len(self.results)), cv_means, yerr=cv_stds, 
                    fmt='o', capsize=5, capthick=2, markersize=8)
        ax4.set_xticks(range(len(self.results)))
        ax4.set_xticklabels(self.results.keys(), rotation=45, ha='right')
        ax4.set_ylabel('CV F1 Score')
        ax4.set_title('Cross-Validation Scores with Std Dev')
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.viz_dir / '15_hyperparameter_analysis.png', dpi=150, bbox_inches='tight')
        plt.close()

    def generate_all_visualizations(self):
        """Generate all enhanced visualizations"""
        logger.info("="*60)
        logger.info("Generating Enhanced Visualizations")
        logger.info("="*60)
        
        self.plot_price_overview()
        self.plot_fibonacci_analysis()
        self.plot_smc_analysis()
        self.plot_volume_analysis()
        self.plot_correlation_analysis()
        self.plot_feature_distributions()
        self.plot_dimensionality_reduction()
        self.plot_learning_curves()
        self.plot_model_comparison()
        self.plot_confusion_matrices_detailed()
        self.plot_feature_importance_detailed()
        self.plot_prediction_analysis()
        self.plot_residual_analysis()
        self.plot_trading_simulation()
        self.plot_hyperparameter_analysis()
        
        logger.info("All visualizations saved to %s", self.viz_dir)
        return self
    
    def save_model(self):
        """Save the best trained model"""
        model_path = self.output_dir / 'best_model.pkl'
        with open(model_path, 'wb') as f:
            pickle.dump({
                'model': self.best_model,
                'scaler': self.scaler,
                'feature_names': self.feature_names,
                'model_name': self.best_model_name
            }, f)
        logger.info("Best model saved to %s", model_path)
        return self
    
    def save_results_csv(self):
        """Save evaluation results to CSV"""
        results_df = pd.DataFrame([
            {
                'Model': name,
                'Accuracy': metrics['accuracy'],
                'F1_Score': metrics['f1_score'],
                'Precision': metrics['precision'],
                'Recall': metrics['recall'],
                'CV_F1_Mean': metrics['cv_f1_mean'],
                'CV_F1_Std': metrics['cv_f1_std'],
                'Best_Params': json.dumps(metrics['best_params'])
            }
            for name, metrics in self.results.items()
        ])
        
        results_df = results_df.sort_values('F1_Score', ascending=False)
        csv_path = self.output_dir / 'model_evaluation_results.csv'
        results_df.to_csv(csv_path, index=False)
        logger.info("Results saved to %s", csv_path)
        return self
    
    def predict(self, new_data: pd.DataFrame) -> np.ndarray:
        """Make predictions on new data"""
        if self.best_model is None:
            raise ValueError("Model not trained yet. Run the pipeline first.")
        
        X = new_data[self.feature_names].values
        X_scaled = self.scaler.transform(X)
        return self.best_model.predict(X_scaled)
    
    def run_pipeline(self):
        """Execute the complete pipeline with all visualizations"""
        logger.info("="*60)
        logger.info("XAU/USD ML Pipeline - Starting")
        logger.info("="*60)
        
        (self
            .load_data()
            .filter_data(2022, 2025)
            .preprocess_data()
            .calculate_fibonacci_levels()
            .calculate_smc_signals()
            .calculate_volume_features()
            .calculate_technical_indicators()
            .create_target_variable()
            .prepare_features()
            .train_models()
            .select_best_model()
            .generate_all_visualizations()  # Enhanced visualization method
            .save_model()
            .save_results_csv()
        )
        
        logger.info("="*60)
        logger.info("Pipeline completed successfully!")
        logger.info("Best Model: %s", self.best_model_name)
        logger.info("Accuracy: %.2f%%", self.results[self.best_model_name]['accuracy'] * 100)
        logger.info("="*60)
        
        return self


def format_classification_report(report_dict):
    """Format classification report dictionary for printing"""
    lines = []
    lines.append(f"{'Class':<10} {'Precision':<10} {'Recall':<10} {'F1-Score':<10} {'Support':<10}")
    lines.append("-" * 50)
    
    for cls in ['0', '1']:
        if cls in report_dict:
            metrics = report_dict[cls]
            lines.append(f"{cls:<10} {metrics['precision']:<10.4f} {metrics['recall']:<10.4f} "
                        f"{metrics['f1-score']:<10.4f} {metrics['support']:<10.0f}")
    
    lines.append("-" * 50)
    lines.append(f"{'Accuracy':<10} {'':<10} {'':<10} {report_dict['accuracy']:<10.4f} {'':<10}")
    
    for avg in ['macro avg', 'weighted avg']:
        if avg in report_dict:
            metrics = report_dict[avg]
            lines.append(f"{avg:<10} {metrics['precision']:<10.4f} {metrics['recall']:<10.4f} "
                        f"{metrics['f1-score']:<10.4f} {metrics['support']:<10.0f}")
    
    return "\n".join(lines)



if __name__ == "__main__":
    # Run the pipeline
    pipeline = XAUPipeline('/mnt/kimi/upload/XAU_15m_data.csv')
    pipeline.run_pipeline()
    
    # Print final results
    print("\n" + "="*60)
    print("FINAL RESULTS")
    print("="*60)
    print(f"Best Model: {pipeline.best_model_name}")
    print(f"Accuracy: {pipeline.results[pipeline.best_model_name]['accuracy']:.2%}")
    print(f"F1-Score: {pipeline.results[pipeline.best_model_name]['f1_score']:.4f}")
    print("="*60)
    
    # Print classification report for best model
    print(f"\nClassification Report - {pipeline.best_model_name}:")
    print(format_classification_report(pipeline.results[pipeline.best_model_name]['classification_report']))