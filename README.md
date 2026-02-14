# XAU/USD Gold Price Prediction

[![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://www.python.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3+-orange.svg)](https://scikit-learn.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

Machine Learning pipeline for predicting XAU/USD (Gold) price direction using 15-minute timeframe data. Built with advanced feature engineering including Fibonacci retracements, Smart Money Concepts (SMC), and volume-based indicators.

## 📊 Project Overview

| Metric | Value |
|--------|-------|
| **Best Model Accuracy** | 77.91% |
| **AUC Score** | 0.714 |
| **Dataset Period** | 2022-2025 |
| **Total Records** | 92,457 |
| **Features** | 8 engineered features |
| **Target** | Binary price direction (Up/Down) |

## 🎯 Objective

Predict whether gold price will increase by &gt;0.1% within the next 6 periods (1.5 hours) using historical OHLCV data and technical indicators.

## 🏗️ Project Structure
XAU_ML_Prediction/
├── data/
│   ├── raw/                    # Original 2004-2025 dataset
│   ├── processed/              # Filtered 2022-2025 data
│   └── features/               # Engineered feature sets
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_feature_engineering.ipynb
│   ├── 03_model_training.ipynb
│   └── 04_model_evaluation.ipynb
├── src/
│   ├── data_preprocessing.py
│   ├── feature_engineering.py
│   ├── models/
│   │   ├── random_forest.py
│   │   ├── xgboost_model.py
│   │   └── logistic_regression.py
│   └── evaluation.py
├── reports/
│   └── XAU_ML_Report.pdf       # Full analysis report
├── models/
│   └── best_random_forest.pkl
├── config.yaml
├── requirements.txt
└── README.md

## 🔧 Features Engineered

### Technical Indicators
| Feature | Description |
|---------|-------------|
| `SMA_20` | 20-period Simple Moving Average |
| `RSI` | Relative Strength Index (14-period) |
| `Volume_MA` | 20-period Volume Moving Average |

### Price Action Features
- OHLC data (Open, High, Low, Close)
- Raw Volume
- Fibonacci retracement levels
- Smart Money Concepts (SMC) signals

### Feature Importance (Random Forest)
1. **Volume** (15.6%)
2. **SMA_20** (14.0%)
3. **Volume_MA** (13.7%)
4. **Low** (11.7%)
5. **Close** (11.5%)

## 🤖 Models Evaluated

| Model | Accuracy | F1-Score | AUC | Status |
|-------|----------|----------|-----|--------|
| **Random Forest** | **77.91%** | **0.4616** | **0.714** | ✅ Deployed |
| XGBoost | 68.2% | 0.462 | 0.698 | Tested |
| Logistic Regression | 62.1% | 0.389 | 0.642 | Baseline |

### Best Model Configuration
```python
RandomForestClassifier(
    n_estimators=100,
    max_depth=15,
    class_weight='balanced',
    random_state=42
)
              precision    recall  f1-score   support
Down (0)         0.81      0.78      0.79     13,457
Up (1)           0.46      0.50      0.48      5,035

Accuracy:                              0.78     18,492
Macro Avg:        0.63      0.64      0.64
Weighted Avg:     0.71      0.78      0.71

Key Insights
Class Imbalance Handled: 73% down vs 27% up moves
Strong Precision on Down Moves: 81% accuracy predicting declines
Balanced Recall: 50% capture rate on upside opportunities
Robust Generalization: Cross-validated stability confirmed
🚀 Quick Start
# Clone repository
git clone https://github.com/yourusername/XAU_ML_Prediction.git
cd XAU_ML_Prediction

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
from src.models.random_forest import GoldPricePredictor

# Initialize model
predictor = GoldPricePredictor()

# Load and preprocess data
df = predictor.load_data('data/processed/XAU_15min_2022_2025.csv')

# Generate features
features = predictor.engineer_features(df)

# Train or load pretrained model
model = predictor.train(features)  # or predictor.load_model('models/best_random_forest.pkl')

# Predict
prediction = predictor.predict(features)
# Output: 1 (Up) or 0 (Down) with probability
# Full pipeline execution
python src/train_pipeline.py --config config.yaml

# With custom parameters
python src/train_pipeline.py --start-date 2022-01-01 --test-size 0.2 --model random_forest
⚠️ Limitations & Risk Disclaimer
IMPORTANT: This project is for educational and research purposes only. Not financial advice.
Market Regime Risk: Model trained on 2022-2025 data; past performance ≠ future results
News Events: High-impact economic events can cause unpredictable price movements
Timeframe Constraints: 15-minute bars may miss sub-minute volatility dynamics
Class Imbalance: Model biased toward predicting downward moves (higher precision on Class 0)
🔮 Future Improvements
[ ] Deep Learning: Implement LSTM/Transformer architectures for sequential patterns
[ ] Advanced SMC: Add order blocks, fair value gaps, and liquidity sweep detection
[ ] Multi-Timeframe: Aggregate signals from 5min, 1H, and 4H timeframes
[ ] Sentiment Analysis: Integrate news sentiment and CFTC positioning data
[ ] Reinforcement Learning: Optimize entry/exit timing with RL agents
🛠️ Tech Stack
Python 3.9+
scikit-learn - Machine learning models
pandas/numpy - Data manipulation
matplotlib/seaborn - Visualization
TA-Lib - Technical indicators
Optuna - Hyperparameter optimization
📚 References
Murphy, J.J. Technical Analysis of the Financial Markets
Granger, C.W.J. Forecasting in Business and Economics
Lopez de Prado, M. Advances in Financial Machine Learning
