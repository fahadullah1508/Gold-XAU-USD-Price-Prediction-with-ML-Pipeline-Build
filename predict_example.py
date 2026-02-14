#!/usr/bin/env python3
"""
Example: How to use the trained model for predictions
"""

import pickle
import numpy as np
import pandas as pd

# Load the trained model
with open('best_model.pkl', 'rb') as f:
    saved_data = pickle.load(f)

model = saved_data['model']
scaler = saved_data['scaler']
feature_names = saved_data['feature_names']
model_name = saved_data['model_name']

print(f"Loaded model: {model_name}")
print(f"Features required: {feature_names}")

# Example: Create a new data point for prediction
# This should be a DataFrame with the same features used during training
new_data = pd.DataFrame({
    'Open': [4105],
    'High': [4108],
    'Low': [4099],
    'Close': [4105],
    'Volume_MA': [7019],
    'SMA_20': [4120],
    'RSI': [30.5],
    'Volume' : [1500]
        
})

# Ensure all required features are present
X_new = new_data[feature_names].values

# Scale the features
X_new_scaled = scaler.transform(X_new)

# Make prediction
prediction = model.predict(X_new_scaled)
probability = model.predict_proba(X_new_scaled)

print(f"\nPrediction: {'UP' if prediction[0] == 1 else 'DOWN'}")
print(f"Probability (Down): {probability[0][0]:.4f}")
print(f"Probability (Up): {probability[0][1]:.4f}")
