"""Quick training script for demonstration"""
import sys
sys.stdout.reconfigure(line_buffering=True)

print("Starting training...", flush=True)

import numpy as np
import pandas as pd
import torch
from pinn_model import PINN, prepare_data, train_model, evaluate_model, save_model
import os

# Reduced epochs for faster training
CONFIG = {
    'data_path': 'data/algal_growth_data.csv',
    'epochs': 1000,  # Reduced from 5000
    'learning_rate': 0.001,
    'hidden_dims': [64, 64, 32],
}

print(f"Loading data from {CONFIG['data_path']}...", flush=True)
df = pd.read_csv(CONFIG['data_path'])
print(f"Data loaded: {len(df)} samples", flush=True)

print("Preparing data splits...", flush=True)
data_dict = prepare_data(df, train_ratio=0.6, val_ratio=0.2)

# Train PINN
print("\nTraining PINN with physics...", flush=True)
pinn = PINN(input_dim=8, hidden_dims=CONFIG['hidden_dims'], use_physics=True)
history_pinn = train_model(
    pinn, data_dict,
    epochs=CONFIG['epochs'],
    lr=CONFIG['learning_rate'],
    lambda_data=1.0,
    lambda_phys=0.5,
    lambda_ic=1.0,
    lambda_reg=0.0001
)
print("PINN training complete!", flush=True)

results_pinn = evaluate_model(pinn, data_dict)
print(f"\nPINN Results:", flush=True)
print(f"  RMSE: {results_pinn['rmse']:.4f}", flush=True)
print(f"  MAE: {results_pinn['mae']:.4f}", flush=True)
print(f"  R²: {results_pinn['r2']:.4f}", flush=True)

# Train Data-Only NN
print("\nTraining Data-Only NN...", flush=True)
nn = PINN(input_dim=8, hidden_dims=CONFIG['hidden_dims'], use_physics=False)
history_nn = train_model(
    nn, data_dict,
    epochs=CONFIG['epochs'],
    lr=CONFIG['learning_rate'],
    lambda_data=1.0,
    lambda_phys=0.0,
    lambda_ic=1.0,
    lambda_reg=0.0001
)
print("Data-Only NN training complete!", flush=True)

results_nn = evaluate_model(nn, data_dict)
print(f"\nData-Only NN Results:", flush=True)
print(f"  RMSE: {results_nn['rmse']:.4f}", flush=True)
print(f"  MAE: {results_nn['mae']:.4f}", flush=True)
print(f"  R²: {results_nn['r2']:.4f}", flush=True)

# Save models
print("\nSaving models...", flush=True)
os.makedirs('models', exist_ok=True)
save_model(pinn, data_dict['scalers'], history_pinn, results_pinn, 'models/pinn_strong')
save_model(nn, data_dict['scalers'], history_nn, results_nn, 'models/data_only')

print("\n" + "="*60, flush=True)
print("TRAINING COMPLETE!", flush=True)
print("="*60, flush=True)
print(f"\nComparison:", flush=True)
improvement = (results_nn['rmse'] - results_pinn['rmse']) / results_nn['rmse'] * 100
print(f"Physics-informed approach {'improved' if improvement > 0 else 'did not improve'} RMSE by {abs(improvement):.2f}%", flush=True)
print(f"\nModels saved to models/ directory", flush=True)
