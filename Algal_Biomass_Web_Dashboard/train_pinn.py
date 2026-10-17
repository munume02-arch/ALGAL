"""
Training script for Physics-Informed Neural Network (PINN) and baseline models.

This script trains:
1. Physics-Informed Neural Network (PINN) - with full physics loss
2. Data-Only Neural Network - no physics constraints
3. PINN with weak physics - ablation study
4. Simple baseline (Random Forest)

Tests robustness under:
- Complete data
- 20% sparse data
- 40% sparse data
- Noisy observations
"""

import numpy as np
import pandas as pd
import torch
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import json
import os
from pinn_model import (
    PINN, prepare_data, introduce_sparsity, introduce_noise,
    train_model, evaluate_model, save_model
)

# Set random seeds for reproducibility
np.random.seed(42)
torch.manual_seed(42)

# Configuration
CONFIG = {
    'data_path': 'data/algal_growth_data.csv',
    'output_dir': 'models',
    'plots_dir': 'plots',
    
    # Data split (chronological)
    'train_ratio': 0.6,  # Days 1-60
    'val_ratio': 0.2,    # Days 61-80
    'test_ratio': 0.2,   # Days 81-100
    
    # Training hyperparameters
    'epochs': 5000,
    'learning_rate': 0.001,
    'hidden_dims': [64, 64, 32],
    
    # Loss weights
    'lambda_data': 1.0,
    'lambda_phys_strong': 0.5,   # Strong physics
    'lambda_phys_weak': 0.05,    # Weak physics
    'lambda_ic': 1.0,
    'lambda_reg': 0.0001,
    
    # Robustness tests
    'sparsity_ratios': [0.0, 0.2, 0.4],
    'noise_levels': [0.0, 0.1, 0.2],
}


def plot_training_curves(histories, labels, save_path):
    """
    Plot training and validation loss curves.
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    for history, label in zip(histories, labels):
        # Total loss
        axes[0, 0].plot(history['train_loss'], label=f'{label} - Train', alpha=0.7)
        axes[0, 0].plot(history['val_loss'], label=f'{label} - Val', linestyle='--', alpha=0.7)
        
        # Data loss
        axes[0, 1].plot(history['data_loss'], label=label)
        
        # Physics loss
        if 'phys_loss' in history and max(history['phys_loss']) > 0:
            axes[1, 0].plot(history['phys_loss'], label=label)
        
        # IC loss
        axes[1, 1].plot(history['ic_loss'], label=label, alpha=0.7)
    
    axes[0, 0].set_xlabel('Epoch (×100)')
    axes[0, 0].set_ylabel('Total Loss')
    axes[0, 0].set_title('Training & Validation Loss')
    axes[0, 0].legend()
    axes[0, 0].set_yscale('log')
    axes[0, 0].grid(True, alpha=0.3)
    
    axes[0, 1].set_xlabel('Epoch (×100)')
    axes[0, 1].set_ylabel('Data Loss (MSE)')
    axes[0, 1].set_title('Data Loss Component')
    axes[0, 1].legend()
    axes[0, 1].set_yscale('log')
    axes[0, 1].grid(True, alpha=0.3)
    
    axes[1, 0].set_xlabel('Epoch (×100)')
    axes[1, 0].set_ylabel('Physics Loss')
    axes[1, 0].set_title('Physics Residual Loss')
    axes[1, 0].legend()
    axes[1, 0].set_yscale('log')
    axes[1, 0].grid(True, alpha=0.3)
    
    axes[1, 1].set_xlabel('Epoch (×100)')
    axes[1, 1].set_ylabel('Initial Condition Loss')
    axes[1, 1].set_title('Initial Condition Loss')
    axes[1, 1].legend()
    axes[1, 1].set_yscale('log')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Training curves saved to {save_path}")


def plot_predictions(models_results, test_df, save_path):
    """
    Plot predictions vs actual for all models.
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    days_test = test_df['Day'].values
    
    for idx, (model_name, results) in enumerate(models_results.items()):
        ax = axes[idx // 2, idx % 2]
        
        y_true = results['y_true'].flatten()
        y_pred = results['y_pred'].flatten()
        
        # Scatter plot
        ax.scatter(days_test, y_true, label='Measured', alpha=0.6, s=50, color='blue')
        ax.scatter(days_test, y_pred, label='Predicted', alpha=0.6, s=50, color='red', marker='x')
        ax.plot(days_test, y_true, linestyle='--', alpha=0.3, color='blue')
        ax.plot(days_test, y_pred, linestyle='-', alpha=0.6, color='red')
        
        ax.set_xlabel('Day', fontsize=11)
        ax.set_ylabel('Biomass (g/L)', fontsize=11)
        ax.set_title(f'{model_name}\nRMSE={results["rmse"]:.4f}, R²={results["r2"]:.4f}', fontsize=12)
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Predictions plot saved to {save_path}")


def plot_physics_residual(model, data_dict, save_path):
    """
    Calculate and plot physics residual to verify physics loss is meaningful.
    """
    if not model.use_physics:
        print("Model does not use physics - skipping residual plot")
        return
    
    X_test, y_test = data_dict['test']
    x_scaler, y_scaler = data_dict['scalers']
    
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32).requires_grad_(True)
    
    model.eval()
    with torch.enable_grad():
        y_pred = model(X_test_tensor)
        
        # Calculate physics residual
        residuals = []
        for i in range(len(X_test_tensor)):
            x_single = X_test_tensor[i:i+1]
            y_single = y_pred[i:i+1]
            
            # Get residual
            dB_dDay = torch.autograd.grad(
                outputs=y_single,
                inputs=x_single,
                create_graph=True,
                retain_graph=True
            )[0][0, 0]
            
            # Denormalize
            Day = x_single[0, 0].item() * x_scaler.scale_[0] + x_scaler.mean_[0]
            T = x_single[0, 1].item() * x_scaler.scale_[1] + x_scaler.mean_[1]
            PAR = x_single[0, 2].item() * x_scaler.scale_[2] + x_scaler.mean_[2]
            pH = x_single[0, 3].item() * x_scaler.scale_[3] + x_scaler.mean_[3]
            N = x_single[0, 5].item() * x_scaler.scale_[5] + x_scaler.mean_[5]
            CO2 = x_single[0, 6].item() * x_scaler.scale_[6] + x_scaler.mean_[6]
            DO = x_single[0, 7].item() * x_scaler.scale_[7] + x_scaler.mean_[7]
            B = y_single.item() * y_scaler.scale_[0] + y_scaler.mean_[0]
            
            dB_dt = dB_dDay.item() * (y_scaler.scale_[0] / x_scaler.scale_[0])
            
            # Physics prediction
            T_t = torch.tensor([[T]], dtype=torch.float32)
            PAR_t = torch.tensor([[PAR]], dtype=torch.float32)
            pH_t = torch.tensor([[pH]], dtype=torch.float32)
            N_t = torch.tensor([[N]], dtype=torch.float32)
            CO2_t = torch.tensor([[CO2]], dtype=torch.float32)
            DO_t = torch.tensor([[DO]], dtype=torch.float32)
            B_t = torch.tensor([[B]], dtype=torch.float32)
            
            dB_dt_phys = model.physics.growth_equation(B_t, T_t, PAR_t, pH_t, N_t, CO2_t, DO_t).item()
            
            residual = dB_dt - dB_dt_phys
            residuals.append(residual)
    
    test_df = data_dict['dataframes'][2]
    days = test_df['Day'].values
    
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(days, residuals, marker='o', linestyle='-', alpha=0.7)
    plt.axhline(y=0, color='r', linestyle='--', alpha=0.5)
    plt.xlabel('Day')
    plt.ylabel('Physics Residual\n(dB/dt_NN - dB/dt_physics)')
    plt.title('Physics Residual Over Time')
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 2, 2)
    plt.hist(residuals, bins=20, alpha=0.7, edgecolor='black')
    plt.xlabel('Residual Value')
    plt.ylabel('Frequency')
    plt.title(f'Residual Distribution\nMean={np.mean(residuals):.6f}, Std={np.std(residuals):.6f}')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Physics residual plot saved to {save_path}")


def plot_comparison_metrics(results_df, save_path):
    """
    Plot comparison of all models across metrics.
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    metrics = ['rmse', 'mae', 'r2']
    titles = ['RMSE (lower is better)', 'MAE (lower is better)', 'R² (higher is better)']
    
    for ax, metric, title in zip(axes, metrics, titles):
        data = results_df.pivot(index='model', columns='condition', values=metric)
        data.plot(kind='bar', ax=ax, width=0.8)
        ax.set_title(title, fontsize=12)
        ax.set_ylabel(metric.upper())
        ax.set_xlabel('Model')
        ax.legend(title='Condition')
        ax.grid(True, alpha=0.3, axis='y')
        ax.tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Comparison metrics plot saved to {save_path}")


def train_random_forest_baseline(data_dict):
    """
    Train Random Forest as a simple baseline.
    """
    print("\n" + "="*60)
    print("Training Random Forest Baseline")
    print("="*60)
    
    X_train, y_train = data_dict['train']
    X_test, y_test = data_dict['test']
    x_scaler, y_scaler = data_dict['scalers']
    
    # Random Forest doesn't need normalized data, but we'll use it for consistency
    rf = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train.ravel())
    
    # Predict
    y_pred_norm = rf.predict(X_test).reshape(-1, 1)
    
    # Denormalize
    y_pred = y_scaler.inverse_transform(y_pred_norm)
    y_true = y_scaler.inverse_transform(y_test)
    
    # Metrics
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    mape = np.mean(np.abs((y_true - y_pred) / (y_true + 1e-8))) * 100
    
    print(f"Random Forest - RMSE: {rmse:.4f}, MAE: {mae:.4f}, R²: {r2:.4f}")
    
    return {
        'rmse': rmse,
        'mae': mae,
        'r2': r2,
        'mape': mape,
        'y_true': y_true,
        'y_pred': y_pred
    }


def main():
    """
    Main training pipeline.
    """
    # Create output directories
    os.makedirs(CONFIG['output_dir'], exist_ok=True)
    os.makedirs(CONFIG['plots_dir'], exist_ok=True)
    
    print("="*70)
    print("PHYSICS-INFORMED NEURAL NETWORK FOR ALGAL BIOMASS ESTIMATION")
    print("="*70)
    print("\nFYP: Robust Biomass Estimation under Sparse and Noisy Sensor Data")
    print("\nThis is a SYNTHETIC dataset validation - not experimental data")
    print("="*70)
    
    # Load data
    print("\nLoading data...")
    df = pd.read_csv(CONFIG['data_path'])
    print(f"Dataset shape: {df.shape}")
    print(f"Date range: Day {df.Day.min():.0f} to Day {df.Day.max():.0f}")
    
    # Prepare data with chronological split
    data_dict = prepare_data(df, CONFIG['train_ratio'], CONFIG['val_ratio'])
    
    # ========== EXPERIMENT 1: Complete Data ==========
    print("\n" + "="*60)
    print("EXPERIMENT 1: Complete Data (No Missing Values)")
    print("="*60)
    
    # 1A. PINN with strong physics
    print("\n1A. Training PINN with STRONG physics loss...")
    pinn_strong = PINN(input_dim=8, hidden_dims=CONFIG['hidden_dims'], use_physics=True)
    history_pinn_strong = train_model(
        pinn_strong, data_dict,
        epochs=CONFIG['epochs'],
        lr=CONFIG['learning_rate'],
        lambda_data=CONFIG['lambda_data'],
        lambda_phys=CONFIG['lambda_phys_strong'],
        lambda_ic=CONFIG['lambda_ic'],
        lambda_reg=CONFIG['lambda_reg']
    )
    results_pinn_strong = evaluate_model(pinn_strong, data_dict)
    print(f"PINN (Strong Physics) - RMSE: {results_pinn_strong['rmse']:.4f}, "
          f"MAE: {results_pinn_strong['mae']:.4f}, R²: {results_pinn_strong['r2']:.4f}")
    
    # 1B. PINN with weak physics (ablation)
    print("\n1B. Training PINN with WEAK physics loss (ablation)...")
    pinn_weak = PINN(input_dim=8, hidden_dims=CONFIG['hidden_dims'], use_physics=True)
    history_pinn_weak = train_model(
        pinn_weak, data_dict,
        epochs=CONFIG['epochs'],
        lr=CONFIG['learning_rate'],
        lambda_data=CONFIG['lambda_data'],
        lambda_phys=CONFIG['lambda_phys_weak'],
        lambda_ic=CONFIG['lambda_ic'],
        lambda_reg=CONFIG['lambda_reg']
    )
    results_pinn_weak = evaluate_model(pinn_weak, data_dict)
    print(f"PINN (Weak Physics) - RMSE: {results_pinn_weak['rmse']:.4f}, "
          f"MAE: {results_pinn_weak['mae']:.4f}, R²: {results_pinn_weak['r2']:.4f}")
    
    # 1C. Data-only NN (no physics)
    print("\n1C. Training Data-Only Neural Network (NO physics)...")
    nn_data_only = PINN(input_dim=8, hidden_dims=CONFIG['hidden_dims'], use_physics=False)
    history_data_only = train_model(
        nn_data_only, data_dict,
        epochs=CONFIG['epochs'],
        lr=CONFIG['learning_rate'],
        lambda_data=CONFIG['lambda_data'],
        lambda_phys=0.0,
        lambda_ic=CONFIG['lambda_ic'],
        lambda_reg=CONFIG['lambda_reg']
    )
    results_data_only = evaluate_model(nn_data_only, data_dict)
    print(f"Data-Only NN - RMSE: {results_data_only['rmse']:.4f}, "
          f"MAE: {results_data_only['mae']:.4f}, R²: {results_data_only['r2']:.4f}")
    
    # 1D. Random Forest baseline
    results_rf = train_random_forest_baseline(data_dict)
    
    # Plot training curves
    plot_training_curves(
        [history_pinn_strong, history_pinn_weak, history_data_only],
        ['PINN (Strong)', 'PINN (Weak)', 'Data-Only'],
        os.path.join(CONFIG['plots_dir'], 'training_curves.png')
    )
    
    # Plot predictions
    test_df = data_dict['dataframes'][2]
    models_results = {
        'PINN (Strong Physics)': results_pinn_strong,
        'PINN (Weak Physics)': results_pinn_weak,
        'Data-Only NN': results_data_only,
        'Random Forest': results_rf
    }
    plot_predictions(models_results, test_df, 
                    os.path.join(CONFIG['plots_dir'], 'predictions_comparison.png'))
    
    # Plot physics residual
    plot_physics_residual(pinn_strong, data_dict,
                         os.path.join(CONFIG['plots_dir'], 'physics_residual.png'))
    
    # ========== EXPERIMENT 2: Sparse Data ==========
    print("\n" + "="*60)
    print("EXPERIMENT 2: Robustness to Sparse Data")
    print("="*60)
    
    sparse_results = []
    
    for sparsity in CONFIG['sparsity_ratios']:
        if sparsity == 0:
            continue  # Already tested
        
        print(f"\nTesting with {int(sparsity*100)}% missing data...")
        
        # Create sparse mask
        X_train, y_train = data_dict['train']
        sparse_mask = introduce_sparsity(y_train, sparsity_ratio=sparsity, seed=42)
        print(f"Training with {sparse_mask.sum()}/{len(sparse_mask)} observations")
        
        # Train PINN with sparse data
        pinn_sparse = PINN(input_dim=8, hidden_dims=CONFIG['hidden_dims'], use_physics=True)
        _ = train_model(
            pinn_sparse, data_dict,
            epochs=CONFIG['epochs'],
            lr=CONFIG['learning_rate'],
            lambda_data=CONFIG['lambda_data'],
            lambda_phys=CONFIG['lambda_phys_strong'],
            lambda_ic=CONFIG['lambda_ic'],
            lambda_reg=CONFIG['lambda_reg'],
            sparse_mask=sparse_mask
        )
        results_sparse_pinn = evaluate_model(pinn_sparse, data_dict)
        
        # Train data-only with sparse data
        nn_sparse = PINN(input_dim=8, hidden_dims=CONFIG['hidden_dims'], use_physics=False)
        _ = train_model(
            nn_sparse, data_dict,
            epochs=CONFIG['epochs'],
            lr=CONFIG['learning_rate'],
            lambda_data=CONFIG['lambda_data'],
            lambda_phys=0.0,
            lambda_ic=CONFIG['lambda_ic'],
            lambda_reg=CONFIG['lambda_reg'],
            sparse_mask=sparse_mask
        )
        results_sparse_nn = evaluate_model(nn_sparse, data_dict)
        
        print(f"PINN (Sparse {int(sparsity*100)}%) - RMSE: {results_sparse_pinn['rmse']:.4f}, R²: {results_sparse_pinn['r2']:.4f}")
        print(f"Data-Only (Sparse {int(sparsity*100)}%) - RMSE: {results_sparse_nn['rmse']:.4f}, R²: {results_sparse_nn['r2']:.4f}")
        
        sparse_results.append({
            'sparsity': sparsity,
            'pinn': results_sparse_pinn,
            'data_only': results_sparse_nn
        })
    
    # ========== EXPERIMENT 3: Noisy Data ==========
    print("\n" + "="*60)
    print("EXPERIMENT 3: Robustness to Noisy Data")
    print("="*60)
    
    for noise in CONFIG['noise_levels']:
        if noise == 0:
            continue
        
        print(f"\nTesting with {int(noise*100)}% noise level...")
        
        # Add noise to training data
        X_train, y_train = data_dict['train']
        y_train_noisy = introduce_noise(y_train, noise_level=noise, seed=42)
        
        data_dict_noisy = data_dict.copy()
        data_dict_noisy['train'] = (X_train, y_train_noisy)
        
        # Train PINN with noisy data
        pinn_noisy = PINN(input_dim=8, hidden_dims=CONFIG['hidden_dims'], use_physics=True)
        _ = train_model(
            pinn_noisy, data_dict_noisy,
            epochs=CONFIG['epochs'],
            lr=CONFIG['learning_rate'],
            lambda_data=CONFIG['lambda_data'],
            lambda_phys=CONFIG['lambda_phys_strong'],
            lambda_ic=CONFIG['lambda_ic'],
            lambda_reg=CONFIG['lambda_reg']
        )
        results_noisy_pinn = evaluate_model(pinn_noisy, data_dict)
        
        # Train data-only with noisy data
        nn_noisy = PINN(input_dim=8, hidden_dims=CONFIG['hidden_dims'], use_physics=False)
        _ = train_model(
            nn_noisy, data_dict_noisy,
            epochs=CONFIG['epochs'],
            lr=CONFIG['learning_rate'],
            lambda_data=CONFIG['lambda_data'],
            lambda_phys=0.0,
            lambda_ic=CONFIG['lambda_ic'],
            lambda_reg=CONFIG['lambda_reg']
        )
        results_noisy_nn = evaluate_model(nn_noisy, data_dict)
        
        print(f"PINN (Noise {int(noise*100)}%) - RMSE: {results_noisy_pinn['rmse']:.4f}, R²: {results_noisy_pinn['r2']:.4f}")
        print(f"Data-Only (Noise {int(noise*100)}%) - RMSE: {results_noisy_nn['rmse']:.4f}, R²: {results_noisy_nn['r2']:.4f}")
    
    # ========== Summary Results ==========
    print("\n" + "="*70)
    print("FINAL RESULTS SUMMARY")
    print("="*70)
    
    results_summary = []
    for name, res in models_results.items():
        results_summary.append({
            'model': name,
            'condition': 'Complete',
            'rmse': res['rmse'],
            'mae': res['mae'],
            'r2': res['r2'],
            'mape': res['mape']
        })
    
    for sparse_res in sparse_results:
        results_summary.append({
            'model': 'PINN',
            'condition': f'Sparse {int(sparse_res["sparsity"]*100)}%',
            'rmse': sparse_res['pinn']['rmse'],
            'mae': sparse_res['pinn']['mae'],
            'r2': sparse_res['pinn']['r2'],
            'mape': sparse_res['pinn']['mape']
        })
        results_summary.append({
            'model': 'Data-Only NN',
            'condition': f'Sparse {int(sparse_res["sparsity"]*100)}%',
            'rmse': sparse_res['data_only']['rmse'],
            'mae': sparse_res['data_only']['mae'],
            'r2': sparse_res['data_only']['r2'],
            'mape': sparse_res['data_only']['mape']
        })
    
    results_df = pd.DataFrame(results_summary)
    print("\n", results_df.to_string(index=False))
    
    # Save results
    results_df.to_csv(os.path.join(CONFIG['output_dir'], 'results_summary.csv'), index=False)
    
    # Plot comparison
    plot_comparison_metrics(results_df, 
                           os.path.join(CONFIG['plots_dir'], 'metrics_comparison.png'))
    
    # Save best models
    save_model(pinn_strong, data_dict['scalers'], history_pinn_strong, 
              results_pinn_strong, os.path.join(CONFIG['output_dir'], 'pinn_strong'))
    save_model(nn_data_only, data_dict['scalers'], history_data_only, 
              results_data_only, os.path.join(CONFIG['output_dir'], 'data_only'))
    
    print("\n" + "="*70)
    print("Training complete! Check 'models/' and 'plots/' directories for outputs.")
    print("="*70)
    
    # Print key findings
    print("\n🔬 KEY FINDINGS:")
    print(f"1. PINN (Strong Physics) achieved RMSE: {results_pinn_strong['rmse']:.4f}, R²: {results_pinn_strong['r2']:.4f}")
    print(f"2. Data-Only NN achieved RMSE: {results_data_only['rmse']:.4f}, R²: {results_data_only['r2']:.4f}")
    print(f"3. Physics loss {'IMPROVED' if results_pinn_strong['rmse'] < results_data_only['rmse'] else 'DID NOT IMPROVE'} performance")
    
    if len(sparse_results) > 0:
        last_sparse = sparse_results[-1]
        improvement = (last_sparse['data_only']['rmse'] - last_sparse['pinn']['rmse']) / last_sparse['data_only']['rmse'] * 100
        print(f"4. Under sparse data, PINN showed {improvement:.1f}% improvement over data-only")
    
    print("\n⚠️  IMPORTANT: This validation uses SYNTHETIC data. Real-world validation required.")


if __name__ == "__main__":
    main()
