"""
Physics-Informed Neural Network for Algal Biomass Estimation

This module implements a GENUINE PINN that enforces biological growth dynamics
through physics-based loss functions.

Governing Equation:
    dB/dt = μ_eff * B * (1 - B/K)

where:
    B = biomass concentration (g/L)
    μ_eff = effective growth rate (1/day)
    K = carrying capacity (g/L)

Effective Growth Rate:
    μ_eff = μ_max × f_T × f_PAR × f_pH × f_N × f_CO2 × f_DO

Response Functions:
    - Temperature: Cardinal temperature model
    - Light (PAR): Monod saturation with photoinhibition
    - pH: Gaussian optimum response
    - Nitrogen: Monod kinetics
    - CO2: Monod kinetics
    - DO: Observational (not forced into physics unless justified)
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import json
from typing import Dict, Tuple, List
import os


class BiomassPhysics:
    """
    Encapsulates the biological growth model and response functions.
    """
    def __init__(self):
        # Growth parameters
        self.mu_max = nn.Parameter(torch.tensor(0.5))  # Maximum growth rate (1/day)
        self.K_capacity = nn.Parameter(torch.tensor(2.0))  # Carrying capacity (g/L)
        
        # Temperature response parameters (cardinal temps)
        self.T_min = nn.Parameter(torch.tensor(15.0))
        self.T_opt = nn.Parameter(torch.tensor(27.0))
        self.T_max = nn.Parameter(torch.tensor(35.0))
        
        # Light response parameters
        self.K_PAR = nn.Parameter(torch.tensor(100.0))  # Half-saturation constant
        self.K_I = nn.Parameter(torch.tensor(500.0))   # Photoinhibition constant
        
        # pH response parameters
        self.pH_opt = nn.Parameter(torch.tensor(8.0))
        self.pH_width = nn.Parameter(torch.tensor(1.0))
        
        # Nutrient parameters
        self.K_N = nn.Parameter(torch.tensor(5.0))     # Nitrogen half-saturation
        self.K_CO2 = nn.Parameter(torch.tensor(1.5))   # CO2 half-saturation
    
    def f_temperature(self, T):
        """
        Cardinal temperature model (bounded between 0 and 1).
        """
        # Avoid division by zero
        T = torch.clamp(T, min=self.T_min + 0.1, max=self.T_max - 0.1)
        
        numerator = (T - self.T_max) * (T - self.T_min)**2
        denominator = (self.T_opt - self.T_min) * (
            (self.T_opt - self.T_min) * (T - self.T_opt) -
            (self.T_opt - self.T_max) * (self.T_opt + self.T_min - 2*T)
        )
        
        # Add small epsilon to prevent division issues
        f_T = numerator / (denominator + 1e-6)
        return torch.clamp(f_T, min=0.0, max=1.0)
    
    def f_light(self, PAR):
        """
        Monod saturation with optional photoinhibition.
        """
        saturation = PAR / (self.K_PAR + PAR)
        inhibition = self.K_I / (self.K_I + PAR)
        return saturation * inhibition
    
    def f_pH(self, pH):
        """
        Gaussian optimum response centered at pH_opt.
        """
        return torch.exp(-0.5 * ((pH - self.pH_opt) / self.pH_width) ** 2)
    
    def f_nitrogen(self, N):
        """
        Monod kinetics for nitrogen limitation.
        """
        return N / (self.K_N + N)
    
    def f_CO2(self, CO2):
        """
        Monod kinetics for CO2 limitation.
        """
        return CO2 / (self.K_CO2 + CO2)
    
    def f_DO(self, DO):
        """
        Dissolved oxygen - currently observational.
        Could be modeled as inhibitory at very high levels or required at minimum levels.
        For now, we'll use a simple bounded response.
        """
        # Assume optimal DO around 6-8 mg/L
        DO_opt = 7.0
        DO_width = 3.0
        return torch.exp(-0.5 * ((DO - DO_opt) / DO_width) ** 2)
    
    def effective_growth_rate(self, T, PAR, pH, N, CO2, DO):
        """
        Calculate effective growth rate from environmental conditions.
        """
        mu_eff = (
            torch.abs(self.mu_max) *  # Ensure positive
            self.f_temperature(T) *
            self.f_light(PAR) *
            self.f_pH(pH) *
            self.f_nitrogen(N) *
            self.f_CO2(CO2) *
            self.f_DO(DO)
        )
        return mu_eff
    
    def growth_equation(self, B, T, PAR, pH, N, CO2, DO):
        """
        Logistic growth with environmental modulation:
        dB/dt = μ_eff * B * (1 - B/K)
        """
        mu_eff = self.effective_growth_rate(T, PAR, pH, N, CO2, DO)
        K = torch.abs(self.K_capacity)  # Ensure positive
        
        # Logistic growth term
        growth_term = mu_eff * B * (1.0 - B / K)
        
        return growth_term


class PINN(nn.Module):
    """
    Physics-Informed Neural Network for biomass estimation.
    """
    def __init__(self, input_dim=8, hidden_dims=[64, 64, 32], use_physics=True):
        super(PINN, self).__init__()
        
        self.use_physics = use_physics
        self.physics = BiomassPhysics() if use_physics else None
        
        # Neural network architecture
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.Tanh())
            prev_dim = hidden_dim
        
        # Output layer with Softplus to ensure positivity
        layers.append(nn.Linear(prev_dim, 1))
        
        self.network = nn.Sequential(*layers)
        self.softplus = nn.Softplus()
    
    def forward(self, x):
        """
        Forward pass through the network.
        x shape: (batch, 8) = [Day, T, PAR, pH, OD680, N, CO2, DO]
        """
        out = self.network(x)
        # Ensure positive biomass predictions
        return self.softplus(out)
    
    def physics_loss(self, x, B_pred, x_scaler, y_scaler):
        """
        Calculate physics-based loss using automatic differentiation.
        
        Args:
            x: Input tensor with requires_grad=True
            B_pred: Predicted biomass
            x_scaler: StandardScaler for inputs
            y_scaler: StandardScaler for biomass
        
        Returns:
            Physics residual loss
        """
        if not self.use_physics:
            return torch.tensor(0.0)
        
        # Extract environmental variables (denormalized)
        Day = x[:, 0:1] * x_scaler.scale_[0] + x_scaler.mean_[0]
        T = x[:, 1:2] * x_scaler.scale_[1] + x_scaler.mean_[1]
        PAR = x[:, 2:3] * x_scaler.scale_[2] + x_scaler.mean_[2]
        pH = x[:, 3:4] * x_scaler.scale_[3] + x_scaler.mean_[3]
        N = x[:, 5:6] * x_scaler.scale_[5] + x_scaler.mean_[5]
        CO2 = x[:, 6:7] * x_scaler.scale_[6] + x_scaler.mean_[6]
        DO = x[:, 7:8] * x_scaler.scale_[7] + x_scaler.mean_[7]
        
        # Denormalize biomass prediction
        B_denorm = B_pred * y_scaler.scale_[0] + y_scaler.mean_[0]
        
        # Calculate dB/dt using automatic differentiation
        # We need gradient of B with respect to Day
        dB_dDay = torch.autograd.grad(
            outputs=B_pred,
            inputs=x,
            grad_outputs=torch.ones_like(B_pred),
            create_graph=True,
            retain_graph=True
        )[0][:, 0:1]  # Extract gradient w.r.t. Day column
        
        # Scale the derivative properly
        dB_dt = dB_dDay * (y_scaler.scale_[0] / x_scaler.scale_[0])
        
        # Calculate physics-based dB/dt
        dB_dt_physics = self.physics.growth_equation(B_denorm, T, PAR, pH, N, CO2, DO)
        
        # Physics residual
        residual = dB_dt - dB_dt_physics
        
        return torch.mean(residual ** 2)


def prepare_data(df: pd.DataFrame, train_ratio=0.6, val_ratio=0.2):
    """
    Prepare data with chronological splitting (NO SHUFFLING).
    
    Args:
        df: DataFrame with columns [Day, Temperature, PAR, pH, OD680, Nitrogen_mg_L, CO2_mg_L, Dissolved_Oxygen_mg_L, Biomass]
        train_ratio: Proportion for training
        val_ratio: Proportion for validation
    
    Returns:
        Dictionaries containing train, val, test splits
    """
    # Sort by Day to ensure chronological order
    df = df.sort_values('Day').reset_index(drop=True)
    
    n_samples = len(df)
    train_size = int(n_samples * train_ratio)
    val_size = int(n_samples * val_ratio)
    
    # Chronological split
    train_df = df.iloc[:train_size]
    val_df = df.iloc[train_size:train_size + val_size]
    test_df = df.iloc[train_size + val_size:]
    
    print(f"\nChronological Split:")
    print(f"Training: Days {train_df.Day.min():.0f}-{train_df.Day.max():.0f} ({len(train_df)} samples)")
    print(f"Validation: Days {val_df.Day.min():.0f}-{val_df.Day.max():.0f} ({len(val_df)} samples)")
    print(f"Testing: Days {test_df.Day.min():.0f}-{test_df.Day.max():.0f} ({len(test_df)} samples)")
    
    # Features and target
    feature_cols = ['Day', 'Temperature', 'PAR', 'pH', 'OD680', 'Nitrogen_mg_L', 'CO2_mg_L', 'Dissolved_Oxygen_mg_L']
    target_col = 'Biomass'
    
    # Fit scalers on training data only
    x_scaler = StandardScaler()
    y_scaler = StandardScaler()
    
    X_train = x_scaler.fit_transform(train_df[feature_cols].values)
    y_train = y_scaler.fit_transform(train_df[[target_col]].values)
    
    X_val = x_scaler.transform(val_df[feature_cols].values)
    y_val = y_scaler.transform(val_df[[target_col]].values)
    
    X_test = x_scaler.transform(test_df[feature_cols].values)
    y_test = y_scaler.transform(test_df[[target_col]].values)
    
    return {
        'train': (X_train, y_train),
        'val': (X_val, y_val),
        'test': (X_test, y_test),
        'scalers': (x_scaler, y_scaler),
        'dataframes': (train_df, val_df, test_df)
    }


def introduce_sparsity(y, sparsity_ratio=0.2, seed=42):
    """
    Randomly remove observations to simulate sparse data.
    
    Returns:
        y_sparse: Array with some values masked
        mask: Boolean mask (True = observed, False = missing)
    """
    np.random.seed(seed)
    n_samples = len(y)
    n_remove = int(n_samples * sparsity_ratio)
    
    mask = np.ones(n_samples, dtype=bool)
    remove_indices = np.random.choice(n_samples, size=n_remove, replace=False)
    mask[remove_indices] = False
    
    return mask


def introduce_noise(y, noise_level=0.1, seed=42):
    """
    Add Gaussian noise to observations.
    
    Args:
        y: Target values
        noise_level: Standard deviation as fraction of signal std
        
    Returns:
        Noisy observations
    """
    np.random.seed(seed)
    noise_std = noise_level * np.std(y)
    noise = np.random.normal(0, noise_std, size=y.shape)
    return y + noise


def train_model(
    model: PINN,
    data_dict: Dict,
    epochs=5000,
    lr=0.001,
    lambda_data=1.0,
    lambda_phys=0.1,
    lambda_ic=1.0,
    lambda_reg=0.0001,
    sparse_mask=None,
    device='cpu'
):
    """
    Train PINN with combined data and physics loss.
    
    Loss components:
        L_total = λ_data * L_data + λ_phys * L_phys + λ_ic * L_ic + λ_reg * L_reg
    """
    X_train, y_train = data_dict['train']
    X_val, y_val = data_dict['val']
    x_scaler, y_scaler = data_dict['scalers']
    
    # Convert to tensors
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32, device=device)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32, device=device)
    X_val_tensor = torch.tensor(X_val, dtype=torch.float32, device=device)
    y_val_tensor = torch.tensor(y_val, dtype=torch.float32, device=device)
    
    # Apply sparsity mask if provided
    if sparse_mask is not None:
        X_train_tensor = X_train_tensor[sparse_mask]
        y_train_tensor = y_train_tensor[sparse_mask]
    
    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    # Training history
    history = {
        'train_loss': [],
        'val_loss': [],
        'data_loss': [],
        'phys_loss': [],
        'ic_loss': [],
        'reg_loss': []
    }
    
    # Get initial condition
    y_initial = y_train_tensor[0]
    
    for epoch in range(epochs):
        model.train()
        
        # Enable gradient computation for physics loss
        X_train_grad = X_train_tensor.clone().detach().requires_grad_(True)
        
        # Forward pass
        y_pred = model(X_train_grad)
        
        # Data loss
        loss_data = nn.MSELoss()(y_pred, y_train_tensor)
        
        # Physics loss
        loss_phys = model.physics_loss(X_train_grad, y_pred, x_scaler, y_scaler) if model.use_physics else torch.tensor(0.0)
        
        # Initial condition loss
        y_pred_initial = y_pred[0]
        loss_ic = (y_pred_initial - y_initial) ** 2
        
        # Regularization loss (L2)
        loss_reg = sum(p.pow(2).sum() for p in model.parameters())
        
        # Total loss
        loss_total = (
            lambda_data * loss_data +
            lambda_phys * loss_phys +
            lambda_ic * loss_ic +
            lambda_reg * loss_reg
        )
        
        # Backward pass
        optimizer.zero_grad()
        loss_total.backward()
        optimizer.step()
        
        # Validation
        if epoch % 100 == 0:
            model.eval()
            with torch.no_grad():
                y_val_pred = model(X_val_tensor)
                val_loss = nn.MSELoss()(y_val_pred, y_val_tensor)
            
            history['train_loss'].append(loss_total.item())
            history['val_loss'].append(val_loss.item())
            history['data_loss'].append(loss_data.item())
            history['phys_loss'].append(loss_phys.item() if isinstance(loss_phys, torch.Tensor) else 0.0)
            history['ic_loss'].append(loss_ic.item())
            history['reg_loss'].append(loss_reg.item())
            
            if epoch % 500 == 0:
                print(f"Epoch {epoch}: Train={loss_total.item():.6f}, Val={val_loss.item():.6f}, "
                      f"Data={loss_data.item():.6f}, Phys={loss_phys.item() if isinstance(loss_phys, torch.Tensor) else 0:.6f}")
    
    return history


def evaluate_model(model: PINN, data_dict: Dict, device='cpu') -> Dict:
    """
    Evaluate model on test set.
    """
    X_test, y_test = data_dict['test']
    x_scaler, y_scaler = data_dict['scalers']
    
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32, device=device)
    
    model.eval()
    with torch.no_grad():
        y_pred_norm = model(X_test_tensor)
    
    # Denormalize predictions
    y_pred = y_scaler.inverse_transform(y_pred_norm.cpu().numpy())
    y_true = y_scaler.inverse_transform(y_test)
    
    # Calculate metrics
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    
    # MAPE (avoid division by zero)
    mape = np.mean(np.abs((y_true - y_pred) / (y_true + 1e-8))) * 100
    
    return {
        'rmse': rmse,
        'mae': mae,
        'r2': r2,
        'mape': mape,
        'y_true': y_true,
        'y_pred': y_pred
    }


def save_model(model: PINN, scalers, history, metrics, save_dir='models'):
    """
    Save trained model and associated data.
    """
    os.makedirs(save_dir, exist_ok=True)
    
    # Save model state
    model_path = os.path.join(save_dir, 'pinn_model.pt')
    torch.save({
        'model_state_dict': model.state_dict(),
        'use_physics': model.use_physics
    }, model_path)
    
    # Save scalers
    scaler_path = os.path.join(save_dir, 'scalers.pt')
    torch.save({
        'x_scaler': scalers[0],
        'y_scaler': scalers[1]
    }, scaler_path)
    
    # Save history and metrics
    results_path = os.path.join(save_dir, 'results.json')
    with open(results_path, 'w') as f:
        json.dump({
            'history': history,
            'metrics': {k: float(v) if not isinstance(v, np.ndarray) else v.tolist() 
                       for k, v in metrics.items()}
        }, f, indent=2)
    
    print(f"\nModel saved to {save_dir}/")


if __name__ == "__main__":
    print("PINN Model Module - Use train_pinn.py to train models")
