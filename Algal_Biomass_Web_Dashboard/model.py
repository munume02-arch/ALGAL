import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler

FEATURES = ["Day", "Temperature", "PAR", "pH", "OD680"]
TARGET = "Biomass"

class PINN(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(5, 32), torch.nn.Tanh(),
            torch.nn.Linear(32, 32), torch.nn.Tanh(),
            torch.nn.Linear(32, 1), torch.nn.Softplus()
        )
    def forward(self, x):
        return self.net(x)

def physics_mu(T, ph, par):
    # Smooth, interpretable growth-rate prior: favorable temperature, pH and light
    mu_max = 0.65
    temp_factor = torch.exp(-0.5 * ((T - 28.0) / 4.0) ** 2)
    ph_factor = torch.exp(-0.5 * ((ph - 8.2) / 0.5) ** 2)
    light_factor = par / (par + 80.0)
    return mu_max * temp_factor * ph_factor * light_factor

def train_model(df, seed=0, epochs=3500, lr=0.01, physics_weight=0.05):
    torch.manual_seed(seed)
    np.random.seed(seed)
    scaler = StandardScaler().fit(df[FEATURES].values)
    Xs = scaler.transform(df[FEATURES].values)
    X = torch.tensor(Xs, dtype=torch.float32, requires_grad=True)
    y = torch.tensor(df[TARGET].values.reshape(-1,1), dtype=torch.float32)
    model = PINN()
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    xscale_day = float(scaler.scale_[0])
    for _ in range(epochs):
        opt.zero_grad()
        pred = model(X)
        grads = torch.autograd.grad(pred, X, torch.ones_like(pred), create_graph=True)[0]
        dXdt = grads[:,0:1] / xscale_day
        T = X[:,1:2] * float(scaler.scale_[1]) + float(scaler.mean_[1])
        PAR = X[:,2:3] * float(scaler.scale_[2]) + float(scaler.mean_[2])
        ph = X[:,3:4] * float(scaler.scale_[3]) + float(scaler.mean_[3])
        mu = physics_mu(T, ph, PAR)
        Xmax = 1.9
        residual = dXdt - mu * pred * (1.0 - pred / Xmax)
        data_loss = torch.mean((pred-y)**2)
        physics_loss = torch.mean(residual**2)
        loss = data_loss + physics_weight * physics_loss
        loss.backward()
        opt.step()
    return model, scaler

def predict(model, scaler, rows):
    if isinstance(rows, dict):
        rows = pd.DataFrame([rows])
    arr = scaler.transform(rows[FEATURES].values)
    with torch.no_grad():
        p = model(torch.tensor(arr, dtype=torch.float32)).numpy().ravel()
    return np.maximum(p, 0.0)

def train_ensemble(df, n=5):
    models = []
    for seed in range(n):
        models.append(train_model(df, seed=seed))
    return models

def ensemble_predict(models, row):
    preds = np.array([predict(m, s, row)[0] for m,s in models])
    return float(preds.mean()), float(preds.std(ddof=1) if len(preds)>1 else 0.0), preds
