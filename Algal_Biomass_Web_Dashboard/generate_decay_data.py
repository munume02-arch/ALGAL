"""
Generate synthetic algal decay dataset (100 days)

Simulates algal biomass decay due to:
- Nutrient depletion
- Suboptimal environmental conditions
- Cell death and lysis
- Reduced light availability
"""

import numpy as np
import pandas as pd

np.random.seed(42)

# Initialize parameters
days = np.arange(1, 101)
n_days = len(days)

# Start with high biomass and decay
biomass_initial = 1.8  # g/L - starting high
decay_rate = 0.015  # decay rate per day

# Generate decay with some noise
biomass = np.zeros(n_days)
biomass[0] = biomass_initial

for i in range(1, n_days):
    # Exponential decay with environmental factors
    base_decay = biomass[i-1] * (1 - decay_rate)
    # Add some stochastic variation
    noise = np.random.normal(0, 0.01)
    biomass[i] = max(0.05, base_decay + noise)  # Minimum biomass threshold

# Add slight fluctuations to make it realistic
biomass += np.random.normal(0, 0.02, n_days)
biomass = np.clip(biomass, 0.05, biomass_initial)

# Temperature - declining and suboptimal
temperature_base = 24.0  # Suboptimal (too cold)
temperature = temperature_base + np.random.normal(0, 1.5, n_days)
temperature += -0.02 * days  # Gradual cooling
temperature = np.clip(temperature, 18, 28)

# PAR - decreasing light availability
par_base = 80  # Low light
par = par_base + np.random.normal(0, 15, n_days)
par += -0.15 * days  # Declining light
par = np.clip(par, 30, 120)

# pH - becoming more acidic (stress)
ph_base = 7.2  # Suboptimal (too acidic)
ph = ph_base + np.random.normal(0, 0.2, n_days)
ph += -0.005 * days  # Gradually acidifying
ph = np.clip(ph, 6.5, 7.8)

# OD680 - correlated with biomass (declining)
od680 = biomass * 1.5 + np.random.normal(0, 0.05, n_days)
od680 = np.clip(od680, 0.1, 3.0)

# Nitrogen - severely depleted
nitrogen_initial = 8.0  # Low starting
nitrogen = nitrogen_initial * np.exp(-0.025 * days) + np.random.normal(0, 0.5, n_days)
nitrogen = np.clip(nitrogen, 0.5, 10)

# CO2 - depleted
co2_initial = 1.2
co2 = co2_initial * np.exp(-0.015 * days) + np.random.normal(0, 0.1, n_days)
co2 = np.clip(co2, 0.3, 1.5)

# Dissolved Oxygen - decreasing (less photosynthesis)
do_base = 5.5  # Lower oxygen
do = do_base + np.random.normal(0, 0.8, n_days)
do += -0.02 * days  # Declining oxygen
do = np.clip(do, 3.0, 7.0)

# Create DataFrame
df_decay = pd.DataFrame({
    'Day': days,
    'Temperature': np.round(temperature, 2),
    'PAR': np.round(par, 2),
    'pH': np.round(ph, 3),
    'OD680': np.round(od680, 4),
    'Nitrogen_mg_L': np.round(nitrogen, 2),
    'CO2_mg_L': np.round(co2, 3),
    'Dissolved_Oxygen_mg_L': np.round(do, 2),
    'Biomass': np.round(biomass, 4)
})

# Save to CSV
df_decay.to_csv('data/algal_decay_100_days.csv', index=False)

print("Algal Decay Dataset Generated!")
print(f"Days: {days[0]} to {days[-1]}")
print(f"Biomass range: {biomass.min():.4f} to {biomass.max():.4f} g/L")
print(f"Biomass decline: {biomass[0]:.4f} → {biomass[-1]:.4f} g/L")
print(f"Temperature range: {temperature.min():.1f} to {temperature.max():.1f} °C")
print(f"PAR range: {par.min():.1f} to {par.max():.1f} μmol/m²/s")
print(f"Nitrogen range: {nitrogen.min():.2f} to {nitrogen.max():.2f} mg/L")
print("\nDataset saved to: data/algal_decay_100_days.csv")
print("\nFirst 5 rows:")
print(df_decay.head())
print("\nLast 5 rows:")
print(df_decay.tail())
