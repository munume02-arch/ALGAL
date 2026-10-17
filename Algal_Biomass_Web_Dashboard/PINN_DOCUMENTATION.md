# Physics-Informed Neural Network for Algal Biomass Estimation

## FYP Title
**"Physics-Informed Neural Networks for Robust Algal Biomass Estimation under Sparse and Noisy Low-Cost Sensor Datasets"**

---

## 1. GOVERNING EQUATION

The PINN enforces a logistic growth model with environmental modulation:

```
dB/dt = μ_eff × B × (1 - B/K)
```

where:
- **B** = Biomass concentration (g/L)
- **μ_eff** = Effective growth rate (1/day)
- **K** = Carrying capacity (g/L)
- **t** = Time (days)

This represents:
1. **Exponential growth** when B << K
2. **Resource limitation** as B approaches K
3. **Environmental modulation** through μ_eff

---

## 2. EFFECTIVE GROWTH RATE

The effective growth rate is the product of limitation functions:

```
μ_eff = μ_max × f_T(T) × f_PAR(PAR) × f_pH(pH) × f_N(N) × f_CO2(CO2) × f_DO(DO)
```

### Parameters:
- **μ_max** = Maximum specific growth rate (learnable parameter)
- Each f_X represents a biological response function (0 to 1)

---

## 3. RESPONSE FUNCTIONS

### 3.1 Temperature Response: f_T(T)

**Cardinal Temperature Model:**

```
f_T = [(T - T_max) × (T - T_min)²] / [denominator]

where denominator = (T_opt - T_min) × [(T_opt - T_min)(T - T_opt) - (T_opt - T_max)(T_opt + T_min - 2T)]
```

**Parameters:**
- T_min ≈ 15°C (minimum growth temperature)
- T_opt ≈ 27°C (optimal temperature)
- T_max ≈ 35°C (maximum temperature)

**Justification:** Microalgae exhibit cardinal temperature behavior with growth near zero at extremes and optimal at T_opt.

### 3.2 Light Response: f_PAR(PAR)

**Monod Saturation with Photoinhibition:**

```
f_PAR = [PAR / (K_PAR + PAR)] × [K_I / (K_I + PAR)]
```

**Parameters:**
- K_PAR ≈ 100 μmol/m²/s (half-saturation constant)
- K_I ≈ 500 μmol/m²/s (photoinhibition constant)

**Justification:** 
- First term: Saturation at high light (Monod kinetics)
- Second term: Photoinhibition at very high PAR
- Product gives realistic light response curve

### 3.3 pH Response: f_pH(pH)

**Gaussian Optimum:**

```
f_pH = exp[-0.5 × ((pH - pH_opt) / σ_pH)²]
```

**Parameters:**
- pH_opt ≈ 8.0 (optimal pH for most green algae)
- σ_pH ≈ 1.0 (width of pH tolerance)

**Justification:** Most microalgae show Gaussian response around optimal pH.

### 3.4 Nitrogen Limitation: f_N(N)

**Monod Kinetics:**

```
f_N = N / (K_N + N)
```

**Parameters:**
- K_N ≈ 5 mg/L (half-saturation constant for nitrogen)

**Justification:** Classic Monod model for nutrient limitation widely used in algal growth models.

### 3.5 CO2 Limitation: f_CO2(CO2)

**Monod Kinetics:**

```
f_CO2 = CO2 / (K_CO2 + CO2)
```

**Parameters:**
- K_CO2 ≈ 1.5 mg/L (half-saturation for dissolved CO2)

**Justification:** CO2 limits photosynthetic carbon fixation following saturation kinetics.

### 3.6 Dissolved Oxygen: f_DO(DO)

**Gaussian Response (Observational):**

```
f_DO = exp[-0.5 × ((DO - DO_opt) / σ_DO)²]
```

**Parameters:**
- DO_opt ≈ 7.0 mg/L
- σ_DO ≈ 3.0 mg/L

**Important Note:** DO is treated as observational rather than directly limiting. Very high DO can indicate photoinhibition stress, while very low DO may indicate respiration dominance. The relationship is complex and not blindly assumed to be monotonic.

---

## 4. NEURAL NETWORK ARCHITECTURE

```
Input Layer (8 features):
    - Day
    - Temperature (°C)
    - PAR (μmol/m²/s)
    - pH
    - OD680
    - Nitrogen (mg/L)
    - CO2 (mg/L)
    - Dissolved Oxygen (mg/L)
    ↓
Hidden Layer 1: 64 neurons, Tanh activation
    ↓
Hidden Layer 2: 64 neurons, Tanh activation
    ↓
Hidden Layer 3: 32 neurons, Tanh activation
    ↓
Output Layer: 1 neuron (Biomass)
    ↓
Softplus Activation (ensures B ≥ 0)
```

**Total Parameters:** ~5,000 trainable weights + physics parameters

---

## 5. LOSS FUNCTION

```
L_total = λ_data × L_data + λ_phys × L_phys + λ_ic × L_ic + λ_reg × L_reg
```

### 5.1 Data Loss (L_data)

```
L_data = MSE(B_pred, B_measured) = (1/N) Σ(B_pred - B_measured)²
```

Ensures predictions match observed biomass.

### 5.2 Physics Loss (L_phys)

```
L_phys = (1/N) Σ[dB/dt_NN - dB/dt_physics]²

where:
    dB/dt_NN = ∂B_pred/∂t  (computed via automatic differentiation)
    dB/dt_physics = μ_eff × B × (1 - B/K)
```

**This is the key PINN component:** Forces the neural network's predictions to satisfy the biological growth equation.

### 5.3 Initial Condition Loss (L_ic)

```
L_ic = (B_pred(t=t_0) - B_initial)²
```

Ensures the model respects the initial biomass measurement.

### 5.4 Regularization Loss (L_reg)

```
L_reg = Σ(θ²)
```

L2 regularization on all network weights to prevent overfitting.

### 5.5 Loss Weights

**Standard Configuration:**
- λ_data = 1.0 (data fidelity)
- λ_phys = 0.5 (strong physics enforcement)
- λ_ic = 1.0 (initial condition)
- λ_reg = 0.0001 (regularization)

**Ablation Study:**
- Weak physics: λ_phys = 0.05
- No physics: λ_phys = 0.0 (data-only NN)

---

## 6. TRAINING PROCEDURE

### 6.1 Data Preparation

**Chronological Split (NO SHUFFLING):**
- Training: Days 1-60 (60 samples, 60%)
- Validation: Days 61-80 (20 samples, 20%)
- Testing: Days 81-100 (20 samples, 20%)

**Normalization:**
- StandardScaler fitted on training data only
- Applied to features and target
- Prevents data leakage from validation/test sets

### 6.2 Optimization

- **Optimizer:** Adam
- **Learning Rate:** 0.001
- **Epochs:** 5,000 (with early stopping option)
- **Batch Processing:** Full batch (small dataset)

### 6.3 Automatic Differentiation

PyTorch's autograd computes ∂B/∂t:

```python
dB_dt = torch.autograd.grad(
    outputs=B_pred,
    inputs=Day,
    grad_outputs=torch.ones_like(B_pred),
    create_graph=True
)[0]
```

This gradient is compared against the physics-based dB/dt.

---

## 7. BASELINE MODELS

### Model A: PINN (Strong Physics)
- Full physics loss with λ_phys = 0.5
- Uses all response functions
- Enforces growth equation

### Model B: PINN (Weak Physics)
- Reduced physics loss with λ_phys = 0.05
- **Ablation study:** Does physics help?

### Model C: Data-Only Neural Network
- Same architecture, λ_phys = 0.0
- Pure data-driven learning
- **Comparison baseline**

### Model D: Random Forest
- Simple non-neural baseline
- 100 trees, max depth 10
- Represents classical ML approach

---

## 8. ROBUSTNESS EXPERIMENTS

### 8.1 Complete Data
All 60 training samples with clean measurements.

### 8.2 Sparse Data (20% Missing)
Randomly remove 20% of biomass observations.
- Tests if physics compensates for missing data
- Simulates sensor failures

### 8.3 Sparse Data (40% Missing)
Randomly remove 40% of biomass observations.
- More challenging sparsity scenario

### 8.4 Noisy Data (10% Noise)
Add Gaussian noise: σ_noise = 0.1 × σ_biomass
- Simulates low-cost sensor inaccuracy

### 8.5 Noisy Data (20% Noise)
Add Gaussian noise: σ_noise = 0.2 × σ_biomass
- More challenging noise scenario

---

## 9. EVALUATION METRICS

### 9.1 Root Mean Squared Error (RMSE)

```
RMSE = √[(1/N) Σ(y_true - y_pred)²]
```

**Units:** g/L (same as biomass)
**Interpretation:** Lower is better, represents typical prediction error

### 9.2 Mean Absolute Error (MAE)

```
MAE = (1/N) Σ|y_true - y_pred|
```

**Units:** g/L
**Interpretation:** Average absolute deviation, less sensitive to outliers than RMSE

### 9.3 R² Score (Coefficient of Determination)

```
R² = 1 - [SS_res / SS_tot]

where:
    SS_res = Σ(y_true - y_pred)²
    SS_tot = Σ(y_true - ȳ)²
```

**Range:** -∞ to 1
**Interpretation:** 
- R² = 1: Perfect predictions
- R² = 0: Model no better than mean
- R² < 0: Model worse than predicting mean

**Note:** Can be negative on test set if model performs poorly.

### 9.4 Mean Absolute Percentage Error (MAPE)

```
MAPE = (100/N) Σ|((y_true - y_pred) / y_true)|
```

**Units:** Percentage
**Interpretation:** Scale-independent error metric

---

## 10. OD680 CAVEAT

**⚠️ CRITICAL NOTE ON OD680:**

Optical Density at 680nm (OD680) is strongly correlated with biomass because:
1. Chlorophyll absorbs strongly at 680nm
2. OD680 is often used to estimate biomass

**In this dataset:**
- OD680 data was included in the synthetic dataset
- If OD680 was directly calculated from biomass, it creates a circular dependency
- This would inflate model performance artificially

**Proper Treatment:**
- OD680 should be treated as an **independent sensor measurement**
- If correlation > 0.95, carefully document this in results
- Consider training models **with and without** OD680 to show its impact

**Alternative Experiment:**
- Train Model 1: With OD680
- Train Model 2: Without OD680
- Compare performance to quantify OD680's contribution

---

## 11. ASSUMPTIONS AND LIMITATIONS

### 11.1 Model Assumptions

1. **Logistic Growth:** Assumes single-population logistic dynamics
   - Does not model lag phase explicitly
   - May not capture complex multi-phase behavior

2. **Multiplicative Limitations:** Environmental factors multiply
   - Assumes no synergistic or antagonistic interactions
   - Liebig's law of the minimum not enforced

3. **Constant Parameters:** K, μ_max assumed constant
   - In reality, may vary with nutrients, culture age
   - PINN learns effective average values

4. **Well-Mixed Culture:** No spatial gradients
   - Valid for small bioreactors with mixing
   - Invalid for large ponds with stratification

5. **No Cell Death:** Model omits explicit mortality term
   - Could add: dB/dt = growth - death
   - Simplification valid during exponential phase

### 11.2 Data Limitations

1. **Synthetic Dataset:** All data is computer-generated
   - Not validated against real experiments
   - Real algae behavior may differ

2. **Single Run:** Only one growth trajectory
   - Need multiple runs for statistical validation
   - Biological replicates essential for real FYP

3. **100 Days:** Long time series, but:
   - Single environmental trajectory
   - Need diverse conditions for generalization

4. **No Ground Truth for μ_eff:** Cannot validate learned growth rate
   - Would need controlled experiments at fixed conditions

### 11.3 PINN Limitations

1. **Local Minimum:** Neural networks may converge to suboptimal solutions
2. **Hyperparameter Sensitivity:** λ_phys tuning is crucial
3. **Computational Cost:** Automatic differentiation adds overhead
4. **Interpretability:** Learned physics parameters may not match literature

---

## 12. REPRODUCIBILITY

### 12.1 Random Seeds

```python
np.random.seed(42)
torch.manual_seed(42)
```

### 12.2 Saved Artifacts

- `models/pinn_strong/pinn_model.pt` - PINN state dict
- `models/data_only/pinn_model.pt` - Data-only NN state dict
- `models/*/scalers.pt` - StandardScalers for normalization
- `models/*/results.json` - Training history and metrics
- `plots/*.png` - All visualizations

### 12.3 Configuration

All hyperparameters documented in `train_pinn.py` CONFIG dict.

---

## 13. FUTURE ENHANCEMENTS

### 13.1 Model Improvements

1. **Add Death Term:**
   ```
   dB/dt = μ_eff × B × (1 - B/K) - δ × B
   ```
   where δ = death rate

2. **Multi-Population Model:**
   ```
   dB1/dt = f1(...)
   dB2/dt = f2(...)
   ```
   Model mixed algal cultures

3. **Nutrient Dynamics:**
   ```
   dN/dt = -Y_N × dB/dt
   ```
   Couple nutrient depletion to growth

### 13.2 Data Enhancements

1. Collect real experimental data
2. Multiple cultivation runs with replicates
3. Controlled experiments varying single factors
4. Include more sensors (turbidity, fluorescence)

### 13.3 Physics Enhancements

1. Use known literature values for K_PAR, K_N, etc.
2. Incorporate Droop model for internal nutrient quotas
3. Add light attenuation through culture (Beer-Lambert)
4. Model photoadaptation dynamics

---

## 14. KEY FINDINGS (After Training)

### 14.1 Does Physics Help?

**Hypothesis:** Physics loss should improve generalization, especially under sparse/noisy data.

**Expected Behavior:**
- Complete data: PINN ≈ Data-Only (plenty of data)
- Sparse data: PINN > Data-Only (physics fills gaps)
- Noisy data: PINN > Data-Only (physics acts as regularizer)

### 14.2 Physics Residual Evidence

Plot of `dB/dt_NN - dB/dt_physics` should:
- Be small (< 0.01) for good physics adherence
- Cluster around zero
- Not show systematic bias

**If residual is large:** Physics and data are inconsistent, suggesting:
- Wrong growth model
- Wrong response functions
- Insufficient λ_phys weight

---

## 15. CITATION AND ACKNOWLEDGMENT

This implementation is based on:

**Physics-Informed Neural Networks (PINNs):**
- Raissi, M., Perdikaris, P., & Karniadakis, G. E. (2019). Physics-informed neural networks: A deep learning framework for solving forward and inverse problems involving nonlinear partial differential equations. *Journal of Computational Physics*, 378, 686-707.

**Algal Growth Modeling:**
- Bernard, O. (2011). Hurdles and challenges for modelling and control of microalgae for CO2 mitigation and biofuel production. *Journal of Process Control*, 21(10), 1378-1389.

---

## 16. CONCLUSION

This PINN implementation genuinely incorporates biological physics through:

1. ✅ Explicit growth equation: dB/dt = μ_eff × B × (1 - B/K)
2. ✅ Biologically justified response functions
3. ✅ Physics residual calculated via automatic differentiation
4. ✅ Physics loss actively trained alongside data loss
5. ✅ Ablation study comparing physics vs. no physics
6. ✅ Robustness tests under sparse and noisy conditions

**This is NOT just a regular neural network labeled as a PINN.**

The physics loss directly influences learning, and the model must satisfy both:
- Match observed data (data loss)
- Obey biological laws (physics loss)

---

## SYNTHETIC DATA DISCLAIMER

⚠️ **IMPORTANT:** All results are from SYNTHETIC data generated for demonstration purposes.

**Before claiming experimental validity:**
1. Validate with real algal culture data
2. Compare against laboratory measurements
3. Test on multiple species and conditions
4. Verify response function parameters match literature

**This is a proof-of-concept for the PINN methodology, NOT a validated prediction tool for production use.**
