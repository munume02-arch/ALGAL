"""
Enhanced Streamlit Dashboard with PINN Integration

This dashboard displays:
1. Experiment overview with environmental parameters
2. PINN vs Data-Only NN predictions
3. Physics residual analysis
4. Training metrics comparison
5. Prediction interface
"""

import os
import pandas as pd
import numpy as np
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import torch
import json

# Import PINN model
from pinn_model import PINN

st.set_page_config(page_title="PINN Biomass Dashboard", page_icon="🧬", layout="wide")

DATA_PATH = "data/algal_growth_data.csv"
PINN_MODEL_PATH = "models/pinn_strong"
DATA_MODEL_PATH = "models/data_only"


@st.cache_data
def load_data():
    """Load the experimental data."""
    return pd.read_csv(DATA_PATH)


@st.cache_resource
def load_models():
    """Load trained PINN and Data-Only models."""
    models = {}
    
    # Try to load PINN
    pinn_path = os.path.join(PINN_MODEL_PATH, 'pinn_model.pt')
    pinn_scaler_path = os.path.join(PINN_MODEL_PATH, 'scalers.pt')
    pinn_results_path = os.path.join(PINN_MODEL_PATH, 'results.json')
    
    if os.path.exists(pinn_path):
        checkpoint = torch.load(pinn_path, map_location='cpu', weights_only=False)
        pinn_model = PINN(input_dim=8, hidden_dims=[64, 64, 32], use_physics=True)
        pinn_model.load_state_dict(checkpoint['model_state_dict'])
        pinn_model.eval()
        
        scalers = torch.load(pinn_scaler_path, map_location='cpu', weights_only=False)
        
        with open(pinn_results_path, 'r') as f:
            pinn_results = json.load(f)
        
        models['pinn'] = {
            'model': pinn_model,
            'scalers': scalers,
            'results': pinn_results
        }
    
    # Try to load Data-Only model
    data_path = os.path.join(DATA_MODEL_PATH, 'pinn_model.pt')
    data_scaler_path = os.path.join(DATA_MODEL_PATH, 'scalers.pt')
    data_results_path = os.path.join(DATA_MODEL_PATH, 'results.json')
    
    if os.path.exists(data_path):
        checkpoint = torch.load(data_path, map_location='cpu', weights_only=False)
        data_model = PINN(input_dim=8, hidden_dims=[64, 64, 32], use_physics=False)
        data_model.load_state_dict(checkpoint['model_state_dict'])
        data_model.eval()
        
        scalers = torch.load(data_scaler_path, map_location='cpu', weights_only=False)
        
        with open(data_results_path, 'r') as f:
            data_results = json.load(f)
        
        models['data_only'] = {
            'model': data_model,
            'scalers': scalers,
            'results': data_results
        }
    
    return models


def predict_biomass(model_dict, input_df):
    """Make predictions using a loaded model."""
    model = model_dict['model']
    x_scaler = model_dict['scalers']['x_scaler']
    y_scaler = model_dict['scalers']['y_scaler']
    
    # Feature columns
    feature_cols = ['Day', 'Temperature', 'PAR', 'pH', 'OD680', 
                   'Nitrogen_mg_L', 'CO2_mg_L', 'Dissolved_Oxygen_mg_L']
    
    # Scale inputs
    X_scaled = x_scaler.transform(input_df[feature_cols].values)
    X_tensor = torch.tensor(X_scaled, dtype=torch.float32)
    
    # Predict
    with torch.no_grad():
        y_pred_scaled = model(X_tensor).numpy()
    
    # Denormalize
    y_pred = y_scaler.inverse_transform(y_pred_scaled)
    
    return y_pred.flatten()


# Load data and models
df = load_data()
models = load_models()

# Header
st.markdown("# 🧬 Physics-Informed Neural Network Dashboard")
st.caption("FYP: Robust Algal Biomass Estimation under Sparse and Noisy Sensor Data")
st.caption("⚠️ SYNTHETIC DATA - Proof of Concept Only")

# Sidebar navigation
st.sidebar.header("Navigation")
page = st.sidebar.radio(
    "Go to",
    [
        "📊 Overview & Data",
        "🧠 Model Comparison",
        "⚛️ Physics Analysis",
        "📈 Training Metrics",
        "🔮 Make Predictions",
        "📖 Documentation"
    ]
)

# KPI cards
if models:
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Dataset Size", f"{len(df)} days")
    c2.metric("Training Days", "1-60")
    c3.metric("Test Days", "81-100")
    
    if 'pinn' in models:
        pinn_rmse = models['pinn']['results']['metrics']['rmse']
        c4.metric("PINN RMSE", f"{pinn_rmse:.4f} g/L")
else:
    st.warning("⚠️ Models not found. Please run `python train_pinn.py` or `python quick_train.py` first.")

# ========== PAGE 1: OVERVIEW & DATA ==========
if page == "📊 Overview & Data":
    st.header("Dataset Overview")
    
    st.subheader("1. Biomass Growth Trajectory")
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df.Day, y=df.Biomass,
        mode='lines+markers',
        name='Measured Biomass',
        line=dict(width=3, color='#2E86AB')
    ))
    
    # Add split markers
    fig.add_vline(x=60, line_dash="dash", line_color="green", 
                  annotation_text="Train/Val Split")
    fig.add_vline(x=80, line_dash="dash", line_color="orange", 
                  annotation_text="Val/Test Split")
    
    fig.update_layout(
        height=400,
        xaxis_title="Day",
        yaxis_title="Biomass (g/L)",
        hovermode="x unified"
    )
    st.plotly_chart(fig, use_container_width=True)
    
    st.info("📍 Growth phases: Exponential (Days 1-50) → Stationary (Days 50-100)")
    
    st.subheader("2. Environmental Conditions")
    
    # Core sensors
    fig2 = make_subplots(
        rows=2, cols=2,
        subplot_titles=("Temperature (°C)", "PAR (μmol/m²/s)", "pH", "OD680"),
        vertical_spacing=0.12
    )
    
    fig2.add_trace(go.Scatter(x=df.Day, y=df.Temperature, mode='lines+markers', 
                             name='Temperature', line=dict(color='#E63946')), row=1, col=1)
    fig2.add_trace(go.Scatter(x=df.Day, y=df.PAR, mode='lines+markers', 
                             name='PAR', line=dict(color='#F4A261')), row=1, col=2)
    fig2.add_trace(go.Scatter(x=df.Day, y=df.pH, mode='lines+markers', 
                             name='pH', line=dict(color='#2A9D8F')), row=2, col=1)
    fig2.add_trace(go.Scatter(x=df.Day, y=df.OD680, mode='lines+markers', 
                             name='OD680', line=dict(color='#264653')), row=2, col=2)
    
    fig2.update_layout(height=600, showlegend=False)
    st.plotly_chart(fig2, use_container_width=True)
    
    # Additional parameters
    st.subheader("3. Nutrient and Gas Parameters")
    
    fig3 = make_subplots(
        rows=1, cols=3,
        subplot_titles=("Nitrogen (mg/L)", "CO2 (mg/L)", "Dissolved O2 (mg/L)")
    )
    
    fig3.add_trace(go.Scatter(x=df.Day, y=df.Nitrogen_mg_L, mode='lines+markers',
                             name='Nitrogen', line=dict(color='green')), row=1, col=1)
    fig3.add_trace(go.Scatter(x=df.Day, y=df.CO2_mg_L, mode='lines+markers',
                             name='CO2', line=dict(color='orange')), row=1, col=2)
    fig3.add_trace(go.Scatter(x=df.Day, y=df.Dissolved_Oxygen_mg_L, mode='lines+markers',
                             name='DO', line=dict(color='blue')), row=1, col=3)
    
    fig3.update_layout(height=400, showlegend=False)
    st.plotly_chart(fig3, use_container_width=True)
    
    st.subheader("4. Data Table")
    st.dataframe(df, use_container_width=True, hide_index=True)

# ========== PAGE 2: MODEL COMPARISON ==========
elif page == "🧠 Model Comparison":
    st.header("PINN vs Data-Only Neural Network")
    
    if not models:
        st.error("Models not loaded. Please train models first.")
    else:
        # Split data for visualization
        train_df = df[df.Day <= 60]
        val_df = df[(df.Day > 60) & (df.Day <= 80)]
        test_df = df[df.Day > 80]
        
        # Get predictions on test set
        fig = go.Figure()
        
        # Measured biomass
        fig.add_trace(go.Scatter(
            x=df.Day, y=df.Biomass,
            mode='lines+markers',
            name='Measured Biomass',
            line=dict(width=3, color='black')
        ))
        
        # PINN predictions
        if 'pinn' in models:
            pinn_pred = predict_biomass(models['pinn'], test_df)
            fig.add_trace(go.Scatter(
                x=test_df.Day, y=pinn_pred,
                mode='markers+lines',
                name=f'PINN (RMSE={models["pinn"]["results"]["metrics"]["rmse"]:.4f})',
                marker=dict(size=8, symbol='x'),
                line=dict(dash='dot', color='#E63946')
            ))
        
        # Data-Only predictions
        if 'data_only' in models:
            data_pred = predict_biomass(models['data_only'], test_df)
            fig.add_trace(go.Scatter(
                x=test_df.Day, y=data_pred,
                mode='markers+lines',
                name=f'Data-Only NN (RMSE={models["data_only"]["results"]["metrics"]["rmse"]:.4f})',
                marker=dict(size=8, symbol='diamond'),
                line=dict(dash='dash', color='#2A9D8F')
            ))
        
        # Add region shading
        fig.add_vrect(x0=0, x1=60, fillcolor="green", opacity=0.1, 
                     annotation_text="Train", annotation_position="top left")
        fig.add_vrect(x0=60, x1=80, fillcolor="orange", opacity=0.1,
                     annotation_text="Val", annotation_position="top left")
        fig.add_vrect(x0=80, x1=100, fillcolor="red", opacity=0.1,
                     annotation_text="Test", annotation_position="top left")
        
        fig.update_layout(
            height=500,
            xaxis_title="Day",
            yaxis_title="Biomass (g/L)",
            hovermode="x unified",
            legend=dict(x=0.02, y=0.98)
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Metrics comparison
        st.subheader("Performance Metrics Comparison")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### PINN (Physics-Informed)")
            if 'pinn' in models:
                metrics = models['pinn']['results']['metrics']
                st.metric("RMSE", f"{metrics['rmse']:.4f} g/L")
                st.metric("MAE", f"{metrics['mae']:.4f} g/L")
                st.metric("R² Score", f"{metrics['r2']:.4f}")
        
        with col2:
            st.markdown("### Data-Only NN")
            if 'data_only' in models:
                metrics = models['data_only']['results']['metrics']
                st.metric("RMSE", f"{metrics['rmse']:.4f} g/L")
                st.metric("MAE", f"{metrics['mae']:.4f} g/L")
                st.metric("R² Score", f"{metrics['r2']:.4f}")
        
        # Interpretation
        if 'pinn' in models and 'data_only' in models:
            pinn_rmse = models['pinn']['results']['metrics']['rmse']
            data_rmse = models['data_only']['results']['metrics']['rmse']
            improvement = (data_rmse - pinn_rmse) / data_rmse * 100
            
            if improvement > 0:
                st.success(f"✅ PINN improved RMSE by {improvement:.2f}% compared to data-only approach!")
            else:
                st.warning(f"⚠️ Data-only NN performed {abs(improvement):.2f}% better. Physics may need tuning.")

# ========== PAGE 3: PHYSICS ANALYSIS ==========
elif page == "⚛️ Physics Analysis":
    st.header("Physics Residual Analysis")
    
    if 'pinn' not in models:
        st.error("PINN model not loaded.")
    else:
        st.markdown("""
        ### Governing Equation
        ```
        dB/dt = μ_eff × B × (1 - B/K)
        ```
        
        where:
        - **μ_eff** = Effective growth rate (function of T, PAR, pH, N, CO2, DO)
        - **K** = Carrying capacity (g/L)
        - **B** = Biomass concentration (g/L)
        
        ### Response Functions
        """)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            **Temperature:** Cardinal model  
            **Light (PAR):** Monod + photoinhibition  
            **pH:** Gaussian optimum  
            """)
        
        with col2:
            st.markdown("""
            **Nitrogen:** Monod kinetics  
            **CO2:** Monod kinetics  
            **DO:** Gaussian (observational)  
            """)
        
        st.subheader("Learned Physics Parameters")
        
        pinn_model = models['pinn']['model']
        physics = pinn_model.physics
        
        params_df = pd.DataFrame({
            'Parameter': [
                'μ_max (max growth rate)',
                'K (carrying capacity)',
                'T_min (min temperature)',
                'T_opt (optimal temperature)',
                'T_max (max temperature)',
                'K_PAR (light half-sat)',
                'K_I (photoinhibition)',
                'pH_opt (optimal pH)',
                'pH_width',
                'K_N (nitrogen half-sat)',
                'K_CO2 (CO2 half-sat)'
            ],
            'Value': [
                f"{abs(physics.mu_max.item()):.4f} /day",
                f"{abs(physics.K_capacity.item()):.4f} g/L",
                f"{physics.T_min.item():.2f} °C",
                f"{physics.T_opt.item():.2f} °C",
                f"{physics.T_max.item():.2f} °C",
                f"{physics.K_PAR.item():.2f} μmol/m²/s",
                f"{physics.K_I.item():.2f} μmol/m²/s",
                f"{physics.pH_opt.item():.2f}",
                f"{physics.pH_width.item():.2f}",
                f"{physics.K_N.item():.2f} mg/L",
                f"{physics.K_CO2.item():.3f} mg/L"
            ]
        })
        
        st.dataframe(params_df, use_container_width=True, hide_index=True)
        
        st.info("💡 These parameters were learned during training by minimizing both data loss and physics residual.")

# ========== PAGE 4: TRAINING METRICS ==========
elif page == "📈 Training Metrics":
    st.header("Training History and Loss Curves")
    
    if not models:
        st.error("Models not loaded.")
    else:
        # Plot loss curves
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=("Total Loss", "Data Loss", "Physics Loss", "IC Loss")
        )
        
        if 'pinn' in models:
            history = models['pinn']['results']['history']
            epochs = list(range(len(history['train_loss'])))
            
            fig.add_trace(go.Scatter(x=epochs, y=history['train_loss'], 
                                    name='PINN Train', line=dict(color='red')), row=1, col=1)
            fig.add_trace(go.Scatter(x=epochs, y=history['val_loss'], 
                                    name='PINN Val', line=dict(dash='dash', color='red')), row=1, col=1)
            fig.add_trace(go.Scatter(x=epochs, y=history['data_loss'], 
                                    name='PINN', line=dict(color='red')), row=1, col=2)
            fig.add_trace(go.Scatter(x=epochs, y=history['phys_loss'], 
                                    name='PINN', line=dict(color='red')), row=2, col=1)
            fig.add_trace(go.Scatter(x=epochs, y=history['ic_loss'], 
                                    name='PINN', line=dict(color='red')), row=2, col=2)
        
        if 'data_only' in models:
            history = models['data_only']['results']['history']
            epochs = list(range(len(history['train_loss'])))
            
            fig.add_trace(go.Scatter(x=epochs, y=history['train_loss'], 
                                    name='Data-Only Train', line=dict(color='blue')), row=1, col=1)
            fig.add_trace(go.Scatter(x=epochs, y=history['val_loss'], 
                                    name='Data-Only Val', line=dict(dash='dash', color='blue')), row=1, col=1)
            fig.add_trace(go.Scatter(x=epochs, y=history['data_loss'], 
                                    name='Data-Only', line=dict(color='blue')), row=1, col=2)
        
        fig.update_yaxes(type="log")
        fig.update_layout(height=700)
        st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("""
        ### Loss Components Explained
        
        - **Total Loss:** Combined weighted loss (data + physics + IC + regularization)
        - **Data Loss:** MSE between predictions and measurements
        - **Physics Loss:** Mean squared physics residual (dB/dt_NN - dB/dt_physics)²
        - **IC Loss:** Initial condition enforcement
        
        ⚠️ Note: Physics loss is zero for data-only model.
        """)

# ========== PAGE 5: PREDICTIONS ==========
elif page == "🔮 Make Predictions":
    st.header("Biomass Prediction")
    
    if not models:
        st.error("Models not loaded.")
    else:
        st.markdown("### Enter Sensor Readings")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            day = st.number_input("Day", min_value=1.0, value=101.0, step=1.0)
            temp = st.number_input("Temperature (°C)", value=26.2, step=0.1)
            par = st.number_input("PAR (μmol/m²/s)", value=129.0, step=1.0)
        
        with col2:
            ph = st.number_input("pH", value=7.74, step=0.01)
            od = st.number_input("OD680", value=2.60, step=0.01)
            nitrogen = st.number_input("Nitrogen (mg/L)", value=4.0, step=0.1)
        
        with col3:
            co2 = st.number_input("CO2 (mg/L)", value=1.45, step=0.01)
            do = st.number_input("Dissolved O2 (mg/L)", value=7.50, step=0.1)
        
        if st.button("🚀 Predict Biomass", type="primary"):
            input_data = pd.DataFrame([{
                'Day': day,
                'Temperature': temp,
                'PAR': par,
                'pH': ph,
                'OD680': od,
                'Nitrogen_mg_L': nitrogen,
                'CO2_mg_L': co2,
                'Dissolved_Oxygen_mg_L': do
            }])
            
            col1, col2 = st.columns(2)
            
            with col1:
                if 'pinn' in models:
                    pinn_pred = predict_biomass(models['pinn'], input_data)[0]
                    st.success(f"### PINN Prediction\n## {pinn_pred:.3f} g/L")
            
            with col2:
                if 'data_only' in models:
                    data_pred = predict_biomass(models['data_only'], input_data)[0]
                    st.info(f"### Data-Only Prediction\n## {data_pred:.3f} g/L")
            
            st.markdown("---")
            st.markdown("### Input Summary")
            st.dataframe(input_data.T, use_container_width=True)

# ========== PAGE 6: DOCUMENTATION ==========
elif page == "📖 Documentation":
    st.header("Scientific Documentation")
    
    st.markdown("""
    ### FYP Title
    **"Physics-Informed Neural Networks for Robust Algal Biomass Estimation under Sparse and Noisy Low-Cost Sensor Datasets"**
    
    ### What is a PINN?
    
    A Physics-Informed Neural Network (PINN) is a neural network that learns from both:
    1. **Data:** Observed measurements (biomass values)
    2. **Physics:** Known biological laws (growth equations)
    
    The key innovation: The neural network must satisfy both data AND physics simultaneously.
    
    ### Key Equations
    
    **Growth Model:**
    ```
    dB/dt = μ_eff × B × (1 - B/K)
    ```
    
    **Effective Growth Rate:**
    ```
    μ_eff = μ_max × f_T × f_PAR × f_pH × f_N × f_CO2 × f_DO
    ```
    
    **Loss Function:**
    ```
    L_total = λ_data × L_data + λ_phys × L_phys + λ_ic × L_ic + λ_reg × L_reg
    ```
    
    ### Why Use PINN?
    
    **Advantages:**
    - ✅ Better generalization with limited data
    - ✅ Handles missing observations (sparse data)
    - ✅ More robust to sensor noise
    - ✅ Predictions respect biological constraints
    - ✅ Interpretable learned parameters
    
    **Challenges:**
    - ❌ Requires domain knowledge for physics model
    - ❌ Hyperparameter tuning (loss weights)
    - ❌ Computational cost of automatic differentiation
    - ❌ May converge to local minima
    
    ### Model Architecture
    
    ```
    Input (8 features) → Hidden(64) → Hidden(64) → Hidden(32) → Output(1)
                          Tanh        Tanh        Tanh        Softplus
    ```
    
    **Total Parameters:** ~5,000 weights + physics parameters
    
    ### Validation Strategy
    
    **Chronological Split (NO SHUFFLING):**
    - Training: Days 1-60 (60%)
    - Validation: Days 61-80 (20%)
    - Testing: Days 81-100 (20%)
    
    **Robustness Tests:**
    - Complete data
    - 20% sparse (missing observations)
    - 40% sparse
    - 10% noise
    - 20% noise
    
    ### Important Disclaimers
    
    ⚠️ **SYNTHETIC DATA:** All results are from computer-generated data, NOT real experiments.
    
    ⚠️ **OD680 CAVEAT:** Optical density may be highly correlated with biomass. Real validation requires independent sensors.
    
    ⚠️ **SINGLE RUN:** Need multiple biological replicates for statistical validity.
    
    ### For More Details
    
    See `PINN_DOCUMENTATION.md` for comprehensive scientific documentation including:
    - Complete mathematical formulations
    - Response function justifications
    - Assumptions and limitations
    - Future enhancements
    - Literature citations
    """)
    
    if st.button("📄 View Full Documentation"):
        try:
            with open("PINN_DOCUMENTATION.md", "r") as f:
                doc_content = f.read()
            st.download_button(
                "Download PINN_DOCUMENTATION.md",
                doc_content,
                file_name="PINN_DOCUMENTATION.md",
                mime="text/markdown"
            )
        except:
            st.error("Documentation file not found.")

# Footer
st.markdown("---")
st.caption("🧬 Physics-Informed Neural Network Dashboard | FYP Project | ⚠️ Synthetic Data Validation Only")
