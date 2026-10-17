import os
import pandas as pd
import numpy as np
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from model import FEATURES, TARGET, train_ensemble, predict, ensemble_predict

st.set_page_config(page_title="Microalgae Biomass AI Dashboard", page_icon="🌱", layout="wide")

DATA_PATH = os.path.join(os.path.dirname(__file__), "data", "algal_growth_data.csv")

@st.cache_data
def load_data(path):
    return pd.read_csv(path)

@st.cache_resource
def get_models(df_hash):
    df = load_data(DATA_PATH)
    return train_ensemble(df, n=5)

df = load_data(DATA_PATH)
models = get_models(tuple(df.to_numpy().ravel()))

st.markdown("# 🌱 Microalgae Biomass AI Dashboard")
st.caption("Physics-Informed Neural Network (PINN) prototype • 100-day experimental dataset")

# Sidebar
st.sidebar.header("Navigation")
page = st.sidebar.radio("Go to", ["📊 Experiment Overview", "🔮 Predict Current Day", "📁 Analyze New CSV", "🧠 Model & Project"])

# KPI cards
c1,c2,c3,c4 = st.columns(4)
c1.metric("Days recorded", f"{len(df)}")
c2.metric("Peak biomass", f"{df.Biomass.max():.2f} g/L", f"Day {int(df.loc[df.Biomass.idxmax(),'Day'])}")
c3.metric("Latest biomass", f"{df.iloc[-1].Biomass:.2f} g/L")
c4.metric("Latest OD680", f"{df.iloc[-1].OD680:.2f}")

if page == "📊 Experiment Overview":
    st.subheader("1. What happened during the 100-day experiment?")
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df.Day,y=df.Biomass,mode="lines+markers",name="Measured biomass",line=dict(width=3)))
    fig.add_trace(go.Scatter(x=df.Day,y=df.OD680*0.70,mode="lines+markers",name="OD680 × 0.70 (visual scale)",line=dict(dash="dot")))
    fig.update_layout(height=430,xaxis_title="Day",yaxis_title="Biomass (g/L)",hovermode="x unified")
    st.plotly_chart(fig,use_container_width=True)

    st.subheader("2. Sensor conditions")
    f2=make_subplots(rows=3,cols=1,shared_xaxes=True,vertical_spacing=0.08,subplot_titles=("Temperature","PAR light","pH and OD680"))
    f2.add_trace(go.Scatter(x=df.Day,y=df.Temperature,mode="lines+markers",name="Temperature (°C)"),row=1,col=1)
    f2.add_trace(go.Scatter(x=df.Day,y=df.PAR,mode="lines+markers",name="PAR (µmol/m²/s)"),row=2,col=1)
    f2.add_trace(go.Scatter(x=df.Day,y=df.pH,mode="lines+markers",name="pH"),row=3,col=1)
    f2.add_trace(go.Scatter(x=df.Day,y=df.OD680,mode="lines+markers",name="OD680"),row=3,col=1)
    f2.update_layout(height=700,hovermode="x unified")
    st.plotly_chart(f2,use_container_width=True)

    st.subheader("3. Additional environmental parameters")
    f3=make_subplots(rows=3,cols=1,shared_xaxes=True,vertical_spacing=0.08,subplot_titles=("Nitrogen (mg/L)","CO2 (mg/L)","Dissolved Oxygen (mg/L)"))
    f3.add_trace(go.Scatter(x=df.Day,y=df.Nitrogen_mg_L,mode="lines+markers",name="Nitrogen",line=dict(color="green")),row=1,col=1)
    f3.add_trace(go.Scatter(x=df.Day,y=df.CO2_mg_L,mode="lines+markers",name="CO2",line=dict(color="orange")),row=2,col=1)
    f3.add_trace(go.Scatter(x=df.Day,y=df.Dissolved_Oxygen_mg_L,mode="lines+markers",name="Dissolved Oxygen",line=dict(color="blue")),row=3,col=1)
    f3.update_layout(height=700,hovermode="x unified")
    st.plotly_chart(f3,use_container_width=True)

    st.subheader("4. Dataset interpretation")
    st.info("Biomass rises from 0.08 g/L on Day 1 to 1.732 g/L on Day 100. The growth shows exponential phase (Days 1-50) followed by a slower stationary phase (Days 50-100), indicating a mature culture approaching maximum carrying capacity. Nitrogen levels decline from 56.5 to 4.35 mg/L, showing nutrient consumption.")
    st.dataframe(df,use_container_width=True,hide_index=True)

elif page == "🔮 Predict Current Day":
    st.subheader("Enter today's sensor readings")
    st.write("Give the model the current day's measurements. It returns an estimated biomass for that day.")
    
    st.markdown("##### Core Parameters (Used by Model)")
    last_day=float(df.Day.max())
    a,b,c=st.columns(3)
    day=a.number_input("Day",min_value=1.0,value=last_day+1,step=1.0)
    temp=b.number_input("Temperature (°C)",value=float(df.iloc[-1].Temperature),step=0.1,format="%.1f")
    par=c.number_input("Light PAR (µmol/m²/s)",value=float(df.iloc[-1].PAR),step=1.0)
    d,e=st.columns(2)
    ph=d.number_input("pH",value=float(df.iloc[-1].pH),step=0.1,format="%.1f")
    od=e.number_input("OD680",value=float(df.iloc[-1].OD680),step=0.01,format="%.2f")
    
    st.markdown("##### Additional Environmental Parameters (For Record)")
    f,g,h=st.columns(3)
    nitrogen=f.number_input("Nitrogen (mg/L)",value=float(df.iloc[-1].Nitrogen_mg_L),step=0.1,format="%.2f")
    co2=g.number_input("CO2 (mg/L)",value=float(df.iloc[-1].CO2_mg_L),step=0.01,format="%.3f")
    do=h.number_input("Dissolved Oxygen (mg/L)",value=float(df.iloc[-1].Dissolved_Oxygen_mg_L),step=0.1,format="%.2f")
    
    row=pd.DataFrame([{"Day":day,"Temperature":temp,"PAR":par,"pH":ph,"OD680":od,"Nitrogen_mg_L":nitrogen,"CO2_mg_L":co2,"Dissolved_Oxygen_mg_L":do}])
    if st.button("🚀 Predict Biomass",type="primary",use_container_width=True):
        mean,spread,preds=ensemble_predict(models,row)
        st.success("Prediction generated from the current sensor inputs.")
        p1,p2,p3=st.columns(3)
        p1.metric("Estimated biomass",f"{mean:.3f} g/L")
        p2.metric("Model spread",f"±{spread:.3f} g/L")
        p3.metric("Latest measured biomass",f"{df.iloc[-1].Biomass:.3f} g/L")
        
        # Display input summary
        st.markdown("##### Input Summary")
        col1,col2=st.columns(2)
        with col1:
            st.markdown(f"""
            **Core Parameters:**
            - Day: {day:.0f}
            - Temperature: {temp:.1f} °C
            - PAR: {par:.1f} µmol/m²/s
            - pH: {ph:.2f}
            - OD680: {od:.2f}
            """)
        with col2:
            st.markdown(f"""
            **Environmental Parameters:**
            - Nitrogen: {nitrogen:.2f} mg/L
            - CO2: {co2:.3f} mg/L
            - Dissolved Oxygen: {do:.2f} mg/L
            """)
        
        fig=go.Figure()
        fig.add_trace(go.Scatter(x=df.Day,y=df.Biomass,mode="lines+markers",name="Historical measured biomass"))
        fig.add_trace(go.Scatter(x=[day],y=[mean],mode="markers+text",text=[f"{mean:.3f} g/L"],textposition="top center",marker=dict(size=14),name="Current prediction"))
        fig.update_layout(height=430,xaxis_title="Day",yaxis_title="Biomass (g/L)",hovermode="x unified")
        st.plotly_chart(fig,use_container_width=True)
        st.warning("This is a model estimate, not a laboratory measurement. The training dataset has 100 days covering a complete growth cycle, but predictions outside the observed range should still be validated with additional experiments.")
        st.info("Note: The PINN model currently uses only the 5 core parameters (Day, Temperature, PAR, pH, OD680) for prediction. Additional environmental parameters are recorded for reference.")
    else:
        st.info("Fill in the sensor values and press **Predict Biomass**.")

elif page == "📁 Analyze New CSV":
    st.subheader("Upload a new sensor dataset")
    st.write("CSV must contain: Day, Temperature, PAR, pH, OD680. Biomass is optional for prediction.")
    uploaded=st.file_uploader("Upload CSV",type=["csv"])
    if uploaded:
        newdf=pd.read_csv(uploaded)
        missing=[x for x in FEATURES if x not in newdf.columns]
        if missing:
            st.error("Missing columns: "+", ".join(missing))
        else:
            newdf["Predicted_Biomass_g_L"]=predict(models[0][0],models[0][1],newdf)
            st.dataframe(newdf,use_container_width=True,hide_index=True)
            fig=go.Figure()
            fig.add_trace(go.Scatter(x=newdf.Day,y=newdf.Predicted_Biomass_g_L,mode="lines+markers",name="Predicted biomass"))
            if TARGET in newdf.columns:
                fig.add_trace(go.Scatter(x=newdf.Day,y=newdf[TARGET],mode="lines+markers",name="Measured biomass"))
            fig.update_layout(height=430,xaxis_title="Day",yaxis_title="Biomass (g/L)")
            st.plotly_chart(fig,use_container_width=True)
            st.download_button("⬇️ Download predictions CSV",newdf.to_csv(index=False),file_name="biomass_predictions.csv",mime="text/csv")
    else:
        st.info("Upload a CSV to see predictions.")

else:
    st.subheader("🧠 How this project works")
    st.markdown("""
    **Inputs:** Day + Temperature + PAR + pH + OD680  
    **Target:** Biomass Density (g/L)

    The model is a small **Physics-Informed Neural Network**. It learns from the measured biomass while also receiving a growth-kinetics constraint based on temperature, pH and light. Five independently initialized models are used in the dashboard; their average is shown as the prediction and their spread is shown as a model-spread indicator.
    """)
    st.subheader("Why OD680 is important")
    st.write("OD680 is strongly associated with biomass in this dataset. That makes it useful for low-cost real-time estimation, but it also means the current 100-day dataset is not enough by itself to prove robustness under noisy or missing sensors.")
    st.subheader("Recommended next FYP experiment")
    st.write("Add multiple cultivation runs and deliberately introduce sensor noise, missing values and different growth conditions. Then compare PINN predictions with a data-only neural network and a simple OD680 calibration model using unseen days/runs.")
    st.subheader("Current dataset (100 days)")
    st.dataframe(df,use_container_width=True,hide_index=True)
