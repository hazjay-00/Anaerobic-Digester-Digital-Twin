import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import pickle
import os  
from simulation_engine import run_plant_simulation

st.set_page_config(page_title="Anaerobic Digester Digital Twin", layout="wide")

# EMERGENCY KILL SWITCH
if "kill_switch" not in st.session_state:
    st.session_state.kill_switch = False

if st.button("EMERGENCY SYSTEM KILL SWITCH", use_container_width=True, type="primary"):
    st.session_state.kill_switch = not st.session_state.kill_switch

if st.session_state.kill_switch:
    st.error("SYSTEM LOCKED: Emergency shutdown activated. Feed valves isolated. Core thermal element deactivated.")
    st.info("To restore plant operations, clear site faults and toggle the Kill Switch button again.")
    st.stop()

st.title("Anaerobic Digester Digital Twin")
st.caption("Hybrid Monod-Haldane Kinetics & Empirical Effluent Modeling")
st.markdown("---")

# AUTO-TRAIN CHECK FOR STREAMLIT CLOUD
if not os.path.exists("twin_brain_cod.pkl"):
    st.info("Generating dataset and training AI Brain model for first-time setup...")
    import ml_agent
    df_sim = ml_agent.generate_synthetic_industrial_dataset()
    ml_agent.train_ai_operator_engine(df_sim)

# Load AI Brain safely
@st.cache_resource
def load_ai_brain():
    with open("twin_brain_cod.pkl", "rb") as f:
        return pickle.load(f)

try:
    # Unpacking the dictionary payload
    artifacts = load_ai_brain()
    ai_engine = artifacts["model"]
    biogas_model_accuracy = artifacts["r2_score"]
except FileNotFoundError:
    st.error("Please run ml_agent.py first to generate the data and train the AI brain model!")
    st.stop()
except TypeError:
    st.warning("Upgrading payload format... Please re-run ml_agent.py in your terminal first to synchronize files.")
    st.stop()

if "optimized_mode" not in st.session_state:
    st.session_state.optimized_mode = False

# SIDEBAR CONTROL DIALS
st.sidebar.header("Plant Actuator Valve Knobs")

if st.sidebar.button("Run AI Profit Optimization Engine", use_container_width=True, type="secondary"):
    st.session_state.optimized_mode = not st.session_state.optimized_mode

# Unit Conversions and Labels
if st.session_state.optimized_mode:
    st.sidebar.info("AI Automation Active: Knobs locked to peak performance baseline coordinates.")
    slider_cod = st.sidebar.slider("Incoming Waste Concentration (COD mg/L)", 150, 800, 750, disabled=True)
    slider_hrt = st.sidebar.slider("Hydraulic Retention Time (HRT days)", 3.3, 20.0, 15.0, step=0.1, disabled=True)
    slider_temp = st.sidebar.slider("Digester Thermal Core Temperature (°C)", 25, 45, 37, disabled=True)
else:
    slider_cod = st.sidebar.slider("Incoming Waste Concentration (COD mg/L)", 150, 800, 450)
    slider_hrt = st.sidebar.slider("Hydraulic Retention Time (HRT days)", 3.3, 20.0, 6.7, step=0.1)
    slider_temp = st.sidebar.slider("Digester Thermal Core Temperature (°C)", 20, 60, 37)

# Convert HRT back to dilution rate
dilution_rate_calc = 1.0 / slider_hrt

# READ REAL-TIME PHYSICS SIMULATION & ML BRAIN
s_final, x_final, methane_final = run_plant_simulation(slider_cod, dilution_rate_calc, slider_temp)

input_array = pd.DataFrame([[slider_cod, dilution_rate_calc, slider_temp]], 
                           columns=['Inflow_COD', 'Dilution_Rate', 'Temperature'])
predicted_effluent_cod = ai_engine.predict(input_array)[0]

# UNLIMITED RAW DATASET CALCULATIONS
methane_liters = methane_final
methane_nm3_lab = methane_liters / 1000.0 

# Industrial 500 m³ scaling
industrial_scale_factor = 500.0
methane_liters_industrial = methane_liters * industrial_scale_factor
methane_nm3 = methane_liters_industrial / 1000.0 

# COD Removal Mass (kg/m³) & Thermodynamic Efficiency
cod_removed_mg_l = slider_cod - predicted_effluent_cod

if cod_removed_mg_l <= 5.0 or methane_nm3_lab <= 0:
    thermodynamic_efficiency = 0.0
else:
    cod_removed_kg_m3 = cod_removed_mg_l / 1000.0
    actual_yield_coefficient = methane_nm3_lab / cod_removed_kg_m3
    # Cap thermodynamic efficiency at 100% STP maximum (0.35 Nm³/kg COD)
    thermodynamic_efficiency = min(100.0, (actual_yield_coefficient / 0.35) * 100.0)

# Dynamic Financial Metrics
revenue_per_day = methane_nm3 * 0.80 
heating_cost_per_day = max(0.0, (slider_temp - 15.0) * 0.45)
net_profit_per_day = revenue_per_day - heating_cost_per_day

# Process Wastewater Purification Efficiency KPI
cod_removal_efficiency = ((slider_cod - predicted_effluent_cod) / slider_cod) * 100
cod_removal_efficiency = max(0.0, min(100.0, cod_removal_efficiency))

# Helper function for status badges
def status_badge(text, is_ok):
    color = "#2ecc71" if is_ok else "#e74c3c"
    return f"<span style='color:{color}; font-size: 0.85rem; font-weight: 500;'>{text}</span>"

# FINANCIAL DASHBOARD ROW
st.subheader("Real-Time Plant Economic Performance")
fin_col1, fin_col2, fin_col3 = st.columns(3)

with fin_col1:
    ok_rev = revenue_per_day >= 30.0
    st.metric(label="Gross Biogas Revenue", value=f"${revenue_per_day:,.2f} / day")
    st.markdown(status_badge("Target: >$30.00/day" if ok_rev else "Low Revenue (<$30)", ok_rev), unsafe_allow_html=True)

with fin_col2:
    ok_spend = heating_cost_per_day <= 10.0
    st.metric(label="Thermal Utility Spend", value=f"${heating_cost_per_day:,.2f} / day")
    st.markdown(status_badge("Budget Max: $10.00/day" if ok_spend else "Over Budget (>$10)", ok_spend), unsafe_allow_html=True)

with fin_col3:
    target_profit = 20.0
    ok_profit = net_profit_per_day >= target_profit
    st.metric(label="Net Operational Profit", value=f"${net_profit_per_day:,.2f} / day")
    st.markdown(status_badge("Target: >$20.00/day" if ok_profit else "Low Margin (<$20)", ok_profit), unsafe_allow_html=True)

st.markdown("---")

# TECHNICAL METRICS AND KPIs
st.subheader("Engineering Data & Process KPIs")
col1, col2, col3, col4, col5 = st.columns(5)

with col1:
    ok1 = methane_nm3 >= 20.0
    st.metric(label="Biogas Volumetric Yield", value=f"{methane_nm3:.1f} Nm³/day")
    st.markdown(status_badge("Target: >20.0 Nm³/day" if ok1 else "Low Yield (<20)", ok1), unsafe_allow_html=True)

with col2:
    ok2 = 40.0 <= x_final <= 200.0
    st.metric(label="Active Biomass Density", value=f"{x_final:.1f} mg/L")
    st.markdown(status_badge("Safe: 40-200 mg/L" if ok2 else "Out of Safe Bounds", ok2), unsafe_allow_html=True)

with col3:
    ok3 = predicted_effluent_cod <= 130.0
    st.metric(label="Outgoing Pollution (COD)", value=f"{predicted_effluent_cod:.1f} mg/L")
    st.markdown(status_badge("Limit: <130 mg/L" if ok3 else "Non-Compliant (>130)", ok3), unsafe_allow_html=True)

with col4:
    ok4 = biogas_model_accuracy >= 90.0
    st.metric(label="ML Brain Accuracy (R²)", value=f"{biogas_model_accuracy:.1f}%")
    st.markdown(status_badge("Benchmark: >90%" if ok4 else "Low Accuracy (<90%)", ok4), unsafe_allow_html=True)

with col5:
    ok5 = thermodynamic_efficiency >= 50.0 
    st.metric(
        label="Thermodynamic Efficiency", 
        value=f"{thermodynamic_efficiency:.1f}%",
        help="Actual conversion relative to theoretical maximum (0.35 Nm³/kg COD at STP)"
    )
    st.markdown(status_badge("Benchmark: ≥50%" if ok5 else "Sub-optimal (<50%)", ok5), unsafe_allow_html=True)
    
# RISK ANALYSIS AND GRAPHICAL LAYOUT
ENVIRONMENTAL_LIMIT_COD = 130.0
col_left, col_right = st.columns(2)

with col_left:
    st.subheader("Comprehensive Process Safety Protocols")
    
    if slider_temp < 30:
        st.error(f"THERMAL ACCLIMATIZATION BREACH ({slider_temp}°C): Methanogenic enzyme kinetics restricted. High risk of VFA (Volatile Fatty Acid) accumulation.")
    elif 30 <= slider_temp < 35:
        st.warning(f"Sub-Optimal Mesophilic Window ({slider_temp}°C): Reduced specific growth rate (μ) according to Haldane-type constraints.")
    elif 35 <= slider_temp <= 39:
        st.success(f"Optimal Operating Zone ({slider_temp}°C): Maximum enzymatic substrate utilization rate achieved.")
    elif 39 < slider_temp < 50:
        st.warning(f"High Thermal Stress ({slider_temp}°C): Acidification imbalance; risk of biokinetic souring.")
    else:
        st.error(f"CRITICAL THERMAL DEACTIVATION ({slider_temp}°C): Cellular protein denaturation imminent. Complete biomass washout expected.")

    if predicted_effluent_cod > ENVIRONMENTAL_LIMIT_COD:
        st.error(f"REGULATORY NON-COMPLIANCE: Discharged Effluent ({predicted_effluent_cod:.1f} mg/L) exceeds environmental threshold ({ENVIRONMENTAL_LIMIT_COD} mg/L).")
        st.warning("PROCESS CORRECTION: Increase Hydraulic Retention Time (HRT) to scale up substrate biodegradation residence window.")
    else:
        st.success("ENVIRONMENTAL COMPLIANCE: Outgoing effluent meets strict wastewater discharge directives.")
        
    if x_final < 40.0:
        st.error("BIOMASS WASHOUT CRISIS: Hydraulic dilution rate exceeds maximum specific growth rate (D > μ_max). System wash out active.")

with col_right:
    st.subheader("Substrate Degradation Vector Profile (COD Drop)")
    
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=['Influent Waste Load', 'Predicted Effluent Outflow', 'Regulatory Discharge Limit'],
        y=[slider_cod, predicted_effluent_cod, ENVIRONMENTAL_LIMIT_COD],
        marker_color=['#E67E22', '#2ECC71', '#E74C3C']
    ))
    fig.update_layout(yaxis_title="Chemical Oxygen Demand (mg/L)", template="plotly_white", height=320)
    fig.update_layout(
        xaxis_title="",
        margin=dict(b=60, t=20, l=10, r=10),
    )
    fig.update_xaxes(tickangle=0)
    st.plotly_chart(fig, use_container_width=True)

st.markdown("---")

# EXPANDABLE METHODOLOGY SECTION
with st.expander("View Architecture & Mathematical Methodology"):
    st.markdown("""
    ### Hybrid Digital Twin Structural Logic
    This application utilizes a **hybrid modeling paradigm** to overcome the limitations of purely data-driven or purely mechanistic architectures:
    1. **Mechanistic First-Principles Layer (`simulation_engine.py`):** Solves mass-balance ordinary differential equations (ODEs) derived from **Monod and Haldane kinetics**. It establishes biological mass parameters under idealized conditions.
    2. **Empirical Machine Learning Layer (`twin_brain_cod.pkl`):** Trained via a 2,000-state Monte Carlo pipeline. It acts as an error-correction layer that captures real-world non-linearities and unmodeled perturbations in final effluent characteristics.
    
    ### Thermodynamic Efficiency Limits
    Theoretical conversion at Standard Temperature and Pressure (STP) dictates that the complete conversion of **1 kg of Chemical Oxygen Demand (COD)** yields exactly **0.35 Normal Cubic Meters ($Nm^3$) of Methane gas**. 
    The *Thermodynamic Efficiency* card monitors the ratio of actual yield against this absolute biological boundary.
    """)

# DOWNLOADABLE REPORT
st.subheader("Operational Reporting Metrics")

report_data = pd.DataFrame({
    "Timestamp": [pd.Timestamp.now(tz="UTC").strftime("%Y-%m-%d %H:%M:%S UTC")],
    "Influent COD (mg/L)": [slider_cod],
    "Hydraulic Retention Time (Days)": [slider_hrt],
    "Operating Temperature (°C)": [slider_temp],
    "Effluent COD (mg/L)": [round(predicted_effluent_cod, 2)],
    "Biogas Yield (Nm³/day)": [round(methane_nm3, 2)],
    "Thermodynamic Efficiency (%)": [round(thermodynamic_efficiency, 2)],
    "Net Operational Profit ($/day)": [round(net_profit_per_day, 2)]
})

st.dataframe(report_data, hide_index=True, use_container_width=True)

st.download_button(
    label="Export Current Plant Metrics to Shift CSV Report",
    data=report_data.to_csv(index=False),
    file_name="ad_twin_shift_report.csv",
    mime="text/csv",
    use_container_width=True
)
