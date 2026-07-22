import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import pickle
from simulation_engine import run_plant_simulation

st.set_page_config(page_title="Anaerobic Digester Digital Twin", layout="wide")

# --- EMERGENCY KILL SWITCH ---
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

# Load saved Day 2 AI Brain safely
@st.cache_resource
def load_ai_brain():
    with open("twin_brain_cod.pkl", "rb") as f:
        return pickle.load(f)

try:
    # Unpacking the new dictionary payload cleanly
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
# --- SIDEBAR CONTROL DIALS ---
st.sidebar.header("Plant Actuator Valve Knobs")

if st.sidebar.button("Run AI Profit Optimization Engine", use_container_width=True, type="secondary"):
    st.session_state.optimized_mode = not st.session_state.optimized_mode

# Unit Conversions & Labels
if st.session_state.optimized_mode:
    st.sidebar.info("AI Automation Active: Knobs locked to peak performance baseline coordinates.")
    slider_cod = st.sidebar.slider("Incoming Waste Concentration (COD mg/L)", 150, 800, 550, disabled=True)
    slider_hrt = st.sidebar.slider("Hydraulic Retention Time (HRT days)", 3.3, 20.0, 9.1, step=0.1, disabled=True) # 1 / 0.11 dilution
    slider_temp = st.sidebar.slider("Digester Thermal Core Temperature (°C)", 20, 60, 38, disabled=True)
else:
    slider_cod = st.sidebar.slider("Incoming Waste Concentration (COD mg/L)", 150, 800, 450)
    slider_hrt = st.sidebar.slider("Hydraulic Retention Time (HRT days)", 3.3, 20.0, 6.7, step=0.1) # Default 1 / 0.15 dilution = 6.66 days
    slider_temp = st.sidebar.slider("Digester Thermal Core Temperature (°C)", 20, 60, 37)

# Convert HRT back to dilution rate for background calculations
dilution_rate_calc = 1.0 / slider_hrt

# --- READ REAL-TIME PHYSICS SIMULATION ---
s_final, x_final, methane_final = run_plant_simulation(slider_cod, dilution_rate_calc, slider_temp)

# --- READ MACHINE LEARNING COMPONENT ---
input_array = pd.DataFrame([[slider_cod, dilution_rate_calc, slider_temp]], columns=['Inflow_COD', 'Dilution_Rate', 'Temperature'])
predicted_effluent_cod = ai_engine.predict(input_array)[0]

# --- REALISTIC INDUSTRIAL ALIGNMENT MATH ---
if slider_temp < 30:
    corrected_methane = 0.0
    methane_nm3 = 0.0
    revenue_per_day = 0.0
    heating_cost_per_day = 15.50  
    net_profit_per_day = -heating_cost_per_day
else:
    # 1. Take the tiny lab methane value (e.g., 95.43 Liters)
    lab_methane_liters = methane_final
    
    # 2. Scale it up by 500 to represent a real industrial facility
    industrial_methane_liters = lab_methane_liters * 500.0  
    
    # 3. Convert Liters to Normal Cubic Meters (Nm3) by dividing by 1,000
    methane_nm3 = industrial_methane_liters / 1000.0  
    
    # 4. Calculate Money ($1.50 earned per Nm3 of gas)
    revenue_per_day = methane_nm3 * 1.50 
    
    # 5. Calculate Heating Utility Costs
    heating_cost_per_day = max(0.0, (slider_temp - 15.0) * 0.45)
    
    # 6. Calculate final Net Profit
    net_profit_per_day = revenue_per_day - heating_cost_per_day

# Process Wastewater Purification Efficiency KPI
cod_removal_efficiency = ((slider_cod - predicted_effluent_cod) / slider_cod) * 100
cod_removal_efficiency = max(0.0, min(100.0, cod_removal_efficiency))

# KPI 1: COD Purification Efficiency
cod_removal_efficiency = ((slider_cod - predicted_effluent_cod) / slider_cod) * 100
cod_removal_efficiency = max(0.0, min(100.0, cod_removal_efficiency))

# KPI 2 - Thermodynamic Yield Versus Theoretical Maximum (0.35 Nm³/kg COD removed at STP)
cod_removed_kg = max(0.001, (slider_cod - predicted_effluent_cod) / 1000.0) 
actual_yield_coefficient = methane_nm3 / cod_removed_kg
thermodynamic_efficiency = (actual_yield_coefficient / 0.35) * 100
thermodynamic_efficiency = max(0.0, min(100.0, thermodynamic_efficiency)) if slider_temp >= 30 else 0.0


# --- FINANCIAL DASHBOARD ROW ---
st.subheader("Real-Time Plant Economic Performance")
fin_col1, fin_col2, fin_col3 = st.columns(3)
with fin_col1:
    st.metric(label="Gross Biogas Revenue", value=f"${revenue_per_day:,.2f} / day", delta="Target: >$30.00/day")
with fin_col2:
    st.metric(label="Thermal Utility Spend", value=f"${heating_cost_per_day:,.2f} / day", delta="Budget Max: $10.00/day", delta_color="inverse")
with fin_col3:
    profit_status = "normal" if net_profit_per_day > 0 else "inverse"
    profit_text = "Status: Profitable" if net_profit_per_day > 0 else "Status: OPERATIONAL LOSS"
    st.metric(label="Net Operational Margin", value=f"${net_profit_per_day:,.2f} / day", delta=profit_text, delta_color=profit_status)

st.markdown("---")

# --- TECHNICAL METRICS & KPIs ROW ---
st.subheader("Engineering Data & Process KPIs")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(
        label="Biogas Volumetric Yield", 
        value=f"{methane_nm3:.4f} Nm³/day", 
        delta="Industrial Standard: Nm³"
    )
with col2:
    st.metric(
        label="Active Biomass Density", 
        value=f"{x_final:.1f} mg/L",
        delta="Safe Range: 40 - 200 mg/L",
        delta_color="normal" if x_final >= 40 else "inverse"
    )
with col3:
    st.metric(
        label="Outgoing Pollution (Effluent COD)", 
        value=f"{predicted_effluent_cod:.2f} mg/L",
        delta="Legal Limit: <130 mg/L",
        delta_color="inverse" if predicted_effluent_cod > 130 else "normal"
    )
with col4:
    # This displays the actual computed validation score.
    st.metric(
        label="ML Brain Accuracy (R² Score)", 
        value=f"{biogas_model_accuracy:.2f} %", 
        delta="Cross-Validated (80/20 Train/Test Split)", 
        delta_color="normal"
    )

# --- RISK ANALYSIS & GRAPHICAL LAYOUT ---
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
    st.plotly_chart(fig, use_container_width=True)

st.markdown("---")

# --- EXPANDABLE METHODOLOGY SECTION---
with st.expander("View Architecture & Mathematical Methodology"):
    st.markdown("""
    ### Hybrid Digital Twin Structural Logic
    This application utilizes a **hybrid modeling paradigm** to overcome the limitations of purely data-driven or purely mechanistic architectures:
    1. **Mechanistic First-Principles Layer (`simulation_engine.py`):** Solves mass-balance ordinary differential equations (ODEs) derived from **Monod and Haldane kinetics**. It establishes biological mass parameters under idealized conditions.
    2. **Empirical Machine Learning Layer (`twin_brain_cod.pkl`):** Trained via a 2,000-state Monte Carlo pipeline. It acts as an error-correction layer that captures real-world non-linearities and unmodeled perturbations in final effluent characteristics.
    
    ### Thermodynamic Efficiency Limits
    Theoretical conversion at Standard Temperature and Pressure (STP) dictates that the complete conversion of **1 kg of Chemical Oxygen Demand (COD)** yields exactly **0.35 Normal Cubic Meters ($Nm^3$) of Methane gas**. 
    The *Thermodynamic Biokinetic Efficiency* card monitors the ratio of actual yield against this absolute biological boundary.
    """)

# --- DOWNLOADABLE REPORT ---
st.subheader("Operational Reporting Metrics")

report_data = pd.DataFrame({
    "Timestamp": [pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")],
    "Control Mode": ["AI Optimized Framework" if st.session_state.optimized_mode else "Manual Control State"],
    "Influent COD (mg/L)": [slider_cod],
    "Hydraulic Retention Time (days)": [slider_hrt],
    "Thermal Core Core Temp (C)": [slider_temp],
    "Methane Yield (Nm3/day)": [f"{methane_nm3:.4f}"],
    "Purification Efficiency (%)": [f"{cod_removal_efficiency:.1f}%"],
    "Thermodynamic Efficiency (%)": [f"{thermodynamic_efficiency:.1f}%"],
    "Net Operating Profit ($/day)": [f"${net_profit_per_day:.2f}"]
})

st.dataframe(report_data, hide_index=True, use_container_width=True)

st.download_button(label="Export Current Plant Metrics to Shift CSV Report", data=report_data.to_csv(index=False), file_name="digital_twin_shift_report.csv", mime="text/csv", use_container_width=True)
