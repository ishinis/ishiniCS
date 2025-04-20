import streamlit as st
import pandas as pd
import numpy as np
import joblib
import torch
from pytorch_tabnet.tab_model import TabNetClassifier
from sklearn.metrics import classification_report, confusion_matrix
import plotly.express as px
import plotly.graph_objects as go
from sklearn.linear_model import LogisticRegression

# --- UI Setup ---
st.set_page_config(page_title="Credit Scoring App", layout="wide")

st.markdown("""
    <style>
    .stApp {}
    .header { text-align: center; padding: 2rem 0; }
    .footer { text-align: center; font-size: small; color: #888; margin-top: 3rem; }
    .stButton > button { color: white; background-color: #1f77b4; border-radius: 8px; }
    .stButton > button:hover { background-color: #2980b9; }
    </style>
""", unsafe_allow_html=True)

# --- Authentication ---
if 'logged_in' not in st.session_state:
    st.session_state.logged_in = False
if not st.session_state.logged_in:
    st.title("🔐 Login")
    user = st.text_input("Username")
    pwd = st.text_input("Password", type="password")
    if st.button("Login"):
        if user == "admin" and pwd == "password":
            st.session_state.logged_in = True
            st.success("Login successful")
            st.rerun()
        else:
            st.error("Invalid credentials")
    st.stop()

# --- Logout ---
st.sidebar.success("Logged in as Admin")
if st.sidebar.button("🚪 Logout"):
    st.session_state.logged_in = False
    st.rerun()

# --- Load Models ---
@st.cache_resource
def load_models():
    scaler = joblib.load("scaler.pkl")
    le = joblib.load("label_encoder.pkl")
    xgb = joblib.load("base_xgb_model.pkl")
    lgbm = joblib.load("base_lgbm_model.pkl")
    catb = joblib.load("base_catboost_model.pkl")
    stack = joblib.load("stacking_meta_model.pkl")
    tabnet = TabNetClassifier()
    tabnet.load_model("tabnet_model.zip")
    X_train = pd.read_csv("X_train.csv")
    expected_columns = X_train.columns.tolist()
    return scaler, le, xgb, lgbm, catb, stack, tabnet, expected_columns, X_train

scaler, le, xgb, lgbm, catb, stack, tabnet, expected_columns, X_train = load_models()

st.markdown("""
<div class='header'><h1>💳 Credit Score Predictor</h1>
<p>This intelligent system helps you predict an individual's credit score using an ensemble of powerful machine learning models. It takes into account detailed financial behavior, credit history, and spending patterns to classify creditworthiness as Good, Standard, or Poor.</p>
<p>The app uses models like <b>XGBoost</b>, <b>LightGBM</b>, <b>CatBoost</b>, <b>TabNet</b>, and a <b>Logistic Regression Stacking Ensemble</b> for final prediction, offering confidence scores and interpretability visualizations.</p>
<p><b style='font-size: 1.2em;'>Enter customer profile data below</b></p>
</div>
""", unsafe_allow_html=True)

# [FORM CONTENT REMAINS SAME UP TO PREDICTION HANDLING...]

# --- Radar Chart Enhancement ---
        st.subheader("📈 Financial Behavior")
        feature_scores = {
            'Debt-to-Income': data["Debt_to_Income_Ratio"],
            'Credit Utilization': data["Credit_Utilization_Ratio"],
            'EMI/Income': data["EMI_to_Income_Ratio"],
            'Delayed Payments': data["Num_of_Delayed_Payment"] / 20,
            'Credit Age': data["Credit_History_Age_Months"] / 360
        }

        # Average Profile for Comparison (example values)
        avg_good_profile = {
            'Debt-to-Income': 0.2,
            'Credit Utilization': 0.3,
            'EMI/Income': 0.2,
            'Delayed Payments': 0.05,
            'Credit Age': 0.6
        }

        radar_df = pd.DataFrame({
            "Metric": list(feature_scores.keys()),
            "User": list(feature_scores.values()),
            "Benchmark (Good Score Avg)": list(avg_good_profile.values())
        })

        fig_radar = go.Figure()
        fig_radar.add_trace(go.Scatterpolar(r=radar_df['User'],
                                            theta=radar_df['Metric'],
                                            fill='toself',
                                            name='User Profile',
                                            line=dict(color='blue')))

        fig_radar.add_trace(go.Scatterpolar(r=radar_df['Benchmark (Good Score Avg)'],
                                            theta=radar_df['Metric'],
                                            fill='toself',
                                            name='Benchmark (Good)',
                                            line=dict(color='gray', dash='dot')))

        fig_radar.update_layout(
            polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
            showlegend=True,
            title="📊 Financial Behavior Profile with Benchmarks",
            annotations=[
                dict(text="Guidance:\n- Lower is better for Debt-to-Income, Utilization, EMI/Income, Delays\n- Higher is better for Credit Age",
                     xref="paper", yref="paper", x=0, y=-0.3, showarrow=False, font=dict(size=12))
            ]
        )

        st.plotly_chart(fig_radar)

# --- FOOTER ---
st.markdown("<div class='footer'>© 2025 Credit AI System | Built By Ishini</div>", unsafe_allow_html=True)
