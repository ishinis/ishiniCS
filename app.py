﻿import streamlit as st
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
    .stApp {  }
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
@st.cache_resource  # Cache the model loading
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


st.markdown("<div class='header'><h1>💳 Credit Score Predictor</h1> <p>This intelligent system helps you predict an individual's credit score using an ensemble of powerful machine learning models. It takes into account detailed financial behavior, credit history, and spending patterns to classify creditworthiness as Good, Standard, or Poor.</p><p>The app uses models like <b>XGBoost</b>, <b>LightGBM</b>, <b>CatBoost</b>, <b>TabNet</b>, and a <b>Logistic Regression Stacking Ensemble</b> for final prediction, offering confidence scores and interpretability visualizations.</p><p><b style='font-size: 1.2em;'>Enter customer profile data below</b></p></div>", unsafe_allow_html=True)

with st.form("form"):
    st.subheader("🔍 Basic Information")
    col1, col2 = st.columns(2)
    with col1:
        age = st.slider("Age", 18, 75, 30)
        annual_income = st.number_input("Annual Income (GBP)", 10000, 500000, 30000)
        salary = st.number_input("Monthly Take Home Salary (GBP)", 100, 5000, 1500)
        num_accounts = st.number_input("Bank Accounts", min_value=0, max_value=15, value=3)
        credit_years = st.slider("Credit History (Years)", 0, 30, 5)
    with col2:
        num_loans = st.slider("Number of Active Loans", 0, 10, 2)
        num_cards = st.slider("Number of Credit Cards", 0, 10, 3)
        num_delays = st.slider("Number of Delayed Payments", 0, 20, 2)
        delay_days = st.number_input("Avg Delay From Due Date (Days)", min_value=0, max_value=100, value=5)
        num_inquiries = st.slider("Credit Inquiries This Year", 0, 15, 2)

    st.subheader("💳 Financial Details")
    col3, col4 = st.columns(2)
    with col3:
        total_emi = st.number_input("Total EMI per Month (GBP)", 0, 10000, 3000)
        outstanding_debt = st.number_input("Outstanding Debt (GBP)", 0, 10000, 3000)
        interest_rate = st.slider("Percentage Average Interest Rate of Active Loans (%)", 0.0, 50.0, 13.5)
        monthly_balance = st.number_input("Monthly Balance After Expenses (GBP)", 0, 5000, 2000)
        changed_credit_limit = st.radio("Has Credit Limit Changed Recently?", ["Yes", "No"])
    with col4:
        payment_options = {
            "Low spend, small value payments": "Low_spent_Small_value_payments",
            "High spend, medium value payments": "High_spent_Medium_value_payments",
            "Other": "Other"
        }

        payment_label = st.selectbox(
            "How would you best describe your typical spending and payment habits?",
            list(payment_options.keys())
        )
        payment_behaviour = payment_options[payment_label]
        min_amt_paid = st.radio("Paid Minimum Amount?", ["Yes", "No"])
        credit_mix = st.selectbox("Credit Mix Quality", ["Good", "Standard", "Bad"])

    submitted = st.form_submit_button("🔍 Predict")

    if submitted:
        data = {
            "Age": age,
            "Annual_Income": annual_income,
            "Monthly_Inhand_Salary": salary,
            "Num_Bank_Accounts": num_accounts,
            "Credit_History_Age_Months": credit_years * 12,
            "Num_of_Loan": num_loans,
            "Num_Credit_Card": num_cards,
            "Num_of_Delayed_Payment": num_delays,
            "Delay_from_due_date": delay_days,
            "Num_Credit_Inquiries": num_inquiries,
            "Total_EMI_per_month": total_emi,
            "Outstanding_Debt": outstanding_debt,
            "Interest_Rate": interest_rate,
            "Monthly_Balance": monthly_balance,
            "Changed_Credit_Limit": 1 if changed_credit_limit == "Yes" else 0,
            "Payment_of_Min_Amount_Yes": int(min_amt_paid == "Yes"),
            "Payment_of_Min_Amount_No": int(min_amt_paid == "No"),
            "Payment_Behaviour_Low_spent_Small_value_payments": int(payment_behaviour == "Low_spent_Small_value_payments"),
            "Payment_Behaviour_High_spent_Medium_value_payments": int(payment_behaviour == "High_spent_Medium_value_payments"),
            "Credit_Mix_encoded": {"Good": 2, "Standard": 1, "Bad": 0}[credit_mix],
            "Age_Group_encoded": 1 if age < 30 else (2 if age < 50 else 3),
            "Credit_History_Years_Group_encoded": 1 if credit_years < 5 else (2 if credit_years < 15 else 3),
        }

        # Derived ratios
        data["Debt_to_Income_Ratio"] = round(outstanding_debt / annual_income, 3) if annual_income > 0 else 0
        data["Credit_Utilization_Ratio"] = round(outstanding_debt / (salary * 6), 3) if salary > 0 else 0
        data["EMI_to_Income_Ratio"] = round(total_emi / salary, 3) if salary > 0 else 0
        data["Amount_invested_monthly"] = round(salary * 0.2) if salary > 0 else 0
        data["Investment_to_Income_Ratio"] = round((salary * 0.2) / salary, 2) if salary > 0 else 0
        data["Utilization_Credit_Mix"] = round(data["Credit_Utilization_Ratio"] * data["Credit_Mix_encoded"], 2)
        data["Delayed_Payment_Credit_Mix"] = round(num_delays * data["Credit_Mix_encoded"], 2)

        df_input = pd.DataFrame([data])
        for col in expected_columns:
            if col not in df_input.columns:
                df_input[col] = 0
        df_input = df_input[expected_columns]

        X_scaled = scaler.transform(df_input)

        model_preds = {
            'XGBoost': int(xgb.predict(X_scaled)[0]),
            'LightGBM': int(lgbm.predict(X_scaled)[0]),
            'CatBoost': int(catb.predict(X_scaled)[0]),
            'TabNet': int(tabnet.predict(X_scaled)[0])
        }

        meta_input = np.array([[model_preds[m] for m in model_preds]])
        final_pred = stack.predict(meta_input)[0]
        final_label = le.inverse_transform([final_pred])[0]

        st.subheader("Credit Score Prediction Results")

        # Function to color the final prediction text
        def color_credit_score(label):
            if label == "Poor":
                color = "red"
            elif label == "Good":
                color = "green"
            else:  # Assuming anything else is Standard
                color = "orange"
            return f'<span style="color:{color}; font-weight: bold;">{label}</span>'

        # Use st.markdown instead of st.success for HTML
        st.markdown(f"🎯 Final Predicted Credit Score: {color_credit_score(final_label)}", unsafe_allow_html=True)

        # Expandable Section for Explanation
        with st.expander("About this prediction"):
            st.write(
                """
                This credit score prediction is generated by combining the results of several machine learning models (XGBoost, LightGBM, CatBoost, and TabNet).
                The 'Final Predicted Credit Score' represents the consensus prediction.
                The confidence levels, shown in the chart below, indicate the stacking model's certainty (as a percentage) in assigning the input to each credit score category. Higher values suggest greater certainty.
                """
            )

        st.subheader("🔎 Individual Model Predictions")
        cols = st.columns(len(model_preds))
        for i, (model_name, pred) in enumerate(model_preds.items()):
            label = le.inverse_transform([pred])[0]
            color = "green" if label == "Good" else "red" if label == "Poor" else "orange"
            cols[i].markdown(f"**{model_name}**: <span style='color:{color};'>{label}</span>", unsafe_allow_html=True)

        st.subheader("📊 Confidence Levels from Stacking Model")
        probs = stack.predict_proba(meta_input)[0]
        prob_dict = {le.classes_[i]: probs[i] for i in range(len(probs))}
        # Convert the probability dictionary to a Pandas DataFrame for easier plotting
        probs_df = pd.DataFrame(prob_dict, index=['Confidence']).T.reset_index()
        probs_df.columns = ['Credit Category', 'Confidence']
        probs_df['Confidence (%)'] = probs_df['Confidence'] * 100

        # Define a color mapping for the credit categories
        color_map = {'Poor': 'red', 'Standard': 'orange', 'Good': 'green'}

        # Order the categories for a more intuitive display
        category_order = ['Poor', 'Standard', 'Good']

        # Create the bar chart using Plotly Express
        fig = px.bar(probs_df,
                     x='Credit Category',
                     y='Confidence (%)',
                     color='Credit Category',
                     color_discrete_map=color_map,
                     category_orders={'Credit Category': category_order},
                     text='Confidence (%)',  # Display values on top of bars
                     title='Confidence Levels for Predicted Credit Score')

        # Update layout for better aesthetics
        fig.update_layout(
            yaxis_title='Confidence (%)',
            xaxis_title='Credit Category',
            uniformtext_minsize=8,  # Ensure text labels are readable
            uniformtext_mode='hide',
            yaxis=dict(gridcolor='darkgray'),
            plot_bgcolor='white',
            bargap = 0.5
        )

        # Update traces to position text labels
        fig.update_traces(texttemplate='%{y:.2f}%', textposition='outside')

        # Display the chart in Streamlit
        st.plotly_chart(fig)

        # 📈 Radar Chart (Financial Profile)
        st.subheader("📈 Financial Behavior Profile")
        st.write(
        "This chart visualizes key financial ratios and behaviors that influence creditworthiness. "
        "Hover over each axis point to understand what each metric means."
        )

        # Define financial behavior scores
        feature_scores = {
        'Debt-to-Income': data["Debt_to_Income_Ratio"],
        'Credit Utilization': data["Credit_Utilization_Ratio"],
        'EMI/Income': data["EMI_to_Income_Ratio"],
        'Delayed Payments': data["Num_of_Delayed_Payment"] / 20,  # Normalized
        'Credit Age': data["Credit_History_Age_Months"] / 360     # Normalized
        }

        # Create radar chart dataframe
        radar_df = pd.DataFrame(dict(
            r=list(feature_scores.values()),
            theta=list(feature_scores.keys())
        ))

        # Generate radar chart
        fig_radar = px.line_polar(
            radar_df,
            r='r',
            theta='theta',
            line_close=True,
            title='📊 Financial Behavior'
        )

        # Update chart with descriptive tooltips
        fig_radar.update_traces(
            hovertemplate="<b>%{theta}</b>: %{r}<br>%{customdata}",
            customdata=[
                "Higher values indicate more debt relative to income (potential concern).",
                "Higher values mean you're using more of your available credit (can affect score).",
                "Shows how much of your income goes to EMIs (higher might be riskier).",
                "Normalized count of delayed payments — frequent delays are a red flag.",
                "Normalized age of credit history — older is usually better."
            ]
        )

        # Display chart
        st.plotly_chart(fig_radar)


        # 🎯 Gauge Confidence Meter
        st.subheader("🎯 Confidence Gauge")

        # Get confidence score as percentage
        score_confidence = probs[final_pred] * 100

        # Determine gauge color and confidence message
        if score_confidence > 70:
            gauge_color = "green"
            confidence_text = "High Confidence: The model is highly certain this individual's credit score is 'Standard'."
        elif score_confidence > 40:
            gauge_color = "orange"
            confidence_text = "Moderate Confidence: The model has a reasonable level of certainty that this individual's credit score is 'Standard'."
        else:
            gauge_color = "red"
            confidence_text = "Low Confidence: The model has lower certainty in this 'Standard' credit score prediction. Reviewing the input data might be helpful."

        # Create the gauge chart
        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=score_confidence,
            title={'text': f"Confidence in '{final_label}'"},
            gauge={
                'axis': {'range': [0, 100]},
                'bar': {'color': gauge_color},
                'steps': [
                    {'range': [0, 40], 'color': '#ffcccc'},
                    {'range': [40, 70], 'color': '#fff4cc'},
                    {'range': [70, 100], 'color': '#ccffcc'}
                ]
            }
        ))

        # Display chart and interpretation
        st.plotly_chart(fig_gauge)
        st.markdown(f"**Interpretation:** {confidence_text}")

        # 💡 Recommendations for Improvement
        if final_label != "Good":
            st.subheader("💡 Recommendations to Improve Credit Score")

            recs = []

            # Based on high debt-to-income ratio
            if data["Debt_to_Income_Ratio"] > 0.4:
                recs.append("• Try reducing your overall debt. A high debt-to-income ratio can negatively impact your creditworthiness.")

            # Based on credit utilization
            if data["Credit_Utilization_Ratio"] > 0.3:
                recs.append("• Lower your credit utilization. Keeping it below 30% is ideal for a better credit score.")

            # Based on delayed payments
            if data["Num_of_Delayed_Payment"] >= 3:
                recs.append("• Reduce delayed payments. Regular on-time payments build a positive credit history.")

            # Based on EMI to income ratio
            if data["EMI_to_Income_Ratio"] > 0.5:
                recs.append("• Your EMI commitments are high. Try consolidating or refinancing to lower the monthly burden.")

            # Based on short credit history
            if data["Credit_History_Age_Months"] < 24:
                recs.append("• Build a longer credit history. Maintaining older accounts can help over time.")

            # Based on credit mix
            if data["Credit_Mix_encoded"] < 2:
                recs.append("• Diversify your credit mix. A healthy mix of loans and credit cards is seen favorably.")

            if recs:
                for r in recs:
                    st.markdown(r)
            else:
                st.markdown("No major red flags were found, but consistently making timely payments and managing your credit wisely will help.")



        # Top Feature Importance ---
        st.subheader("📌 Top Features (XGBoost)")
        st.write("This chart shows the top 10 features that the XGBoost model deemed most influential in predicting the credit score. Higher bars indicate greater importance.")
        feat_imp = pd.DataFrame({'Feature': X_train.columns, 'Importance': xgb.feature_importances_})
        feat_imp = feat_imp.sort_values("Importance", ascending=False).head(10)
        st.bar_chart(feat_imp.set_index("Feature"))

        with st.expander("📋 Show Input Feature Summary"):
            summary_df = df_input.T
            summary_df.columns = ["User Input Value"] 
            summary_df.index.name = "Features"
            st.dataframe(summary_df)


st.markdown("<div class='footer'>© 2025 Credit AI System | Built By Hewa</div>", unsafe_allow_html=True)