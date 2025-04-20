import plotly.graph_objects as go
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# Load the cleaned dataset
df = pd.read_csv('final_cleaned_credit_score_data_v2.csv')

# --- Step 3: Comprehensive Feature Engineering ---

# 3.1 Create Financial Ratios
print("\n--- 3.1 Creating Financial Ratios ---")
# Debt-to-Income Ratio: Outstanding_Debt / Annual_Income
df['Debt_to_Income_Ratio'] = df['Outstanding_Debt'] / df['Annual_Income'].replace(0, 1e-6)  # Avoid division by 0

# EMI-to-Income Ratio: Total_EMI_per_month / Monthly_Inhand_Salary
df['EMI_to_Income_Ratio'] = df['Total_EMI_per_month'] / df['Monthly_Inhand_Salary'].replace(0, 1e-6)

# Investment-to-Income Ratio: Amount_invested_monthly / Monthly_Inhand_Salary
df['Investment_to_Income_Ratio'] = df['Amount_invested_monthly'] / df['Monthly_Inhand_Salary'].replace(0, 1e-6)

# Analysis: Check distributions and summaries of financial ratios
for col in ['Debt_to_Income_Ratio', 'EMI_to_Income_Ratio', 'Investment_to_Income_Ratio']:
    print(f"\nSummary of {col}:\n", df[col].describe())
    plt.figure(figsize=(6, 4))
    sns.histplot(df[col].clip(upper=df[col].quantile(0.99)), kde=True)  # Clip extreme values for visualization
    plt.title(f"Distribution of {col}")
    plt.show()

# 3.2 Aggregate Features
print("\n--- 3.2 Creating Aggregate Features ---")
# Total Credit Accounts: Num_Bank_Accounts + Num_Credit_Card
df['Total_Credit_Accounts'] = df['Num_Bank_Accounts'] + df['Num_Credit_Card']

# Total Credit Load: Num_of_Loan + Num_Credit_Card
df['Total_Credit_Load'] = df['Num_of_Loan'] + df['Num_Credit_Card']

# Average Credit Inquiries per Year: Num_Credit_Inquiries / (Credit_History_Age_Months / 12)
df['Inquiries_per_Year'] = df['Num_Credit_Inquiries'] / (df['Credit_History_Age_Months'] / 12).replace(0, 1e-6)

# Analysis: Check distributions and summaries of aggregate features
for col in ['Total_Credit_Accounts', 'Total_Credit_Load', 'Inquiries_per_Year']:
    print(f"\nSummary of {col}:\n", df[col].describe())
    plt.figure(figsize=(6, 4))
    sns.histplot(df[col].clip(upper=df[col].quantile(0.99)), kde=True)
    plt.title(f"Distribution of {col}")
    plt.show()

# 3.3 Binning Continuous Features
print("\n--- 3.3 Binning Continuous Features ---")
# Age Bins
df['Age_Group'] = pd.cut(df['Age'], bins=[0, 25, 35, 45, 60, 100], labels=['0-25', '26-35', '36-45', '46-60', '60+'])
# Credit History Age Bins (in years)
df['Credit_History_Years_Group'] = pd.cut(df['Credit_History_Age_Months'] / 12, bins=[0, 5, 10, 15, 20, 35],
                                        labels=['0-5', '6-10', '11-15', '16-20', '20+'])
# Debt-to-Income Ratio Bins
df['Debt_to_Income_Group'] = pd.cut(df['Debt_to_Income_Ratio'], bins=[0, 0.2, 0.4, 0.6, 1.0, float('inf')],
                                    labels=['Low', 'Moderate', 'High', 'Very High', 'Extreme'])

# Analysis: Check distributions of binned features
for col in ['Age_Group', 'Credit_History_Years_Group', 'Debt_to_Income_Group']:
    print(f"\nDistribution of {col}:\n", df[col].value_counts())
    plt.figure(figsize=(6, 4))
    sns.countplot(x=col, data=df)
    plt.title(f"Distribution of {col}")
    plt.xticks(rotation=45)
    plt.show()

# 3.4 Interaction Features
print("\n--- 3.4 Creating Interaction Features ---")
# Interaction between Num_of_Delayed_Payment and Credit_Mix
df['Delayed_Payment_Credit_Mix'] = df['Num_of_Delayed_Payment'] * df['Credit_Mix'].map(
    {'Bad': 1, 'Standard': 2, 'Good': 3})
# Interaction between Interest_Rate and Num_of_Loan (to capture combined effect of loan burden)
df['Interest_Rate_Loan_Interaction'] = df['Interest_Rate'] * df['Num_of_Loan']
# Interaction between Credit_Utilization_Ratio and Credit_Mix
df['Utilization_Credit_Mix'] = df['Credit_Utilization_Ratio'] * df['Credit_Mix'].map(
    {'Bad': 1, 'Standard': 2, 'Good': 3})

# Analysis: Check distributions and summaries of interaction features
for col in ['Delayed_Payment_Credit_Mix', 'Interest_Rate_Loan_Interaction', 'Utilization_Credit_Mix']:
    print(f"\nSummary of {col}:\n", df[col].describe())
    plt.figure(figsize=(6, 4))
    sns.histplot(df[col].clip(upper=df[col].quantile(0.99)), kde=True)
    plt.title(f"Distribution of {col}")
    plt.show()


# 3.5  Function to Create Improved Radar Chart
import plotly.graph_objects as go
import pandas as pd  # If you're using DataFrames
import numpy as np

def create_financial_radar_chart(data,  # Expecting a dictionary or DataFrame
                                 data_labels=None, # List of labels if data is a list/dict
                                 title="Financial Behavior Profile",
                                 interpretations=None, # Dict for interpretations
                                 benchmark_data=None, # Optional benchmark data
                                 benchmark_label="Average 'Good' Score"):
    """
    Generates a financial radar chart with enhancements.

    Args:
        data (dict or DataFrame):  Financial data. If dict, keys are labels.
        data_labels (list, optional): List of labels if data is a list or dict.
        title (str, optional): Title of the chart.
        interpretations (dict, optional):  Guidance on metric interpretation.
        benchmark_data (list, optional): Data for a benchmark comparison.
        benchmark_label (str, optional): Label for the benchmark line.

    Returns:
        plotly.graph_objects.Figure: The radar chart figure.
    """

    if isinstance(data, dict):
        labels = list(data.keys())
        values = list(data.values())
    elif isinstance(data, pd.DataFrame):
        # Assuming data is a DataFrame, adjust as needed
        labels = data.columns.tolist()
        values = data.iloc[0].tolist()  # Or however you access the row
    else:
        labels = data_labels
        values = data

    fig = go.Figure()

    # Individual's profile
    fig.add_trace(go.Scatterpolar(
          r=values,
          theta=labels,
          fill='toself',
          name="Individual"
    ))

    # Add benchmark if provided
    if benchmark_data:
        fig.add_trace(go.Scatterpolar(
          r=benchmark_data,
          theta=labels,
          fill='toself',
          name=benchmark_label,
          opacity=0.5 # Make benchmark fainter
        ))

    fig.update_layout(
      title=title,
      polar=dict(
        radialaxis=dict(
          visible=True,
          range=[0, 1]  # Assuming normalized data
        ),
      ),
      showlegend=True  # Ensure legend is shown
    )

    # Add annotations for interpretations (positioning might need tweaking)
    if interpretations:
        annotations = []
        for i, label in enumerate(labels):
            angle = 360 * (i / len(labels))
            x_pos = 1.3 * np.cos(np.radians(angle))
            y_pos = 1.3 * np.sin(np.radians(angle))

            annotations.append(dict(
                x=x_pos,
                y=y_pos,
                text=f"<b>{label}</b>: {interpretations[label]}",
                showarrow=False,
                xref="paper", yref="paper",
                xanchor="center", yanchor="middle"
            ))
        fig.update_layout(annotations=annotations)

    return fig


# --- Example Usage of Radar Chart Function ---

# Sample data (replace with your actual data)
individual_data_radar = {
    "Debt-to-Income": 0.7,
    "Credit Utilization": 0.4,
    "EMI/Income": 0.5,
    "Delayed Payments": 0.2,
    "Credit Age": 0.8
}

interpretations_radar = {
    "Debt-to-Income": "Lower is better.",
    "Credit Utilization": "Below 0.5 is good.",
    "EMI/Income": "Keep this low.",
    "Delayed Payments": "Fewer delays are preferred.",
    "Credit Age": "Longer history is favorable."
}

benchmark_data_radar = [0.5, 0.6, 0.4, 0.3, 0.7]  # Example benchmark

fig_radar = create_financial_radar_chart(individual_data_radar,
                                  title="Financial Behavior Profile",
                                  interpretations=interpretations_radar,
                                  benchmark_data=benchmark_data_radar)

fig_radar.show()