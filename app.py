import streamlit as st
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix

import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# -----------------------------
# 1. Page config
# -----------------------------
st.set_page_config(
    page_title="Smart IT Operations – Predictive Risk Dashboard",
    layout="wide"
)

st.title("Smart IT Operations – AI Predictive Risk Dashboard (PoC)")
st.markdown(
    """
    This dashboard simulates an Aramco-style **LAN / Backbone / Facility** environment using synthetic data.
    The AI model predicts the risk of failure in the next 7 days and prioritizes devices based on their risk.
    """
)

# -----------------------------
# 2. Load data (cached)
# -----------------------------
@st.cache_data
def load_data():
    # Make sure smart_ops_synthetic.csv is in the same folder as app.py in GitHub
    df = pd.read_csv("smart_ops_synthetic.csv", parse_dates=["timestamp"])
    return df

df = load_data()

# Sidebar filters
st.sidebar.header("Filters")

group_filter = st.sidebar.multiselect(
    "Select group(s)",
    options=sorted(df["group"].unique()),
    default=sorted(df["group"].unique())
)

site_filter = st.sidebar.multiselect(
    "Select site(s)",
    options=sorted(df["site"].unique()),
    default=sorted(df["site"].unique())
)

criticality_filter = st.sidebar.multiselect(
    "Select criticality",
    options=sorted(df["criticality"].unique()),
    default=sorted(df["criticality"].unique())
)

test_size = st.sidebar.slider(
    "Test size (fraction used for testing)",
    0.1, 0.4, 0.2, 0.05
)

n_estimators = st.sidebar.slider(
    "RandomForest trees (n_estimators)",
    50, 400, 200, 50
)

random_state = 42

# Apply filters
df_filtered = df[
    df["group"].isin(group_filter)
    & df["site"].isin(site_filter)
    & df["criticality"].isin(criticality_filter)
].copy()

st.markdown(f"**Filtered records:** {len(df_filtered):,}")

if len(df_filtered) < 500:
    st.warning("Filtered dataset is quite small. Consider selecting more groups/sites/criticality levels for a stronger model.")

# -----------------------------
# 3. Preprocess & train model
# -----------------------------
@st.cache_resource
def train_model(df_in, test_size, n_estimators, random_state):
    df_model = df_in.copy()

    # Target
    y = df_model["failure_next_7d"].astype(int)

    # Features to use
    feature_cols = [
        "cpu_util",
        "interface_errors",
        "optical_power_dbm",
        "battery_voltage",
        "load_percent",
        "temperature_c",
        "group",
        "device_type",
        "site",
        "criticality",
    ]

    X = df_model[feature_cols].copy()

    # Handle missing numeric values
    num_cols = ["cpu_util", "interface_errors", "optical_power_dbm",
                "battery_voltage", "load_percent", "temperature_c"]
    for col in num_cols:
        X[col] = X[col].fillna(X[col].median())

    # One-hot encode categorical features
    X = pd.get_dummies(
        X,
        columns=["group", "device_type", "site", "criticality"],
        drop_first=True
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    # Scale numeric columns
    scaler = StandardScaler()
    X_train[num_cols] = scaler.fit_transform(X_train[num_cols])
    X_test[num_cols] = scaler.transform(X_test[num_cols])

    model = RandomForestClassifier(
        n_estimators=n_estimators,
        random_state=random_state,
        class_weight="balanced",
        n_jobs=-1
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0

    metrics = {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "n_train": len(X_train),
        "n_test": len(X_test)
    }

    # Build risk table on test set
    # Use .loc to align with original df indices
    risk_table = df_in.loc[y_test.index].copy()
    risk_table["Failure_Probability"] = y_proba
    risk_table["Predicted_Failure"] = y_pred
    risk_table["Risk_Score"] = (risk_table["Failure_Probability"] * 100).round(1)

    def risk_band(score):
        if score >= 80:
            return "Critical"
        elif score >= 60:
            return "High"
        elif score >= 30:
            return "Medium"
        else:
            return "Low"

    risk_table["Risk_Level"] = risk_table["Risk_Score"].apply(risk_band)

    # Feature importance
    importance_df = pd.DataFrame({
        "Feature": X_train.columns,
        "Importance": model.feature_importances_
    }).sort_values(by="Importance", ascending=False)

    return model, risk_table, importance_df, metrics

model, risk_table, importance_df, metrics = train_model(
    df_filtered, test_size, n_estimators, random_state
)

# -----------------------------
# 4. Top-level KPIs for management
# -----------------------------
total_devices = df_filtered["device_id"].nunique()
high_risk_devices = risk_table[risk_table["Risk_Level"].isin(["Critical", "High"])]["device_id"].nunique()
overall_failure_rate = df_filtered["failure_next_7d"].mean() * 100

col1, col2, col3, col4 = st.columns(4)
col1.metric("Devices in scope", f"{total_devices:,}")
col2.metric("High/Critical risk devices", f"{high_risk_devices:,}")
col3.metric("Historical failure rate", f"{overall_failure_rate:.1f}%")
col4.metric("Model recall (failures)", f"{metrics['recall']*100:.1f}%")

st.markdown("---")

# -----------------------------
# 5. One-page dashboard layout
# -----------------------------
risk_counts = risk_table["Risk_Level"].value_counts().reset_index()
risk_counts.columns = ["Risk_Level", "Count"]

top_imp = importance_df.head(15).sort_values(by="Importance", ascending=True)

top10 = (
    risk_table
    .sort_values(by="Risk_Score", ascending=False)
    .head(10)
    .reset_index(drop=True)
)
top10.index = top10.index + 1

fig = make_subplots(
    rows=2, cols=2,
    specs=[[{"type": "domain"}, {"type": "xy"}],
           [{"type": "xy"}, {"type": "table"}]],
    subplot_titles=(
        "Risk Levels (AI-assessed)",
        "Risk Score Distribution",
        "Top Drivers of Failure (Feature Importance)",
        "Top 10 Highest-Risk Devices (Next 7 Days)"
    )
)

# Pie – risk levels
fig.add_trace(
    go.Pie(
        labels=risk_counts["Risk_Level"],
        values=risk_counts["Count"],
        hole=0.4
    ),
    row=1, col=1
)

# Histogram – risk scores
fig.add_trace(
    go.Histogram(
        x=risk_table["Risk_Score"],
        nbinsx=30
    ),
    row=1, col=2
)

# Feature importance bar
fig.add_trace(
    go.Bar(
        x=top_imp["Importance"],
        y=top_imp["Feature"],
        orientation="h"
    ),
    row=2, col=1
)

# Table – top 10 risky devices
fig.add_trace(
    go.Table(
        header=dict(
            values=["#", "Device", "Group", "Site", "Risk Level", "Risk Score", "Fail Prob.", "Actual Fail?"],
            fill_color="lightgrey",
            align="center"
        ),
        cells=dict(
            values=[
                top10.index,
                top10["device_id"],
                top10["group"],
                top10["site"],
                top10["Risk_Level"],
                top10["Risk_Score"],
                top10["Failure_Probability"].round(3),
                top10["failure_next_7d"],
            ],
            align="center"
        )
    ),
    row=2, col=2
)

fig.update_layout(
    height=900,
    showlegend=False
)

fig.update_xaxes(title_text="Risk Score (0–100)", row=1, col=2)
fig.update_yaxes(title_text="Count", row=1, col=2)
fig.update_xaxes(title_text="Importance", row=2, col=1)
fig.update_yaxes(title_text="Feature", row=2, col=1)

st.plotly_chart(fig, use_container_width=True)

# -----------------------------
# 6. Detailed views
# -----------------------------
st.markdown("### Detailed Risk Table (filtered scope)")
st.dataframe(
    risk_table.sort_values(by="Risk_Score", ascending=False).reset_index(drop=True),
    use_container_width=True
)

st.markdown("### Raw Data Sample")
st.dataframe(
    df_filtered.head(100),
    use_container_width=True
)
