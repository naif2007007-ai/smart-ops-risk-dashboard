import streamlit as st
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix

import plotly.graph_objects as go
from plotly.subplots import make_subplots

# -----------------------------
# 1. Page config & basic style
# -----------------------------
st.set_page_config(
    page_title="AI Early Warning – Smart IT Operations",
    layout="wide"
)

# Simple CSS for cards + blinking alert
st.markdown(
    """
    <style>
    .kpi-card {
        padding: 1rem 1.5rem;
        border-radius: 0.7rem;
        margin-bottom: 0.7rem;
        background-color: #1f2933;
        border: 1px solid #374151;
    }
    .kpi-title {
        font-size: 0.9rem;
        color: #9CA3AF;
        margin-bottom: 0.1rem;
    }
    .kpi-value {
        font-size: 1.6rem;
        font-weight: 700;
        color: #F9FAFB;
    }
    .kpi-sub {
        font-size: 0.8rem;
        color: #9CA3AF;
    }
    .status-ok {
        border-left: 6px solid #10B981;
    }
    .status-warn {
        border-left: 6px solid #F59E0B;
    }
    .status-crit {
        border-left: 6px solid #EF4444;
    }
    @keyframes blinker {
        50% { opacity: 0.25; }
    }
    .blink {
        animation: blinker 1s linear infinite;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# -----------------------------
# 2. Title & description
# -----------------------------
st.title("AI Early Warning – Smart IT Operations Dashboard")

st.markdown(
    """
    **Purpose:** This dashboard is **not** a live monitoring tool like Netcool.  
    It is an **AI engine** that analyzes historical behavior of LAN, Backbone, and Facility devices to  
    **predict the risk of failure in the next 7 days** and **prioritize devices and sites** based on that risk.

    The current Proof of Concept uses synthetic data but the design is scalable to real Aramco data sources (Netcool, Remedy, facility monitoring, etc.).
    """
)

# -----------------------------
# 3. Load data
# -----------------------------
@st.cache_data
def load_data():
    df = pd.read_csv("smart_ops_synthetic.csv", parse_dates=["timestamp"])
    return df

df = load_data()

# -----------------------------
# 4. Sidebar filters
# -----------------------------
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
    "RandomForest trees (model complexity)",
    50, 400, 200, 50
)

random_state = 42

df_filtered = df[
    df["group"].isin(group_filter)
    & df["site"].isin(site_filter)
    & df["criticality"].isin(criticality_filter)
].copy()

st.markdown(f"**Filtered records in AI model scope:** {len(df_filtered):,}")

if len(df_filtered) < 500:
    st.warning("Filtered dataset is quite small. Consider including more groups/sites/criticality levels for a stronger AI model.")

# -----------------------------
# 5. Train model & build risk table
# -----------------------------
@st.cache_resource
def train_model(df_in, test_size, n_estimators, random_state):
    df_model = df_in.copy()
    y = df_model["failure_next_7d"].astype(int)

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

    num_cols = ["cpu_util", "interface_errors", "optical_power_dbm",
                "battery_voltage", "load_percent", "temperature_c"]
    for col in num_cols:
        X[col] = X[col].fillna(X[col].median())

    X = pd.get_dummies(
        X,
        columns=["group", "device_type", "site", "criticality"],
        drop_first=True
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

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

    importance_df = pd.DataFrame({
        "Feature": X_train.columns,
        "Importance": model.feature_importances_
    }).sort_values(by="Importance", ascending=False)

    return model, risk_table, importance_df, metrics

model, risk_table, importance_df, metrics = train_model(
    df_filtered, test_size, n_estimators, random_state
)

# -----------------------------
# 6. AI Early Warning Panel
# -----------------------------
total_devices = df_filtered["device_id"].nunique()
critical_df = risk_table[risk_table["Risk_Level"] == "Critical"]
high_df = risk_table[risk_table["Risk_Level"] == "High"]
medium_df = risk_table[risk_table["Risk_Level"] == "Medium"]

crit_devices = critical_df["device_id"].nunique()
high_devices = high_df["device_id"].nunique()
med_devices = medium_df["device_id"].nunique()

predicted_failures = int((risk_table["Predicted_Failure"] == 1).sum())
sites_at_risk = risk_table[risk_table["Risk_Level"].isin(["Critical", "High"])]["site"].nunique()
overall_failure_rate = df_filtered["failure_next_7d"].mean() * 100

col_alert, col_empty = st.columns([2, 3])

with col_alert:
    if crit_devices > 0:
        st.markdown(
            f"""
            <div class="kpi-card status-crit blink">
                <div class="kpi-title">AI Early Warning</div>
                <div class="kpi-value">⚠ {crit_devices} CRITICAL device(s)</div>
                <div class="kpi-sub">
                    Predicted to have a high probability of failure in the next 7 days.  
                    Recommended action: review with LAN / Backbone / Facility teams.
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )
    else:
        st.markdown(
            """
            <div class="kpi-card status-ok">
                <div class="kpi-title">AI Early Warning</div>
                <div class="kpi-value">✅ No Critical AI risks detected</div>
                <div class="kpi-sub">
                    One or more devices may still be at High/Medium risk – see details below.
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

# KPIs row
k1, k2, k3, k4 = st.columns(4)

with k1:
    st.markdown(
        f"""
        <div class="kpi-card status-ok">
            <div class="kpi-title">Devices in AI scope</div>
            <div class="kpi-value">{total_devices:,}</div>
            <div class="kpi-sub">Based on selected groups/sites/criticality</div>
        </div>
        """,
        unsafe_allow_html=True
    )

with k2:
    st.markdown(
        f"""
        <div class="kpi-card status-warn">
            <div class="kpi-title">Predicted failures (next 7 days)</div>
            <div class="kpi-value">{predicted_failures}</div>
            <div class="kpi-sub">Devices where AI predicts a failure event</div>
        </div>
        """,
        unsafe_allow_html=True
    )

with k3:
    high_crit_pct = 0
    if total_devices > 0:
        high_crit_pct = (crit_devices + high_devices) / total_devices * 100
    st.markdown(
        f"""
        <div class="kpi-card status-crit">
            <div class="kpi-title">Devices at High/Critical AI risk</div>
            <div class="kpi-value">{high_crit_pct:.1f}%</div>
            <div class="kpi-sub">{crit_devices} Critical, {high_devices} High</div>
        </div>
        """,
        unsafe_allow_html=True
    )

with k4:
    st.markdown(
        f"""
        <div class="kpi-card status-ok">
            <div class="kpi-title">Model recall (failure detection)</div>
            <div class="kpi-value">{metrics['recall']*100:.1f}%</div>
            <div class="kpi-sub">Share of historical failures that the AI model captures</div>
        </div>
        """,
        unsafe_allow_html=True
    )

st.markdown("---")

# -----------------------------
# 7. Risk distribution & feature importance
# -----------------------------
risk_counts = risk_table["Risk_Level"].value_counts().reset_index()
risk_counts.columns = ["Risk_Level", "Count"]

top_imp = importance_df.head(12).sort_values(by="Importance", ascending=True)

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
        "AI Risk Levels (Critical / High / Medium / Low)",
        "Distribution of AI Risk Scores",
        "Top Drivers of AI Failure Risk",
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
            values=["#", "Device", "Group", "Site", "Risk Level", "Risk Score", "Fail Prob.", "Actual Failure?"],
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
# 8. Expandable detailed sections
# -----------------------------
with st.expander("🔍 Detailed AI risk table (for engineers)"):
    st.dataframe(
        risk_table.sort_values(by="Risk_Score", ascending=False).reset_index(drop=True),
        use_container_width=True
    )

with st.expander("📊 Raw data sample (synthetic PoC data)"):
    st.dataframe(
        df_filtered.head(200),
        use_container_width=True
    )
