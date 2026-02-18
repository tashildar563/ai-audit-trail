import streamlit as st
import pandas as pd
import numpy as np
import sqlite3
import json
import pickle
import os
import sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from datetime import datetime
from xgboost import XGBClassifier

# ── Path Setup ─────────────────────────────────────────────
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.predict import predict_and_log
from src.explainer import explain_prediction, global_shap_summary
from src.drift_monitor import run_drift_check, load_reference, load_recent_predictions

# ── Page Config ────────────────────────────────────────────
st.set_page_config(
    page_title="AI Decision Audit Trail",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Custom CSS ─────────────────────────────────────────────
st.markdown("""
<style>
    .main { background-color: #0e1117; }
    .metric-card {
        background: linear-gradient(135deg, #1e2130, #2d3250);
        border-radius: 12px;
        padding: 20px;
        text-align: center;
        border: 1px solid #3d4466;
    }
    .metric-value {
        font-size: 2.2rem;
        font-weight: 700;
        color: #7eb3ff;
    }
    .metric-label {
        font-size: 0.85rem;
        color: #8892b0;
        margin-top: 4px;
    }
    .alert-high {
        background: linear-gradient(135deg, #3d1515, #5c1f1f);
        border-left: 4px solid #ff4b4b;
        padding: 12px 16px;
        border-radius: 8px;
        color: #ffcccc;
    }
    .alert-ok {
        background: linear-gradient(135deg, #0d2b1e, #0f3d2a);
        border-left: 4px solid #00cc88;
        padding: 12px 16px;
        border-radius: 8px;
        color: #ccffe8;
    }
    .stButton > button {
        background: linear-gradient(135deg, #4a6fa5, #6b8fc9);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 10px 24px;
        font-weight: 600;
        width: 100%;
    }
    .stButton > button:hover {
        background: linear-gradient(135deg, #5a82be, #7fa3e0);
        transform: translateY(-1px);
    }
</style>
""", unsafe_allow_html=True)

# ── Helper: Load Audit Log ─────────────────────────────────
DB_PATH = "logs/audit_log.db"

def load_audit_log():
    if not os.path.exists(DB_PATH):
        return pd.DataFrame()
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql("SELECT * FROM audit_log ORDER BY id DESC", conn)
    conn.close()
    return df

def load_alerts():
    path = "logs/drift_alerts.json"
    if not os.path.exists(path):
        return []
    try:
        with open(path) as f:
            content = f.read().strip()
            if not content:
                return []
            return json.loads(content)
    except (json.JSONDecodeError, ValueError):
        # File corrupted — reset it cleanly
        with open(path, "w") as f:
            json.dump([], f)
        return []

# ══════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("## 🏥 AI Audit Trail")
    st.markdown("*Medical Diagnosis Monitor*")
    st.divider()

    page = st.radio(
    "Navigate",
    ["🏠 Home", "🔮 New Prediction", "📋 Audit Log",
     "📉 Drift Monitor", "📊 Model Report", "🏆 Model Comparison", 
     "⚖️ Fairness & Bias", "🏭 Production Monitor"],
        label_visibility="collapsed"
    )

    st.divider()
    df_log = load_audit_log()
    total  = len(df_log)
    if total > 0:
        positive = int(df_log["prediction"].sum())
        st.markdown(f"**Total Predictions:** {total}")
        st.markdown(f"**Diabetes Detected:** {positive}")
        st.markdown(f"**Clear Cases:** {total - positive}")
    else:
        st.markdown("*No predictions yet*")

    st.divider()
    st.markdown(
        "<small>Built with Python · XGBoost · SHAP · Streamlit</small>",
        unsafe_allow_html=True
    )

# ══════════════════════════════════════════════════════════
# PAGE 1 — HOME
# ══════════════════════════════════════════════════════════
if page == "🏠 Home":
    st.title("🏥 AI Decision Audit Trail System")
    st.markdown("##### Transparent · Explainable · Monitored Medical AI")
    st.divider()

    df_log = load_audit_log()
    total  = len(df_log)

    # ── Metric Cards ───────────────────────────────────────
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.markdown(f"""
        <div class='metric-card'>
            <div class='metric-value'>{total}</div>
            <div class='metric-label'>Total Predictions</div>
        </div>""", unsafe_allow_html=True)

    with col2:
        positive = int(df_log["prediction"].sum()) if total > 0 else 0
        st.markdown(f"""
        <div class='metric-card'>
            <div class='metric-value'>{positive}</div>
            <div class='metric-label'>Diabetes Detected</div>
        </div>""", unsafe_allow_html=True)

    with col3:
        avg_conf = f"{df_log['confidence'].mean():.1%}" if total > 0 else "—"
        st.markdown(f"""
        <div class='metric-card'>
            <div class='metric-value'>{avg_conf}</div>
            <div class='metric-label'>Avg Confidence</div>
        </div>""", unsafe_allow_html=True)

    with col4:
        alerts   = load_alerts()
        severity = alerts[-1]["severity"] if alerts else "OK"
        color    = "#ff4b4b" if severity == "HIGH" else "#ffaa00" if severity == "MEDIUM" else "#00cc88"
        st.markdown(f"""
        <div class='metric-card'>
            <div class='metric-value' style='color:{color}'>{severity}</div>
            <div class='metric-label'>Drift Status</div>
        </div>""", unsafe_allow_html=True)

    st.divider()

    if total == 0:
        st.info("📊 No predictions yet. Go to **New Prediction** to get started!")
        st.stop()

    # ══════════════════════════════════════════════════════════
    # ANALYTICS SECTION — NEW INTERACTIVE CHARTS
    # ══════════════════════════════════════════════════════════
    
    import plotly.express as px
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    # ── Chart 1: Prediction Timeline ───────────────────────
    st.subheader("📈 Prediction Timeline")
    df_log["datetime"] = pd.to_datetime(df_log["timestamp"])
    df_log["date"] = df_log["datetime"].dt.date
    
    timeline = df_log.groupby(["date", "prediction_label"]).size().reset_index(name="count")
    
    fig_timeline = px.bar(
        timeline,
        x="date",
        y="count",
        color="prediction_label",
        title="Predictions Over Time",
        color_discrete_map={"Diabetes": "#ff4b4b", "No Diabetes": "#00cc88"},
        labels={"date": "Date", "count": "Number of Predictions", "prediction_label": "Diagnosis"},
        template="plotly_dark"
    )
    fig_timeline.update_layout(
        plot_bgcolor="#0e1117",
        paper_bgcolor="#0e1117",
        font=dict(color="white"),
        height=400
    )
    st.plotly_chart(fig_timeline, use_container_width=True)

    st.divider()

    # ── Chart 2 & 3: Side by Side ──────────────────────────
    col_left, col_right = st.columns(2)

    with col_left:
        st.subheader("🎯 Confidence Distribution")
        
        fig_conf = go.Figure()
        fig_conf.add_trace(go.Histogram(
            x=df_log["confidence"],
            nbinsx=20,
            marker_color="#7eb3ff",
            name="All Predictions"
        ))
        fig_conf.update_layout(
            title="Model Confidence Distribution",
            xaxis_title="Confidence Score",
            yaxis_title="Frequency",
            template="plotly_dark",
            plot_bgcolor="#0e1117",
            paper_bgcolor="#0e1117",
            font=dict(color="white"),
            height=400,
            showlegend=False
        )
        st.plotly_chart(fig_conf, use_container_width=True)

    with col_right:
        st.subheader("🍩 Diagnosis Split")
        
        counts = df_log["prediction_label"].value_counts()
        fig_pie = go.Figure(data=[go.Pie(
            labels=counts.index,
            values=counts.values,
            marker=dict(colors=["#00cc88", "#ff4b4b"]),
            hole=0.4
        )])
        fig_pie.update_layout(
            title="Overall Diagnosis Distribution",
            template="plotly_dark",
            plot_bgcolor="#0e1117",
            paper_bgcolor="#0e1117",
            font=dict(color="white"),
            height=400
        )
        st.plotly_chart(fig_pie, use_container_width=True)

    st.divider()

    # ── Chart 4: Risk by Age Group ─────────────────────────
    st.subheader("👥 Risk Analysis by Age Group")
    
    df_log["age_group"] = pd.cut(
        df_log["age"],
        bins=[0, 25, 35, 45, 55, 100],
        labels=["18-25", "26-35", "36-45", "46-55", "55+"]
    )
    
    age_risk = df_log.groupby(["age_group", "prediction_label"]).size().unstack(fill_value=0)
    age_risk["Total"] = age_risk.sum(axis=1)
    age_risk["Risk %"] = (age_risk.get("Diabetes", 0) / age_risk["Total"] * 100).round(1)
    
    fig_age = go.Figure()
    
    if "No Diabetes" in age_risk.columns:
        fig_age.add_trace(go.Bar(
            name="No Diabetes",
            x=age_risk.index.astype(str),
            y=age_risk["No Diabetes"],
            marker_color="#00cc88"
        ))
    
    if "Diabetes" in age_risk.columns:
        fig_age.add_trace(go.Bar(
            name="Diabetes",
            x=age_risk.index.astype(str),
            y=age_risk["Diabetes"],
            marker_color="#ff4b4b"
        ))
    
    fig_age.update_layout(
        title="Diabetes Cases by Age Group",
        xaxis_title="Age Group",
        yaxis_title="Number of Cases",
        barmode="stack",
        template="plotly_dark",
        plot_bgcolor="#0e1117",
        paper_bgcolor="#0e1117",
        font=dict(color="white"),
        height=400
    )
    st.plotly_chart(fig_age, use_container_width=True)

    # Risk percentage table below chart
    st.caption("📊 Risk Statistics by Age Group")
    risk_table = age_risk[["Total", "Risk %"]].reset_index()
    risk_table.columns = ["Age Group", "Total Patients", "Diabetes Risk %"]
    st.dataframe(risk_table, use_container_width=True, hide_index=True)

    st.divider()

    # ── Chart 5: Feature Correlation Heatmap ───────────────
    st.subheader("🔥 Feature Correlation Heatmap")
    st.caption("How strongly each feature correlates with diabetes diagnosis")
    
    feature_cols = ["pregnancies", "glucose", "blood_pressure", "skin_thickness",
                    "insulin", "bmi", "diabetes_pedigree", "age", "prediction"]
    
    corr_data = df_log[feature_cols].corr()
    
    fig_heatmap = go.Figure(data=go.Heatmap(
        z=corr_data.values,
        x=corr_data.columns,
        y=corr_data.columns,
        colorscale="RdBu_r",
        zmid=0,
        text=corr_data.values.round(2),
        texttemplate="%{text}",
        textfont={"size": 10},
        colorbar=dict(title="Correlation")
    ))
    
    fig_heatmap.update_layout(
        title="Feature Correlation Matrix",
        template="plotly_dark",
        plot_bgcolor="#0e1117",
        paper_bgcolor="#0e1117",
        font=dict(color="white"),
        height=500,
        xaxis=dict(tickangle=-45)
    )
    st.plotly_chart(fig_heatmap, use_container_width=True)

    # Correlation insights
    diabetes_corr = corr_data["prediction"].drop("prediction").sort_values(ascending=False)
    st.caption("🔑 Top 3 Features Most Correlated with Diabetes:")
    for i, (feat, val) in enumerate(diabetes_corr.head(3).items(), 1):
        st.markdown(f"**{i}.** {feat.replace('_', ' ').title()} — Correlation: {val:+.3f}")

    st.divider()

    # ── Recent Activity Table (kept from original) ────────
    st.subheader("📋 Recent Predictions")
    display_df = df_log[["timestamp", "patient_id",
                          "prediction_label", "confidence"]].head(10).copy()
    display_df["confidence"] = display_df["confidence"].apply(lambda x: f"{x:.1%}")
    display_df.columns = ["Timestamp", "Patient ID", "Diagnosis", "Confidence"]
    st.dataframe(display_df, use_container_width=True, hide_index=True)

    st.divider()

    # ── Pipeline Overview (kept from original) ────────────
    st.subheader("🏗️ System Architecture")
    st.markdown("""
```
    Patient Data → XGBoost Model → Prediction Engine
                                         ↓
                                  Audit Logger (SQLite)
                                         ↓
                         ┌───────────────┴───────────────┐
                    SHAP Explainer               Drift Monitor
                  (Why this result?)        (Is data changing?)
                         └───────────────┬───────────────┘
                                         ↓
                              Streamlit Dashboard
                          (This interface you're using)
```
    """)
# ══════════════════════════════════════════════════════════
# PAGE 2 — NEW PREDICTION
# ══════════════════════════════════════════════════════════
elif page == "🔮 New Prediction":
    st.title("🔮 New Patient Prediction")
    st.markdown("Enter patient details below to get an AI-assisted diagnosis with full explainability.")
    st.divider()

    with st.form("prediction_form"):
        st.subheader("Patient Details")

        col1, col2 = st.columns(2)
        with col1:
            patient_id   = st.text_input("Patient ID", value=f"PAT-{datetime.now().strftime('%H%M%S')}")
            pregnancies  = st.slider("Pregnancies", 0, 17, 2)
            glucose      = st.slider("Glucose Level", 50, 250, 120)
            blood_press  = st.slider("Blood Pressure (mm Hg)", 30, 130, 70)
            skin_thick   = st.slider("Skin Thickness (mm)", 0, 100, 25)

        with col2:
            insulin   = st.slider("Insulin (μU/mL)", 0, 850, 80)
            bmi       = st.slider("BMI", 10.0, 70.0, 28.0, step=0.1)
            pedigree  = st.slider("Diabetes Pedigree Function", 0.0, 2.5, 0.5, step=0.01)
            age       = st.slider("Age", 18, 90, 35)

        submitted = st.form_submit_button("🔍 Analyze Patient")

    if submitted:
        patient_data = {
            "Pregnancies": pregnancies, "Glucose": glucose,
            "BloodPressure": blood_press, "SkinThickness": skin_thick,
            "Insulin": insulin, "BMI": bmi,
            "DiabetesPedigreeFunction": pedigree, "Age": age
        }

        with st.spinner("Running prediction & generating explanation..."):
            label, confidence = predict_and_log(patient_data, patient_id=patient_id)
            top_features, plot_path = explain_prediction(patient_data, patient_id=patient_id)

        st.divider()

        # ── Result Banner ──────────────────────────────────
        if label == "Diabetes":
            st.error(f"### 🔴 Diagnosis: **{label}**  —  Confidence: {confidence:.1%}")
        else:
            st.success(f"### 🟢 Diagnosis: **{label}**  —  Confidence: {confidence:.1%}")

        st.divider()

        col_l, col_r = st.columns(2)

        with col_l:
            st.subheader("📊 SHAP Explanation")
            st.caption("Features pushing the prediction up (red) or down (blue)")
            if os.path.exists(plot_path):
                st.image(plot_path, use_container_width=True)

        with col_r:
            st.subheader("🔑 Top Contributing Factors")
            for feat, val in top_features[:5]:
                direction = "🔴 Increases risk" if val > 0 else "🔵 Decreases risk"
                bar_color = "#ff4b4b" if val > 0 else "#4b9eff"
                bar_width = min(abs(val) * 150, 100)
                st.markdown(f"""
                **{feat}**
                <div style='background:{bar_color};width:{bar_width}%;
                height:8px;border-radius:4px;margin:2px 0 8px 0'></div>
                <small>{direction} &nbsp;|&nbsp; SHAP: {val:+.4f}</small>
                """, unsafe_allow_html=True)

        st.info(f"✅ Prediction logged to audit trail with ID: **{patient_id}**")

# ══════════════════════════════════════════════════════════
# PAGE 3 — AUDIT LOG
# ══════════════════════════════════════════════════════════
elif page == "📋 Audit Log":
    st.title("📋 Full Audit Log")
    st.markdown("Every prediction ever made — fully traceable and searchable.")
    st.divider()

    df_log = load_audit_log()

    if len(df_log) == 0:
        st.info("No predictions logged yet.")
    else:
        # ── Filters ────────────────────────────────────────
        col1, col2, col3 = st.columns(3)
        with col1:
            filter_label = st.selectbox(
                "Filter by Diagnosis",
                ["All", "Diabetes", "No Diabetes"]
            )
        with col2:
            min_conf = st.slider("Min Confidence", 0.0, 1.0, 0.0, 0.05)
        with col3:
            search_id = st.text_input("Search Patient ID", "")

        # Apply filters
        filtered = df_log.copy()
        if filter_label != "All":
            filtered = filtered[filtered["prediction_label"] == filter_label]
        filtered = filtered[filtered["confidence"] >= min_conf]
        if search_id:
            filtered = filtered[filtered["patient_id"].str.contains(
                search_id, case=False, na=False)]

        st.caption(f"Showing {len(filtered)} of {len(df_log)} records")

        display = filtered[[
            "id", "timestamp", "patient_id", "prediction_label",
            "confidence", "glucose", "bmi", "age", "model_version"
        ]].copy()
        display["confidence"] = display["confidence"].apply(lambda x: f"{x:.1%}")
        display.columns = [
            "ID", "Timestamp", "Patient ID", "Diagnosis",
            "Confidence", "Glucose", "BMI", "Age", "Model"
        ]
        st.dataframe(display, use_container_width=True, hide_index=True)

        # ── Download Button ────────────────────────────────
        csv = filtered.to_csv(index=False)
        st.download_button(
            "⬇️ Download Full Log as CSV",
            data=csv,
            file_name=f"audit_log_{datetime.now().strftime('%Y%m%d')}.csv",
            mime="text/csv"
        )

# ══════════════════════════════════════════════════════════
# PAGE 4 — DRIFT MONITOR
# ══════════════════════════════════════════════════════════
elif page == "📉 Drift Monitor":
    st.title("📉 Data Drift Monitor")
    st.markdown("Monitoring whether incoming patient data is shifting from the model's training distribution.")
    st.divider()

    col1, col2 = st.columns([1, 3])

    with col1:
        if st.button("🔄 Run Drift Check Now"):
            with st.spinner("Analyzing distributions..."):
                result = run_drift_check()
            if result:
                st.success("Analysis complete!")
            else:
                st.warning("Not enough data yet.")

    alerts = load_alerts()
    if alerts:
        latest = alerts[-1]
        severity = latest["severity"]

        with col2:
            if severity == "HIGH":
                st.markdown(f"""<div class='alert-high'>
                    🔴 <strong>HIGH DRIFT ALERT</strong> —
                    Detected in: {', '.join(latest['drifted_features'])} &nbsp;|&nbsp;
                    {latest['timestamp'][:19]}
                </div>""", unsafe_allow_html=True)
            elif severity == "MEDIUM":
                st.warning(f"⚠️ Medium drift detected in: "
                           f"{', '.join(latest['drifted_features'])}")
            else:
                st.markdown("""<div class='alert-ok'>
                    ✅ <strong>All Clear</strong> — No significant drift detected
                </div>""", unsafe_allow_html=True)

        st.divider()

        # ── KS Test Results Table ──────────────────────────
        st.subheader("📊 Feature Drift Analysis")
        details = latest["details"]
        rows = []
        for feat, info in details.items():
            rows.append({
                "Feature":      feat,
                "KS Statistic": info["ks_statistic"],
                "P-Value":      info["p_value"],
                "Status":       "🔴 DRIFT" if info["drift"] else "🟢 OK"
            })

        drift_df = pd.DataFrame(rows)
        st.dataframe(drift_df, use_container_width=True, hide_index=True)

        # ── Drift History ──────────────────────────────────
        st.divider()
        st.subheader("📈 Drift Check History")
        history_rows = []
        for a in alerts[-10:]:
            history_rows.append({
                "Timestamp": a["timestamp"][:19],
                "Drifted Features": ", ".join(a["drifted_features"]) or "None",
                "Severity": a["severity"]
            })
        st.dataframe(
            pd.DataFrame(history_rows),
            use_container_width=True,
            hide_index=True
        )

        # ── Distribution Comparison Charts ─────────────────
        st.divider()
        st.subheader("📉 Distribution Comparison")
        st.caption("Reference data (training) vs. recent predictions")

        ref_df     = load_reference()
        current_df = load_recent_predictions()

        if current_df is not None and len(current_df) >= 5:
            feat_list  = list(details.keys())
            cols       = st.columns(2)

            for i, feat in enumerate(feat_list[:6]):
                with cols[i % 2]:
                    fig, ax = plt.subplots(figsize=(5, 3), facecolor="#0e1117")
                    ax.set_facecolor("#1e2130")

                    ref_vals = ref_df[feat].dropna()
                    cur_vals = current_df[feat].dropna() if feat in current_df.columns else pd.Series()

                    ax.hist(ref_vals, bins=20, alpha=0.6,
                            color="#4b9eff", label="Reference", density=True)
                    if len(cur_vals) > 0:
                        ax.hist(cur_vals, bins=20, alpha=0.6,
                                color="#ff4b4b", label="Current", density=True)

                    is_drift = details.get(feat, {}).get("drift", False)
                    title_color = "#ff4b4b" if is_drift else "#00cc88"
                    ax.set_title(f"{feat} {'🔴' if is_drift else '🟢'}",
                                 color=title_color, fontsize=11)
                    ax.tick_params(colors="white")
                    ax.legend(fontsize=8, labelcolor="white",
                              facecolor="#1e2130", edgecolor="#3d4466")
                    for spine in ax.spines.values():
                        spine.set_edgecolor("#3d4466")

                    st.pyplot(fig)
                    plt.close()
    else:
        st.info("No drift analysis run yet. Click **Run Drift Check Now** to start.")

# ══════════════════════════════════════════════════════════
# PAGE 5 — MODEL REPORT
# ══════════════════════════════════════════════════════════
elif page == "📊 Model Report":
    st.title("📊 Model Report")
    st.markdown("Global model performance and feature importance overview.")
    st.divider()

    col1, col2 = st.columns(2)

    # ── Model Info ─────────────────────────────────────────
    with col1:
        st.subheader("🤖 Model Info")
        st.markdown("""
        | Property | Value |
        |---|---|
        | **Algorithm** | XGBoost Classifier |
        | **Version** | xgb_v1.0 |
        | **Dataset** | Pima Indians Diabetes |
        | **Training Samples** | 614 |
        | **Test Samples** | 154 |
        | **Features** | 8 |
        | **Target** | Binary (0 / 1) |
        """)

    with col2:
        st.subheader("📈 Performance Metrics")
        st.markdown("""
        | Metric | Score |
        |---|---|
        | **Accuracy** | ~77% |
        | **ROC-AUC** | ~0.84 |
        | **Precision (Diabetes)** | ~73% |
        | **Recall (Diabetes)** | ~67% |
        | **F1 Score** | ~70% |
        """)

    st.divider()

    # ── Global SHAP Summary ────────────────────────────────
    st.subheader("🔍 Global Feature Importance (SHAP)")
    st.caption("Which features matter most across ALL predictions?")

    summary_path = "logs/shap_plots/global_summary.png"

    if not os.path.exists(summary_path):
        if st.button("📊 Generate Global SHAP Summary"):
            with st.spinner("Calculating SHAP values for all test samples..."):
                global_shap_summary()
            st.rerun()
    else:
        st.image(summary_path, use_container_width=True)
        if st.button("🔄 Regenerate SHAP Summary"):
            with st.spinner("Recalculating..."):
                global_shap_summary()
            st.rerun()

    # ── SHAP Plots Gallery ─────────────────────────────────
    st.divider()
    st.subheader("🖼️ Individual SHAP Explanations Gallery")
    shap_dir = "logs/shap_plots"
    if os.path.exists(shap_dir):
        plots = [f for f in os.listdir(shap_dir)
                 if f.endswith("_shap.png")][:6]
        if plots:
            cols = st.columns(2)
            for i, plot_file in enumerate(plots):
                with cols[i % 2]:
                    patient = plot_file.replace("_shap.png", "")
                    st.caption(f"Patient: {patient}")
                    st.image(
                        os.path.join(shap_dir, plot_file),
                        use_container_width=True
                    )
        else:
            st.info("No individual SHAP plots yet. Make some predictions first!")


# ══════════════════════════════════════════════════════════
# PAGE 6 — MODEL COMPARISON
# ══════════════════════════════════════════════════════════
elif page == "🏆 Model Comparison":
    st.title("🏆 Multi-Model Comparison")
    st.markdown("Compare all trained models and manage the champion/challenger framework.")
    st.divider()

    import sys
    sys.path.append(".")
    from src.model_registry import (
        get_all_models, get_champion_model,
        set_champion, load_registry
    )
    import plotly.graph_objects as go

    registry = load_registry()
    
    if not registry["models"]:
        st.warning("⚠️ No models trained yet. Run `python src/train_all_models.py` first!")
        st.stop()

    models_df = get_all_models()
    champion = get_champion_model()

    # ── Champion Banner ────────────────────────────────────
    if champion:
        st.success(f"👑 **Current Champion:** {champion['model_name']} "
                   f"{champion['version']} (ROC-AUC: {champion['metrics']['roc_auc']:.4f})")
    
    st.divider()

    # ── Model Metrics Table ────────────────────────────────
    st.subheader("📊 All Models Performance")
    
    display_df = models_df[[
        "model_name", "version", "status", "timestamp",
        "metrics"
    ]].copy()
    
    # Extract metrics into columns
    display_df["accuracy"] = display_df["metrics"].apply(lambda x: f"{x['accuracy']:.2%}")
    display_df["precision"] = display_df["metrics"].apply(lambda x: f"{x['precision']:.2%}")
    display_df["recall"] = display_df["metrics"].apply(lambda x: f"{x['recall']:.2%}")
    display_df["f1_score"] = display_df["metrics"].apply(lambda x: f"{x['f1_score']:.2%}")
    display_df["roc_auc"] = display_df["metrics"].apply(lambda x: f"{x['roc_auc']:.4f}")
    
    display_df = display_df.drop("metrics", axis=1)
    display_df["timestamp"] = pd.to_datetime(display_df["timestamp"]).dt.strftime("%Y-%m-%d %H:%M")
    display_df.columns = ["Model", "Version", "Status", "Trained At",
                           "Accuracy", "Precision", "Recall", "F1", "ROC-AUC"]
    
    st.dataframe(display_df, use_container_width=True, hide_index=True)

    st.divider()

    # ── Visual Comparison Charts ───────────────────────────
    st.subheader("📈 Performance Comparison")

    col1, col2 = st.columns(2)

    with col1:
        # Metrics radar chart
        metrics_data = []
        for _, row in models_df.iterrows():
            m = row["metrics"]
            metrics_data.append({
                "model": f"{row['model_name']} {row['version']}",
                "accuracy": m["accuracy"],
                "precision": m["precision"],
                "recall": m["recall"],
                "f1_score": m["f1_score"],
                "roc_auc": m["roc_auc"]
            })

        fig_radar = go.Figure()

        for data in metrics_data:
            fig_radar.add_trace(go.Scatterpolar(
                r=[data["accuracy"], data["precision"], data["recall"],
                   data["f1_score"], data["roc_auc"]],
                theta=["Accuracy", "Precision", "Recall", "F1", "ROC-AUC"],
                fill="toself",
                name=data["model"]
            ))

        fig_radar.update_layout(
            polar=dict(
                radialaxis=dict(visible=True, range=[0, 1])
            ),
            showlegend=True,
            title="Multi-Metric Comparison",
            template="plotly_dark",
            plot_bgcolor="#0e1117",
            paper_bgcolor="#0e1117",
            font=dict(color="white"),
            height=400
        )
        st.plotly_chart(fig_radar, use_container_width=True)

    with col2:
        # Bar chart comparison
        metrics_list = ["accuracy", "precision", "recall", "f1_score", "roc_auc"]
        
        fig_bar = go.Figure()
        
        for metric in metrics_list:
            values = [row["metrics"][metric] for _, row in models_df.iterrows()]
            names = [f"{row['model_name']}" for _, row in models_df.iterrows()]
            
            fig_bar.add_trace(go.Bar(
                name=metric.replace("_", " ").title(),
                x=names,
                y=values
            ))
        
        fig_bar.update_layout(
            barmode="group",
            title="Metrics Side-by-Side",
            xaxis_title="Model",
            yaxis_title="Score",
            template="plotly_dark",
            plot_bgcolor="#0e1117",
            paper_bgcolor="#0e1117",
            font=dict(color="white"),
            height=400
        )
        st.plotly_chart(fig_bar, use_container_width=True)

    st.divider()

    # ── Champion Management ────────────────────────────────
    st.subheader("👑 Champion Management")
    st.caption("Promote a model to champion status (will be used for all new predictions)")

    col_a, col_b = st.columns([3, 1])

    with col_a:
        model_options = [f"{row['model_name']} {row['version']}" 
                         for _, row in models_df.iterrows()]
        selected = st.selectbox("Select model to promote", model_options)

    with col_b:
        st.write("")  # Spacing
        st.write("")  # Spacing
        if st.button("🏆 Set as Champion", use_container_width=True):
            model_name = selected.split()[0]
            set_champion(model_name)
            st.success(f"✅ {model_name} is now champion!")
            st.rerun()

    st.divider()

    # ── Confusion Matrices ─────────────────────────────────
    st.subheader("🎯 Confusion Matrices")
    
    cols = st.columns(3)
    
    for i, (_, row) in enumerate(models_df.iterrows()):
        with cols[i % 3]:
            cm = np.array(row["metrics"]["confusion_matrix"])
            
            fig_cm = go.Figure(data=go.Heatmap(
                z=cm,
                x=["Predicted No", "Predicted Yes"],
                y=["Actual No", "Actual Yes"],
                colorscale="Blues",
                text=cm,
                texttemplate="%{text}",
                textfont={"size": 14},
                showscale=False
            ))
            
            fig_cm.update_layout(
                title=f"{row['model_name']} {row['version']}",
                template="plotly_dark",
                plot_bgcolor="#0e1117",
                paper_bgcolor="#0e1117",
                font=dict(color="white"),
                height=300,
                xaxis=dict(side="bottom")
            )
            st.plotly_chart(fig_cm, use_container_width=True)

# ══════════════════════════════════════════════════════════
# PAGE 7 — FAIRNESS & BIAS
# ══════════════════════════════════════════════════════════
elif page == "⚖️ Fairness & Bias":
    st.title("⚖️ Fairness & Bias Analysis")
    st.markdown("Comprehensive fairness evaluation across demographic groups with bias detection and mitigation strategies.")
    st.divider()

    import sys
    sys.path.append(".")
    from src.fairness_analysis import (
        analyze_bias_from_audit_log,
        load_fairness_report,
        generate_mitigation_strategies
    )
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    # ── Run Analysis Button ────────────────────────────────
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        if st.button("🔍 Run Fairness Analysis", use_container_width=True):
            with st.spinner("Analyzing bias across demographic groups..."):
                report = analyze_bias_from_audit_log()
                if report and "error" not in report:
                    st.success("✅ Analysis complete!")
                    st.rerun()
                else:
                    st.warning(report.get("error", "Analysis failed"))

    st.divider()

    # ── Load Report ────────────────────────────────────────
    report = load_fairness_report()
    
    if not report:
        st.info("📊 No fairness analysis run yet. Click **Run Fairness Analysis** to start.")
        st.stop()
    
    if "error" in report:
        st.warning(f"⚠️ {report['error']}")
        st.stop()

    # ── Overall Assessment Banner ──────────────────────────
    assessment = report["overall_assessment"]
    status = assessment["status"]
    
    if status == "FAIR":
        st.success(f"""
        ### ✅ FAIR — Model passes fairness criteria
        **Recommendation:** {assessment['recommendation']}
        """)
    elif status == "MODERATE_BIAS":
        st.warning(f"""
        ### ⚠️ MODERATE BIAS DETECTED
        **Issues:** {', '.join(assessment['issues'])}
        
        **Recommendation:** {assessment['recommendation']}
        """)
    else:
        st.error(f"""
        ### 🔴 SIGNIFICANT BIAS DETECTED
        **Issues:** {', '.join(assessment['issues'])}
        
        **Recommendation:** {assessment['recommendation']}
        """)

    st.divider()

    # ── Key Metrics Cards ──────────────────────────────────
    st.subheader("📊 Fairness Metrics Overview")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        total = report["total_predictions"]
        st.markdown(f"""
        <div class='metric-card'>
            <div class='metric-value'>{total}</div>
            <div class='metric-label'>Predictions Analyzed</div>
        </div>""", unsafe_allow_html=True)
    
    with col2:
        passes = assessment.get("passes_80_rule", False)
        status_text = "PASS" if passes else "FAIL"
        color = "#00cc88" if passes else "#ff4b4b"
        st.markdown(f"""
        <div class='metric-card'>
            <div class='metric-value' style='color:{color}'>{status_text}</div>
            <div class='metric-label'>80% Rule</div>
        </div>""", unsafe_allow_html=True)
    
    with col3:
        tpr_disp = assessment.get("max_tpr_disparity", 0)
        st.markdown(f"""
        <div class='metric-card'>
            <div class='metric-value'>{tpr_disp:.2%}</div>
            <div class='metric-label'>Max TPR Disparity</div>
        </div>""", unsafe_allow_html=True)
    
    with col4:
        eq_odds = assessment.get("max_equalized_odds", 0)
        st.markdown(f"""
        <div class='metric-card'>
            <div class='metric-value'>{eq_odds:.2%}</div>
            <div class='metric-label'>Equalized Odds Gap</div>
        </div>""", unsafe_allow_html=True)

    st.divider()

    # ── Group Metrics Table ────────────────────────────────
    st.subheader("👥 Metrics by Demographic Group")
    st.caption(f"Protected Attribute: {report['protected_attribute']}")
    
    group_metrics = report["group_metrics"]
    
    metrics_rows = []
    for group, metrics in group_metrics.items():
        metrics_rows.append({
            "Age Group": group,
            "Sample Size": metrics["total_samples"],
            "Selection Rate": f"{metrics['selection_rate']:.2%}",
            "True Positive Rate": f"{metrics['true_positive_rate']:.2%}",
            "False Positive Rate": f"{metrics['false_positive_rate']:.2%}",
            "Precision": f"{metrics['precision']:.2%}"
        })
    
    metrics_df = pd.DataFrame(metrics_rows)
    st.dataframe(metrics_df, use_container_width=True, hide_index=True)

    st.divider()

    # ── Disparity Analysis ─────────────────────────────────
    st.subheader("📉 Disparity Analysis")
    
    disparities = report["disparities"]
    
    if "groups" in disparities:
        st.caption(f"Reference Group: **{disparities['reference_group']}** (largest sample size)")
        
        disp_rows = []
        for group, disp in disparities["groups"].items():
            status_emoji = "✅" if disp["passes_80_rule"] else "⚠️"
            disp_rows.append({
                "Age Group": group,
                "Disparate Impact Ratio": f"{disp['disparate_impact_ratio']:.3f}",
                "80% Rule": f"{status_emoji} {'PASS' if disp['passes_80_rule'] else 'FAIL'}",
                "Selection Rate Diff": f"{disp['selection_rate_diff']:+.2%}",
                "Equal Opportunity Diff": f"{disp['equal_opportunity_diff']:.4f}",
                "Equalized Odds Diff": f"{disp['equalized_odds_diff']:.4f}"
            })
        
        disp_df = pd.DataFrame(disp_rows)
        st.dataframe(disp_df, use_container_width=True, hide_index=True)
        
        st.caption("""
        **80% Rule:** Disparate Impact Ratio should be between 0.80 and 1.25  
        **Equal Opportunity:** TPR difference should be < 0.10  
        **Equalized Odds:** Average of TPR and FPR differences should be < 0.10
        """)

    st.divider()

    # ── Visualization Charts ───────────────────────────────
    st.subheader("📊 Fairness Visualizations")

    col_left, col_right = st.columns(2)

    with col_left:
        # Selection Rate Comparison
        groups = list(group_metrics.keys())
        selection_rates = [group_metrics[g]["selection_rate"] for g in groups]
        
        fig_selection = go.Figure()
        fig_selection.add_trace(go.Bar(
            x=groups,
            y=selection_rates,
            marker_color="#7eb3ff",
            text=[f"{sr:.1%}" for sr in selection_rates],
            textposition="outside"
        ))
        
        # Add 80% rule bounds if we have reference
        if "groups" in disparities:
            ref_rate = group_metrics[disparities['reference_group']]["selection_rate"]
            fig_selection.add_hline(
                y=ref_rate * 0.8,
                line_dash="dash",
                line_color="red",
                annotation_text="80% threshold"
            )
            fig_selection.add_hline(
                y=ref_rate * 1.25,
                line_dash="dash",
                line_color="red",
                annotation_text="125% threshold"
            )
        
        fig_selection.update_layout(
            title="Selection Rate by Age Group",
            xaxis_title="Age Group",
            yaxis_title="Selection Rate",
            template="plotly_dark",
            plot_bgcolor="#0e1117",
            paper_bgcolor="#0e1117",
            font=dict(color="white"),
            height=400
        )
        st.plotly_chart(fig_selection, use_container_width=True)

    with col_right:
        # TPR vs FPR Comparison
        tpr_values = [group_metrics[g]["true_positive_rate"] for g in groups]
        fpr_values = [group_metrics[g]["false_positive_rate"] for g in groups]
        
        fig_rates = go.Figure()
        fig_rates.add_trace(go.Bar(
            name="True Positive Rate",
            x=groups,
            y=tpr_values,
            marker_color="#00cc88"
        ))
        fig_rates.add_trace(go.Bar(
            name="False Positive Rate",
            x=groups,
            y=fpr_values,
            marker_color="#ff4b4b"
        ))
        
        fig_rates.update_layout(
            title="TPR vs FPR by Age Group",
            xaxis_title="Age Group",
            yaxis_title="Rate",
            barmode="group",
            template="plotly_dark",
            plot_bgcolor="#0e1117",
            paper_bgcolor="#0e1117",
            font=dict(color="white"),
            height=400
        )
        st.plotly_chart(fig_rates, use_container_width=True)

    # ── Confusion Matrix Comparison ────────────────────────
    st.subheader("🎯 Confusion Matrices by Group")
    
    cols = st.columns(len(groups))
    
    for i, group in enumerate(groups):
        with cols[i]:
            metrics = group_metrics[group]
            cm = np.array([
                [metrics["tn"], metrics["fp"]],
                [metrics["fn"], metrics["tp"]]
            ])
            
            fig_cm = go.Figure(data=go.Heatmap(
                z=cm,
                x=["Pred No", "Pred Yes"],
                y=["Actual No", "Actual Yes"],
                colorscale="Blues",
                text=cm,
                texttemplate="%{text}",
                textfont={"size": 12},
                showscale=False
            ))
            
            fig_cm.update_layout(
                title=f"{group}",
                template="plotly_dark",
                plot_bgcolor="#0e1117",
                paper_bgcolor="#0e1117",
                font=dict(color="white"),
                height=250
            )
            st.plotly_chart(fig_cm, use_container_width=True)

    st.divider()

    # ── Mitigation Strategies ──────────────────────────────
    st.subheader("🛠️ Bias Mitigation Strategies")
    
    strategies = generate_mitigation_strategies(report)
    
    if strategies:
        for i, strategy in enumerate(strategies, 1):
            priority = strategy["priority"]
            
            if priority == "CRITICAL":
                st.error(f"""
                **{i}. [{priority}] {strategy['strategy']}**
                
                {strategy['description']}
                
                ▶️ **Action:** {strategy['action']}
                """)
            elif priority == "HIGH":
                st.warning(f"""
                **{i}. [{priority}] {strategy['strategy']}**
                
                {strategy['description']}
                
                ▶️ **Action:** {strategy['action']}
                """)
            else:
                st.info(f"""
                **{i}. [{priority}] {strategy['strategy']}**
                
                {strategy['description']}
                
                ▶️ **Action:** {strategy['action']}
                """)
    else:
        st.success("✅ No critical mitigation needed. Continue monitoring.")

    st.divider()

    # ── Fairness Report Download ───────────────────────────
    st.subheader("📄 Export Report")
    
    report_json = json.dumps(report, indent=2)
    st.download_button(
        "⬇️ Download Full Fairness Report (JSON)",
        data=report_json,
        file_name=f"fairness_report_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.json",
        mime="application/json"
    )

# ══════════════════════════════════════════════════════════
# PAGE 8 — PRODUCTION MONITORING
# ══════════════════════════════════════════════════════════
elif page == "🏭 Production Monitor":
    st.title("🏭 Production Monitoring")
    st.markdown("Real-time operational monitoring, performance tracking, and automated retraining triggers.")
    st.divider()

    import sys
    sys.path.append(".")
    from src.production_monitor import (
        calculate_rolling_performance,
        detect_performance_degradation,
        monitor_data_quality,
        monitor_system_health,
        evaluate_retraining_need,
        load_alerts
    )
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    # ── Refresh Data ───────────────────────────────────────
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        if st.button("🔄 Refresh Monitoring Data", use_container_width=True):
            with st.spinner("Refreshing all monitoring metrics..."):
                monitor_data_quality()
                st.success("✅ Data refreshed!")
                st.rerun()

    st.divider()

    # ── System Health Overview ─────────────────────────────
    st.subheader("💚 System Health Overview")
    
    health = monitor_system_health()
    
    if "error" not in health:
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            status = health["operational_status"]
            color = "#00cc88" if status == "HEALTHY" else "#ff4b4b"
            st.markdown(f"""
            <div class='metric-card'>
                <div class='metric-value' style='color:{color}'>{status}</div>
                <div class='metric-label'>System Status</div>
            </div>""", unsafe_allow_html=True)
        
        with col2:
            uptime = health["uptime_days"]
            st.markdown(f"""
            <div class='metric-card'>
                <div class='metric-value'>{uptime}</div>
                <div class='metric-label'>Uptime (Days)</div>
            </div>""", unsafe_allow_html=True)
        
        with col3:
            rate = health["volume_metrics"]["prediction_rate"]
            st.markdown(f"""
            <div class='metric-card'>
                <div class='metric-value'>{rate:.1f}</div>
                <div class='metric-label'>Predictions/Hour</div>
            </div>""", unsafe_allow_html=True)
        
        with col4:
            last_24h = health["volume_metrics"]["last_24h"]
            st.markdown(f"""
            <div class='metric-card'>
                <div class='metric-value'>{last_24h}</div>
                <div class='metric-label'>Last 24 Hours</div>
            </div>""", unsafe_allow_html=True)

    st.divider()

    # ── Performance Tracking ───────────────────────────────
    st.subheader("📊 Performance Tracking")
    
    perf_data = calculate_rolling_performance()
    
    if "error" not in perf_data:
        # Current metrics
        curr = perf_data["current_metrics"]
        
        col_a, col_b = st.columns(2)
        
        with col_a:
            st.metric("Current Accuracy", f"{curr['accuracy']:.2%}")
            st.metric("Average Confidence", f"{curr['avg_confidence']:.2%}")
        
        with col_b:
            # Degradation check
            degrad = detect_performance_degradation()
            if degrad["degradation_detected"]:
                st.error(f"""
                **⚠️ Performance Degradation Detected**
                
                Accuracy dropped by {degrad['drop_amount']:.2%}  
                (Current: {degrad['current_accuracy']:.2%} | Baseline: {degrad['baseline_accuracy']:.2%})
                
                Severity: **{degrad['severity']}**
                """)
            else:
                st.success(f"""
                **✅ Performance Stable**
                
                Current: {degrad['current_accuracy']:.2%}  
                Baseline: {degrad['baseline_accuracy']:.2%}
                """)
        
        # Daily performance trend
        if perf_data["daily_metrics"]:
            st.caption("📈 Daily Performance Trends")
            
            daily_df = pd.DataFrame(perf_data["daily_metrics"])
            
            fig_perf = make_subplots(
                rows=2, cols=1,
                subplot_titles=("Accuracy & F1 Score Over Time", "Precision & Recall Over Time"),
                vertical_spacing=0.12
            )
            
            # Top chart: Accuracy & F1
            fig_perf.add_trace(
                go.Scatter(x=daily_df["date"], y=daily_df["accuracy"],
                          name="Accuracy", mode="lines+markers",
                          line=dict(color="#7eb3ff", width=2)),
                row=1, col=1
            )
            fig_perf.add_trace(
                go.Scatter(x=daily_df["date"], y=daily_df["f1_score"],
                          name="F1 Score", mode="lines+markers",
                          line=dict(color="#00cc88", width=2)),
                row=1, col=1
            )
            
            # Bottom chart: Precision & Recall
            fig_perf.add_trace(
                go.Scatter(x=daily_df["date"], y=daily_df["precision"],
                          name="Precision", mode="lines+markers",
                          line=dict(color="#ff9f40", width=2)),
                row=2, col=1
            )
            fig_perf.add_trace(
                go.Scatter(x=daily_df["date"], y=daily_df["recall"],
                          name="Recall", mode="lines+markers",
                          line=dict(color="#ff6b9d", width=2)),
                row=2, col=1
            )
            
            fig_perf.update_xaxes(title_text="Date", row=2, col=1)
            fig_perf.update_yaxes(title_text="Score", range=[0, 1])
            
            fig_perf.update_layout(
                template="plotly_dark",
                plot_bgcolor="#0e1117",
                paper_bgcolor="#0e1117",
                font=dict(color="white"),
                height=600,
                showlegend=True
            )
            
            st.plotly_chart(fig_perf, use_container_width=True)

    st.divider()

    # ── Data Quality Monitoring ────────────────────────────
    st.subheader("🔍 Data Quality Monitoring")
    
    quality = monitor_data_quality()
    
    if "error" not in quality:
        # Quality score
        score = quality["quality_score"]
        score_color = "#00cc88" if score >= 0.9 else "#ffaa00" if score >= 0.7 else "#ff4b4b"
        
        col1, col2 = st.columns([1, 3])
        
        with col1:
            st.markdown(f"""
            <div class='metric-card'>
                <div class='metric-value' style='color:{score_color}'>{score:.1%}</div>
                <div class='metric-label'>Data Quality Score</div>
            </div>""", unsafe_allow_html=True)
        
        with col2:
            if quality["issues"]:
                st.warning(f"**{len(quality['issues'])} Quality Issues Detected:**")
                for issue in quality["issues"]:
                    st.markdown(f"• {issue}")
            else:
                st.success("✅ No data quality issues detected")
        
        # Feature-level quality metrics
        st.caption("📊 Quality Metrics by Feature")
        
        quality_rows = []
        for feature, metrics in quality["features"].items():
            quality_rows.append({
                "Feature": feature.replace("_", " ").title(),
                "Missing %": f"{metrics['missing_percentage']:.1%}",
                "Outliers %": f"{metrics['outlier_percentage']:.1%}",
                "Range Issues": metrics["range_violations"],
                "Mean": f"{metrics['mean']:.1f}",
                "Std Dev": f"{metrics['std']:.1f}"
            })
        
        quality_df = pd.DataFrame(quality_rows)
        st.dataframe(quality_df, use_container_width=True, hide_index=True)

    st.divider()

    # ── Retraining Assessment ──────────────────────────────
    st.subheader("🔄 Automated Retraining Assessment")
    
    retrain_eval = evaluate_retraining_need()
    
    should_retrain = retrain_eval["should_retrain"]
    
    if should_retrain:
        st.error(f"""
        ### ⚠️ RETRAINING RECOMMENDED
        
        **{retrain_eval['recommendation']}**
        """)
    else:
        st.success(f"""
        ### ✅ No Retraining Needed
        
        {retrain_eval['recommendation']}
        """)
    
    if retrain_eval["triggers"]:
        st.caption("🎯 Detected Triggers:")
        
        for trigger in retrain_eval["triggers"]:
            severity = trigger["severity"]
            
            if severity in ["CRITICAL", "HIGH"]:
                st.error(f"**[{severity}] {trigger['type']}**  \n{trigger['details']}")
            elif severity == "MEDIUM":
                st.warning(f"**[{severity}] {trigger['type']}**  \n{trigger['details']}")
            else:
                st.info(f"**[{severity}] {trigger['type']}**  \n{trigger['details']}")

    st.divider()

    # ── Recent Alerts Log ──────────────────────────────────
    st.subheader("🚨 Recent Production Alerts")
    
    alerts = load_alerts()
    
    if alerts:
        recent_alerts = alerts[-10:][::-1]  # Last 10, newest first
        
        for alert in recent_alerts:
            timestamp = pd.to_datetime(alert["timestamp"]).strftime("%Y-%m-%d %H:%M:%S")
            alert_type = alert["type"]
            
            with st.expander(f"🔔 {alert_type} — {timestamp}"):
                st.json(alert["details"])
    else:
        st.info("No alerts logged yet")

    st.divider()

    # ── Operational Metrics ────────────────────────────────
    st.subheader("📊 Operational Metrics")
    
    if "error" not in health:
        vol = health["volume_metrics"]
        conf = health["confidence_metrics"]
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Volume Metrics**")
            st.markdown(f"• Last Hour: **{vol['last_hour']}** predictions")
            st.markdown(f"• Last 24 Hours: **{vol['last_24h']}** predictions")
            st.markdown(f"• Last 7 Days: **{vol['last_7days']}** predictions")
            st.markdown(f"• Total Lifetime: **{vol['total_predictions']}** predictions")
        
        with col2:
            st.markdown("**Confidence Metrics**")
            st.markdown(f"• Overall Average: **{conf['overall_avg']:.2%}**")
            st.markdown(f"• Recent Average: **{conf['recent_avg']:.2%}**")
            st.markdown(f"• Trend: **{conf['trend']}**")