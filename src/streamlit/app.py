import streamlit as st
import pandas as pd
import plotly.express as px
import joblib
import json
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns

# ================================================================
# ⚙️ CONFIG
# ================================================================
st.set_page_config(page_title="Road Accidents in France (2005–2023)", layout="wide")

# Project root relative to this file:  src/streamlit/app.py  ->  nov24cds_road_accidents
ROOT = Path(__file__).resolve().parents[2]

# Paths (all relative to ROOT now)
MODEL_PATH   = ROOT / "models" / "for_streamlit" / "model_BalancedRandomForest_2class_streamlit_FOR_STREAMLIT.joblib"
SUMMARY_PATH = ROOT / "models" / "for_streamlit" / "streamlit_summary_balancedrandomforest_2class_streamlit_FOR_STREAMLIT.json"
FI_CSV_PATH  = ROOT / "models" / "for_streamlit" / "feature_importances_top30_balancedrandomforest_2class_streamlit_FOR_STREAMLIT.csv"
FIG_DIR      = ROOT / "reports" / "figures_for_streamlit"
DATA_PATH    = ROOT / "data" / "processed" / "df_for_ml.csv"
METRICS_PATH = ROOT / "reports" / "model_results.csv"
MAP_PATH     = ROOT / "reports" / "figures" / "geospatial distribution of road accidents across France.png"

# Safe: tiny JSON, load once
threshold = 0.5
try:
    with open(SUMMARY_PATH, "r", encoding="utf-8") as f:
        summary = json.load(f)
        threshold = float(summary.get("threshold", 0.5))
except Exception as e:
    st.sidebar.warning(f"Could not load summary JSON: {e}")
    threshold = 0.5

# Keep threshold across all pages (global session state)
if "decision_threshold" not in st.session_state:
    st.session_state.decision_threshold = float(threshold)

# Nav
PAGES = [
    "Understanding and manipulation of data",
    "Visualizations",
    "Classification & Modeling",
    "Final Model & Explainability",
    "Make Predictions",
    "Conclusion & Outlook",
]
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", PAGES)

# ================================================================
# Helpers (cached)
# ================================================================
@st.cache_data(show_spinner=False)
def load_df(path: Path) -> pd.DataFrame:
    return pd.read_csv(path)

@st.cache_resource(show_spinner=False)
def load_model_cached() -> object | None:
    """
    Robust loader: try memory-mapped (RAM friendly), fall back to normal load,
    and surface a clean error if the pickle is incompatible.
    """
    try:
        return joblib.load(MODEL_PATH, mmap_mode="r")
    except Exception:
        try:
            return joblib.load(MODEL_PATH)  # fallback without memmap
        except Exception as e:
            st.error(
                "❌ Failed to load the trained model.\n\n"
                "This usually happens when the model was saved with different "
                "NumPy / scikit-learn versions, or the file is corrupted.\n\n"
                f"Details: {e}"
            )
            # Tiny env diagnostics to help align versions
            import numpy as _np, sklearn as _sk, joblib as _jb
            st.caption(
                f"Environment: numpy={_np.__version__}  scikit-learn={_sk.__version__}  joblib={_jb.__version__}"
            )
            st.info("Fix: re-save the model in the same environment as Streamlit, or pin matching versions.")
            return None

# ================================================================
# 1) Understanding and manipulation of data
# ================================================================
if page == PAGES[0]:
    # Header only on this page
    st.markdown(
        """
        <div style="text-align: center; margin-top: 10px; margin-bottom: 12px;">
            <h1 style="font-size:2.8em; color:black; font-weight:700; margin-bottom:0;">
                Road Accidents in France
            </h1>
            <p style="font-size:1.1em; color:#333333; margin-top:10px;">
                Based on Annual Road Traffic Accident Injury Database (2005–2023)
            </p>
        </div>
        <hr style="margin-top:20px; margin-bottom:25px;">
        """,
        unsafe_allow_html=True,
    )

    st.title("Understanding and manipulation of data")
    st.markdown(
        """
This project focuses on predicting the **severity of road accidents in France** using historical data from the **BAAC**, 2005–2023.
The dataset covers **environmental**, **temporal**, **behavioral**, and **road infrastructure** factors influencing outcomes.

**Challenges**
- Missing or inconsistent data  
- Changes in injury classification (since 2018)  
- Strong **class imbalance** (few severe cases)  

Despite these, modeling provides actionable insights for **data-driven road safety**.
        """
    )

    st.markdown(
        """
### Pre-processing and Feature Engineering
- **Integration:** yearly BAAC tables merged on accident ID  
- **Outliers/Validation:** set implausible values to NA; invalid category codes → NA  
- **Datetime:** built timestamp; extracted `hour`, `dayofweek`  
- **Safety equipment:** unified pre/post-2019 into `belt_status`, `helmet_status`, `child_device_status`, `reflective_vest_status`  
- **Cleaning:** dropped high-missing, low-relevance, redundant, and messy text fields  
- **Preprocessing:** numeric → median impute + `StandardScaler`; categorical → mode impute + `OneHotEncoder`  
- **Dimensionality:** variance threshold (0.01), Cramér’s V merges, grouped high-cardinality `catv`  
- **Features:** `rush_hour`, `season`, `age_bins`, `user_belt_status`
        """
    )

# ================================================================
# 2) Visualizations (Compact, Small Fonts)
# ================================================================
elif page == PAGES[1]:
    st.title("Data Visualization")

    # ---------- Load & preprocess ----------
    try:
        VISUAL_DATA_PATH = ROOT / "data" / "processed" / "accidents_cleaned.csv"
        df = load_df(VISUAL_DATA_PATH)

        severity_labels = {1: "Unharmed", 2: "Killed", 3: "Hospitalized", 4: "Light injury"}
        if "grav" in df.columns:
            df["grav_label"] = df["grav"].map(severity_labels)
    except Exception as e:
        st.error(f"❌ Failed to load cleaned data file: {e}")
        st.stop()

    # ---------- Styling ----------
    plt.rcParams.update({
        "axes.titlesize": 9,
        "axes.labelsize": 8,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "legend.fontsize": 7,
    })

    color_map = {
        "Unharmed": "#2ecc71",
        "Light injury": "#e67e22",
        "Hospitalized": "#e74c3c",
        "Killed": "#8e44ad"
    }

    # ============================================================
    # TEMPORAL PATTERNS
    # ============================================================
    st.header("TEMPORAL PATTERNS")

    st.markdown("### **Figure 1 — Road Accidents by Hour of Day and Day of Week**")
    if {"dayofweek", "hour"}.issubset(df.columns):
        df["_ones"] = 1
        pivot = df.pivot_table(index="dayofweek", columns="hour", values="_ones",
                               aggfunc="sum", fill_value=0)
        pivot = pivot.reindex(range(0, 7))
        pivot.index = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]

        fig, ax = plt.subplots(figsize=(5.5, 3.5))
        sns.heatmap(pivot, cmap="YlGnBu", ax=ax, cbar_kws={'shrink': 0.6})
        ax.set_xlabel("Hour of Day")
        ax.set_ylabel("Day of Week")
        ax.set_title("Road Accidents by Hour and Day", fontsize=9)
        plt.tight_layout()
        st.pyplot(fig)

    st.markdown("### **Figure 2 — Monthly Road Accident Trends (2005–2023)**")
    if "datetime" in df.columns:
        df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce")
        df["month_year"] = df["datetime"].dt.to_period("M").dt.to_timestamp()
        monthly_counts = df.groupby("month_year").size()

        fig, ax = plt.subplots(figsize=(7.5, 3.2))
        ax.plot(monthly_counts.index, monthly_counts.values, marker="o", markersize=2.5,
                linewidth=0.9, color="#003366")
        ax.set_xlabel("Month-Year")
        ax.set_ylabel("Accidents")
        ax.set_title("Monthly Road Accident Trends (2005–2023)", fontsize=9)
        plt.xticks(rotation=45)
        plt.tight_layout()
        st.pyplot(fig)

    # ============================================================
    # SEVERITY DISTRIBUTION
    # ============================================================
    st.header("SEVERITY DISTRIBUTION")
    st.markdown("### **Figure 3 — Distribution of Road Accident Severity**")
    if "grav_label" in df.columns:
        fig, ax = plt.subplots(figsize=(5, 3))
        order = ["Unharmed", "Light injury", "Hospitalized", "Killed"]
        df["grav_label"].value_counts().reindex(order).plot(
            kind="bar", color=[color_map[o] for o in order], ax=ax, edgecolor="black"
        )
        ax.set_xlabel("Severity Level")
        ax.set_ylabel("Count")
        ax.set_title("Severity Distribution", fontsize=9)
        plt.tight_layout()
        st.pyplot(fig)

    # ============================================================
    # USER ROLE
    # ============================================================
    st.markdown("### **Figure 4 — Injury Severity by User Role (%)**")
    if {"catu", "grav_label"}.issubset(df.columns):
        df["user_role"] = df["catu"].map({1: "Driver", 2: "Passenger", 3: "Pedestrian"})
        tab = df.groupby(["user_role", "grav_label"]).size().unstack(fill_value=0)
        tab = tab.div(tab.sum(axis=1), axis=0) * 100

        fig, ax = plt.subplots(figsize=(5.5, 3))
        tab.plot(kind="bar", stacked=True, color=[color_map[c] for c in tab.columns],
                 ax=ax, edgecolor="black")
        ax.set_ylabel("%")
        ax.set_title("Injury Severity by User Role", fontsize=9)
        plt.tight_layout()
        st.pyplot(fig)

    # ============================================================
    # SEAT BELT
    # ============================================================
    st.markdown("### **Figure 5 — Seat Belt Usage vs Injury Severity (Light Vehicles)**")
    if {"belt_status", "grav_label", "catv"}.issubset(df.columns):
        light = df[df["catv"] == 7].copy()
        light["belt_status_label"] = light["belt_status"].map({1: "Used", 0: "Not Used"})
        tab = light.groupby(["belt_status_label", "grav_label"]).size().unstack(fill_value=0)
        tab = tab.div(tab.sum(axis=1), axis=0) * 100

        fig, ax = plt.subplots(figsize=(5, 3))
        tab.T.plot(kind="bar", stacked=True, color=[color_map[c] for c in tab.columns],
                   ax=ax, edgecolor="black")
        ax.set_ylabel("%")
        ax.set_title("Seat Belt Usage vs Severity", fontsize=9)
        plt.tight_layout()
        st.pyplot(fig)

    # ============================================================
    # SEX
    # ============================================================
    st.markdown("### **Figure 6 — Accident Severity by Sex (%)**")
    if {"sexe", "grav_label"}.issubset(df.columns):
        df["gender_label"] = df["sexe"].map({1: "Male", 2: "Female"})
        tab = df.groupby(["gender_label", "grav_label"]).size().unstack(fill_value=0)
        tab = tab.div(tab.sum(axis=1), axis=0) * 100

        fig, ax = plt.subplots(figsize=(5, 3))
        tab.plot(kind="bar", stacked=True, color=[color_map[c] for c in tab.columns],
                 ax=ax, edgecolor="black")
        ax.set_ylabel("%")
        ax.set_title("Severity by Sex", fontsize=9)
        plt.tight_layout()
        st.pyplot(fig)

    # ============================================================
    # GEOGRAPHICAL VARIATION
    # ============================================================
    st.header("GEOGRAPHICAL VARIATION")
    st.markdown("### **Figure 7 — Urban vs Rural Severity Distribution (%)**")
    if {"agg", "grav_label"}.issubset(df.columns):
        df["agg_label"] = df["agg"].map({1: "Urban", 2: "Rural"})
        tab = df.groupby(["agg_label", "grav_label"]).size().unstack(fill_value=0)
        tab = tab.div(tab.sum(axis=1), axis=0) * 100

        fig, ax = plt.subplots(figsize=(5.5, 3))
        tab.plot(kind="bar", stacked=True, color=[color_map[c] for c in tab.columns],
                 ax=ax, edgecolor="black")
        ax.set_ylabel("%")
        ax.set_title("Urban vs Rural Severity", fontsize=9)
        plt.tight_layout()
        st.pyplot(fig)

    # ============================================================
    # MAP
    # ============================================================
    st.markdown("### **Figure 8 — Geospatial Distribution of Road Accidents in France**")
    _map_path = ROOT / "reports" / "figures" / "geospatial distribution of road accidents across France.png"
    if _map_path.exists():
        st.image(str(_map_path),
                 caption="Legend: Green – Unharmed, Orange – Light injury, Red – Hospitalized, Purple – Killed.",
                 width=750)

    # ============================================================
    # LIGHTING & WEATHER
    # ============================================================
    st.header("LIGHTING & WEATHER")
    st.markdown("### **Figure 9 — Severity by Lighting Conditions (%)**")
    if {"lum", "atm", "grav_label"}.issubset(df.columns):
        lum_labels = {
            1: "Daylight", 2: "Dawn/Dusk", 3: "Night – No Lighting",
            4: "Night – Lighting Not Lit", 5: "Night – Lighting Lit"
        }
        atm_labels = {
            1: "Normal", 2: "Light Rain", 3: "Heavy Rain", 4: "Snow/Hail",
            5: "Fog/Smoke", 6: "Storm", 7: "Dazzling", 8: "Cloudy", 9: "Other"
        }
        df["lum_label"] = df["lum"].map(lum_labels)
        df["atm_label"] = df["atm"].map(atm_labels)

        tab_lum = df.groupby(["lum_label", "grav_label"]).size().unstack(fill_value=0)
        tab_lum = tab_lum.div(tab_lum.sum(axis=1), axis=0) * 100

        fig, ax = plt.subplots(figsize=(7, 3))
        tab_lum.plot(kind="bar", stacked=True, color=[color_map[c] for c in tab_lum.columns],
                     ax=ax, edgecolor="black")
        ax.set_ylabel("%")
        ax.set_title("Lighting Conditions vs Severity", fontsize=9)
        plt.tight_layout()
        st.pyplot(fig)

        st.markdown("### **Figure 10 — Severity by Atmospheric Conditions (%)**")
        tab_atm = df.groupby(["atm_label", "grav_label"]).size().unstack(fill_value=0)
        tab_atm = tab_atm.div(tab_atm.sum(axis=1), axis=0) * 100

        fig, ax = plt.subplots(figsize=(7, 3))
        tab_atm.plot(kind="bar", stacked=True, color=[color_map[c] for c in tab_atm.columns],
                     ax=ax, edgecolor="black")
        ax.set_ylabel("%")
        ax.set_title("Atmospheric Conditions vs Severity", fontsize=9)
        plt.tight_layout()
        st.pyplot(fig)




# ================================================================
# 3) Classification & Modeling
# ================================================================
elif page == PAGES[2]:
    st.title("Classification & Modeling")

    METRICS_PATH = ROOT / "reports" / "model_results.csv"

    st.markdown(
        """
        <style>
        h3 {
            color: #003366;
            font-weight: 700;
            margin-top: 25px;
            margin-bottom: 10px;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

    st.markdown("""
### Classification of the Problem
This project is formulated as a **supervised classification task**, where each accident record is labeled with a target variable **`grav`**:
- *Unharmed*
- *Light injury*
- *Hospitalized*
- *Killed*

The goal is to **predict injury severity** given accident, vehicle, and user characteristics.
    """)

    st.markdown("""
### Type of Machine Learning Problem & Task Context
The task involves **multi-class classification** (4 original labels) but is reformulated under several **label aggregation schemes** to handle **severe class imbalance**:
- **4-class:** Unharmed, Light injury, Hospitalized, Killed (original labels)  
- **3-class:** Unharmed/Light injury combined, Hospitalized, Killed  
- **2-class:** Minor/None (Unharmed + Light injury) vs. Severe (Hospitalized + Killed)  
    """)

    st.markdown("""
###  Performance Metrics
In road safety analytics, **detecting severe or fatal accidents** is more valuable than minimizing false alarms.  
Thus, the **primary metric** is **Recall for the “Killed” (or Severe)** class — ensuring that true critical cases are not missed.

To evaluate models comprehensively:
- **Recall, Precision, and F1-score** were reported for all severity classes.  
- **Accuracy** serves as a secondary measure but is less informative under imbalance (A naive model could achieve 80–90% accuracy but 0% severe recall)
- **Macro and Weighted Averages** ensure fair comparison across minority classes.  
- **Confusion Matrices** visualize misclassification patterns.  
    """)

    st.markdown("""
### Model Choice and Optimization
To address imbalance and assess label granularity, multiple algorithms were trained under **2-, 3-, and 4-class formulations**:

- **Logistic Regression (LR):** interpretable baseline  
- **Decision Tree (DT):** non-linear patterns, simple interpretability  
- **Random Forest (RF):** ensemble robustness; tested with undersampling and class weighting  
- **Balanced Bagging (RF base):** variance reduction with balanced bootstraps  
- **XGBoost / LightGBM / CatBoost:** gradient boosting with imbalance handling (weights, undersampling)  

**Hyperparameter tuning** was done with Grid Search (without cross-validation) to reduce computation time.  
Benchmarks used a **20% stratified sample** of the full dataset to accelerate evaluation.
""")

    # ========== Load Metrics ==========
    try:
        df = pd.read_csv(METRICS_PATH, sep=";", engine="python")
        if len(df.columns) == 1:
            df = df.iloc[:, 0].str.split(";", expand=True)
            df.columns = [
                "setup", "model", "accuracy", "macro_precision", "macro_recall", "macro_f1",
                "positive_precision", "positive_recall", "positive_f1"
            ]
        # Convert numerics safely
        for col in df.columns[2:]:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        df["model_display"] = df["setup"] + " - " + df["model"]
    except Exception as e:
        st.error(f"❌ Could not load model metrics: {e}")
        st.stop()

    # ========== Interactive Plots ==========
    st.markdown("### **Figure 11 — Accuracy vs. Macro-F1: Trade-off Between Overall and Balanced Performance**")
    fig1 = px.scatter(
        df, x="accuracy", y="macro_f1",
        color="setup", hover_name="model",
        size="positive_f1", size_max=18
    )
    st.caption(
        "Each point represents a trained model. "
        "**Dot size** reflects the *Severe-class F1-score* — larger circles indicate stronger performance "
        "in identifying severe accidents while maintaining balanced precision and recall."
    )
    fig1.update_layout(legend_title_text="Setup", template="plotly_white")
    st.plotly_chart(fig1, use_container_width=True)

    st.markdown("### **Figure 12 — Recall for the Positive (Severe) Class: Detecting Critical Cases**")
    fig2 = px.bar(
        df.sort_values("positive_recall", ascending=False),
        x="positive_recall", y="model_display",
        color="setup", orientation="h",
    )
    fig2.update_layout(yaxis_title="", xaxis_title="Recall", template="plotly_white")
    st.plotly_chart(fig2, use_container_width=True)

    st.markdown("### **Figure 13 — Macro-F1 vs. Severe-Class F1: Balancing Fairness and Risk Detection**")
    fig3 = px.scatter(
        df, x="macro_f1", y="positive_f1",
        color="setup", hover_name="model",
        size="accuracy", size_max=18
    )
    st.caption(
    "**Dot size** corresponds to overall model **accuracy**. "
    "Larger dots indicate models achieving both high balanced performance (Macro-F1) "
    "and high severe-case detection quality (Severe-class F1)."
    )
    fig3.update_layout(template="plotly_white", legend_title_text="Setup")
    st.plotly_chart(fig3, use_container_width=True)

    # ========== Short Insight ==========
    st.markdown("""

Results indicate:
- **2-class models** achieve best overall balance (Macro-F1 ≈ 0.60–0.75).  
- **3-class models** show slightly higher accuracy but lower F1.  
- **4-class models** perform worst due to label fragmentation. 

>**Final Choice:**  
> The **Balanced Random Forest (2-class)** achieved **high recall** on severe cases, offering the best robustness to imbalance.  
> It was selected as the final deployed model for predictive analysis.
    """)

# ================================================================
# 4) Final Model & Explainability
# ================================================================
elif page == PAGES[3]:
    import json, numpy as np, pandas as pd, plotly.express as px
    import matplotlib.pyplot as plt, seaborn as sns

    st.title("Final Model & Explainability")

    # ============================================================
    # STYLE
    # ============================================================
    st.markdown(
        """
        <style>
        h3 {color:#003366; font-weight:700; margin-top:25px; margin-bottom:10px;}
        </style>
        """,
        unsafe_allow_html=True,
    )

    # ============================================================
    # MODEL OVERVIEW
    # ============================================================
    st.markdown("### Model Overview")
    st.markdown(
        """
        This section presents the **Balanced Random Forest (2-Class)** models used to predict  
        **road accident severity** in France (2005–2023).

        The **Streamlit model** was trained on a **20% stratified sample** for interactivity,  
        while the **full model** was trained on the complete dataset for benchmark comparison.  
        Both share the same preprocessing, grid search setup, and feature engineering pipeline.

        #### Training Dataset
        - Total records available: **5,513,703**
        - 20% stratified sample: **1,102,740 rows**, **29 features**
        - Target classes: *Severe (Hospitalized/Killed)* vs *Non-severe (Minor/None)*

        #### Training Parameters (20% Subset)
        - `n_estimators = 500`
        - `max_depth = None`
        - `min_samples_split = 2`
        - Sampling strategy: **balanced bootstrap**
        - **Best threshold = 0.50**
        - **Recall (Severe)** = 0.73 | **Precision (Severe)** = 0.46 | **F1 = 0.57** | **Accuracy = 0.83**

        #### Training Parameters (Full Dataset)
        - `n_estimators = 500`
        - `max_depth = None`
        - `min_samples_split = 2`
        - Sampling strategy: **balanced bootstrap**
        - **Best threshold = 0.30**
        - **Recall (Severe)** = 0.97 | **F1 = 0.95** | **Accuracy = 0.89**

        Both models achieve high performance, with the full dataset improving recall and F1 on the critical “Severe” class.  
        """
)

    # ============================================================
    # CONFUSION MATRICES
    # ============================================================
    st.markdown("---")
    st.markdown("### **Figure 14 — Confusion Matrix Comparison (Normalized %)**")

    FULL_JSON = ROOT / "models" / "results_BalancedRandomForest_2class_FULL_DATA.json"
    SUB_JSON  = ROOT / "models" / "for_streamlit" / "results_BalancedRandomForest_2class_streamlit_FOR_STREAMLIT.json"

    try:
        with open(FULL_JSON, "r", encoding="utf-8") as f:
            full_json = json.load(f)
        with open(SUB_JSON, "r", encoding="utf-8") as f:
            sub_json = json.load(f)

        cm_full = np.array(full_json.get("confusion_matrix_percent", []))
        cm_sub  = np.array(sub_json.get("confusion_matrix_percent", []))
        labels = ["Actual Severe", "Actual Minor/None"]
        cols   = ["Predicted Severe", "Predicted Minor/None"]

        vmax = max(cm_full.max(), cm_sub.max())
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        sns.heatmap(cm_full, annot=True, fmt=".1f", cmap="Blues", cbar=False,
                    xticklabels=cols, yticklabels=labels, ax=axes[0], vmin=0, vmax=vmax)
        axes[0].set_title("Full Model (100 %)")

        sns.heatmap(cm_sub, annot=True, fmt=".1f", cmap="YlGnBu", cbar=False,
                    xticklabels=cols, yticklabels=labels, ax=axes[1], vmin=0, vmax=vmax)
        axes[1].set_title("Subset Model (20 %)")
        plt.tight_layout()
        st.pyplot(fig)

        st.caption(
            "Both matrices show **normalized percentages (%)**. "
            "Labels correspond to severity classes without numeric codes."
        )
    except Exception as e:
        st.error(f"Could not display confusion matrices: {e}")

    # ============================================================
    # CLASSIFICATION REPORTS
    # ============================================================
    st.markdown("---")
    st.markdown("### **Figure 15 — Classification Reports**")

    try:
        full_rep = full_json.get("metrics_final_threshold", {}).get("report", {})
        sub_rep  = sub_json.get("metrics_final_threshold", {}).get("report", {})

        def tidy_report(report_dict):
            df = pd.DataFrame(report_dict).T
            keep = [c for c in ["precision", "recall", "f1-score"] if c in df.columns]
            df = df[keep].astype(float).round(3)
            df.rename_axis("Class", inplace=True)
            df.index = [c.replace("0", "Minor/None").replace("1", "Severe") for c in df.index]
            return df

        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Full Dataset Model (100 %)**")
            st.dataframe(tidy_report(full_rep), use_container_width=True)
        with col2:
            st.markdown("**Subset Model (20 %)**")
            st.dataframe(tidy_report(sub_rep), use_container_width=True)

        st.caption("Precision, recall, and F1-score for each class — rounded to three decimals.")
    except Exception as e:
        st.error(f"Could not render classification reports: {e}")

    # ============================================================
    # ROC & PR CURVES (20%)
    # ============================================================
    st.markdown("---")
    st.markdown("### **Figures 16–17 — ROC & Precision–Recall Curves (20 % Subset)**")

    pr_path = ROOT / "reports" / "figures_for_streamlit" / "BalancedRandomForest_2class_streamlit_pr_FOR_STREAMLIT.png"
    roc_path = ROOT / "reports" / "figures_for_streamlit" / "BalancedRandomForest_2class_streamlit_roc_FOR_STREAMLIT.png"

    if pr_path.exists() and roc_path.exists():
        c1, c2 = st.columns(2)
        with c1:
            st.image(str(pr_path), caption="Figure 16 — Precision–Recall Curve (20 % Subset)", use_column_width=True)
        with c2:
            st.image(str(roc_path), caption="Figure 17 — ROC Curve (20 % Subset)", use_column_width=True)
    else:
        st.warning("PR or ROC curve image not found in `reports/figures_for_streamlit/`.")

    # ============================================================
    # FEATURE IMPORTANCES (20%)
    # ============================================================
    st.markdown("---")
    st.markdown("### **Figure 18 — Top 30 Feature Importances (20 % Subset Model)**")

    try:
        fi_csv_path = ROOT / "models" / "for_streamlit" / "feature_importances_top30_balancedrandomforest_2class_streamlit_FOR_STREAMLIT.csv"
        fi_df = pd.read_csv(fi_csv_path)
        feat_col = next(c for c in fi_df.columns if c.lower().startswith(("feat", "var", "name")))
        imp_col  = next(c for c in fi_df.columns if c.lower().startswith(("imp", "gain", "weight")))

        topn = fi_df.nlargest(30, imp_col).iloc[::-1]
        fig_fi = px.bar(
            topn, x=imp_col, y=feat_col, orientation="h",
            labels={imp_col: "Importance", feat_col: "Feature"},
            title="Top 30 Features — Balanced Random Forest (20 % Subset)",
            template="plotly_white", height=640,
        )
        st.plotly_chart(fig_fi, use_container_width=True)
        st.caption("Feature importances derived from the 20 % subset training run.")
    except Exception as e:
        st.warning(f"Could not load feature importances: {e}")

    # ============================================================
    # SHAP VALUES (20%)
    # ============================================================
    st.markdown("---")
    st.markdown("### **Figure 19 — Explainability (SHAP Values, 20% Subset)**")

    shap_dir = ROOT / "reports" / "shap" / "balancedrandomforest_2class_streamlit"
    interactive_dir = shap_dir / "interactive"
    merged_html = shap_dir / "interactive_shap_dashboard.html"

    # --- Static SHAP images ---
    shap_imgs = sorted(shap_dir.glob("*.png"))
    if shap_imgs:
        st.subheader("Static SHAP Visualizations")
        for p in shap_imgs:
            st.image(str(p), use_column_width=True)
    else:
        st.info("No static SHAP plots found in `reports/shap/balancedrandomforest_2class_streamlit/`.")

    # --- Interactive SHAP HTMLs ---
    if interactive_dir.exists():
        interactive_files = sorted(interactive_dir.glob("*.html"))
        if interactive_files:
            st.subheader("Interactive SHAP Visualizations")
            for html_file in interactive_files:
                name = html_file.stem.replace("_", " ").title()
                with st.expander(f"🔹 {name}"):
                    html_content = open(html_file, encoding="utf-8").read()
                    st.components.v1.html(html_content, height=500, scrolling=True)
        else:
            st.info("No interactive SHAP HTMLs found.")
    else:
        st.info("Interactive SHAP directory not found.")




    # ============================================================
    # KEY INSIGHTS
    # ============================================================
    st.markdown("---")
    st.subheader("🔍 Key Insights")
    st.markdown(
        """
        - The **20 % subset model** mirrors the full model’s behaviour while being lightweight for interactive exploration.  
        - **Balanced Random Forest** effectively handles severe class imbalance through balanced bootstrapping.  
        - **Threshold = 0.30** achieves high recall for severe outcomes with strong overall balance.  
        - **Top predictive features:** seat-belt use, lighting, road type, weather, and user/vehicle factors.  
        - **SHAP** results confirm these as dominant drivers influencing predicted severity.  
        """
    )





# ================================================================
# 5) Make Predictions   (MODEL ONLY LOADS HERE)
# ================================================================
elif page == PAGES[4]:
    st.title("🔮 Make Predictions")
    st.markdown("Upload a CSV file with the **same structure** as the model input features.")

    model = load_model_cached()
    if model is None:
        st.stop()

    # ---- User threshold control ----
    st.sidebar.subheader("Decision Threshold")
    st.session_state.decision_threshold = st.sidebar.slider(
        "Select threshold:",
        min_value=0.0, max_value=1.0, value=float(st.session_state.decision_threshold),
        step=0.01,
        help="Adjusts sensitivity between False Negatives and False Positives."
    )
    threshold = st.session_state.decision_threshold

    # ---- File upload ----
    uploaded = st.file_uploader("📂 Upload input CSV", type="csv")

    if uploaded:
        try:
            with st.spinner("Processing uploaded file..."):
                import csv

                # Try reading first line to detect separator safely
                first_line = uploaded.getvalue().decode("utf-8").splitlines()[0]
                try:
                    dialect = csv.Sniffer().sniff(first_line)
                    sep = dialect.delimiter
                except Exception:
                    sep = ","  # fallback default

                # Load CSV with detected or default separator
                df_input = pd.read_csv(uploaded, sep=sep)
                if df_input.shape[1] == 1 and ";" in df_input.columns[0]:
                    st.warning("⚠️ Detected semicolon-separated file — reloading with sep=';' ...")
                    uploaded.seek(0)
                    df_input = pd.read_csv(uploaded, sep=";")

                st.write(f"✅ Loaded {df_input.shape[0]:,} rows × {df_input.shape[1]:,} columns.")
                st.dataframe(df_input.head(3))

                preproc = model.named_steps["preprocessor"]
                clf = model.named_steps["classifier"]

                # Schema validation
                expected_cols = getattr(preproc, "feature_names_in_", None)
                if expected_cols is not None:
                    missing = set(expected_cols) - set(df_input.columns)
                    extra = set(df_input.columns) - set(expected_cols)
                    if missing:
                        st.warning(f"⚠️ Missing expected columns: {', '.join(sorted(missing))}")
                    if extra:
                        st.info(f"ℹ️ Extra columns ignored: {', '.join(sorted(extra))}")
                    df_input = df_input[[c for c in expected_cols if c in df_input.columns]]

                # Predict
                X_proc = preproc.transform(df_input)
                if hasattr(X_proc, "toarray"):
                    X_proc = X_proc.toarray()

                probas = clf.predict_proba(X_proc)[:, 1]  # 1 = Severe
                preds = (probas >= threshold).astype(int)

                df_input["Severity_Proba"] = probas
                df_input["Prediction"] = preds
                df_input["Prediction_Label"] = df_input["Prediction"].map({1: "Severe", 0: "Minor/None"})

                # --- Display ---
                st.success("✅ Predictions complete.")
                st.dataframe(df_input[["Severity_Proba", "Prediction_Label"]].head(10))

                # Probability distribution
                fig = px.histogram(
                    df_input,
                    x="Severity_Proba",
                    nbins=20,
                    title="Prediction Probability Distribution (Severe Class)",
                    labels={"Severity_Proba": "Predicted Probability (Severe)"},
                    template="plotly_white",
                )
                st.plotly_chart(fig, use_container_width=True)

                # Download
                csv_bytes = df_input.to_csv(index=False).encode("utf-8")
                st.download_button("💾 Download Predictions", csv_bytes, "predictions.csv", "text/csv")

        except Exception as e:
            st.error(f"❌ Error during prediction:\n\n{e}")

# ================================================================
# 6) Conclusion & Outlook
# ================================================================
elif page == PAGES[5]:
    st.title("Conclusion & Outlook")
    st.markdown(
        """
**Key Takeaways**  
- Balanced Random Forest handled imbalance best  
- Binary (2-class) setup outperformed multi-class  
- SHAP aligned with domain knowledge

**Next Steps**  
- Deploy as an interactive tool  
- Tune hyperparameters with more compute  
- Refine targets (e.g., separate Killed vs. Hospitalized)
        """
    )

# ================================================================
# Footer and Visual Consistency
# ================================================================
st.markdown(
    """
    <hr style='margin-top:30px; margin-bottom:10px;'>
    <div style='text-align:center; color:grey; font-size:0.9em;'>
        Road Accident Severity Prediction Dashboard (2005–2023)  
        <br>Developed for Data Science Course 2024–2025
    </div>
    """,
    unsafe_allow_html=True,
)

# Sidebar styling
st.markdown(
    """
    <style>
    [data-testid="stSidebar"] {
        background-color: #f8f9fa;
    }
    </style>
    """,
    unsafe_allow_html=True,
)
