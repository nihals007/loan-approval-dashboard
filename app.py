import streamlit as st
import pandas as pd
from pipeline.eda import run_eda
from pipeline.cleaning import clean_data
from pipeline.feature_selection import select_features
from pipeline.split import split_data
from pipeline.model import train_and_validate, MODELS
from pipeline.metrics import show_metrics
from pipeline.predictor import show_predictor

# ── Page config ──────────────────────────────────────────────
st.set_page_config(
    page_title="Loan Approval ML Pipeline",
    layout="wide",
    page_icon="🏦"
)

def load_css():
    with open("assets/style.css", encoding="utf-8") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

load_css()

# ── Sidebar (like in video — left panel) ─────────────────────
with st.sidebar:
    st.header("1. Data Source")
    uploaded_file = st.file_uploader("Upload CSV", type=["csv"])

    if uploaded_file:
        st.success("File Uploaded!")

    st.markdown("---")
    st.subheader("⚙️ Settings")
    test_size     = st.slider("Test Size", 0.1, 0.4, 0.2)
    k_folds       = st.slider("K-Fold Splits", 3, 10, 5)
    model_choice  = st.selectbox("Select Model", list(MODELS.keys()))
    run_btn       = st.button("🚀 Run Full Pipeline", type="primary")

# ── Main area ─────────────────────────────────────────────────
st.title("🏦 Interactive ML Pipeline Dashboard")

# Load data
if uploaded_file:
    raw_df = pd.read_csv(uploaded_file)
else:
    try:
        raw_df = pd.read_csv("data/loan_data.csv")
        st.info("ℹ️ Using default dataset. Upload your own CSV in the sidebar.")
    except:
        st.warning("⚠️ Please upload a CSV file to get started.")
        st.stop()

# Target column selector (like video dropdown!)
all_cols   = raw_df.columns.tolist()
all_cols = [col.strip() for col in raw_df.columns.tolist()]
raw_df.columns = all_cols

# Only show columns that are text OR have fewer than 5 unique values
valid_targets = [
    col for col in all_cols
    if raw_df[col].dtype == 'object' or raw_df[col].nunique() <= 4
]

target_col = st.selectbox(
    "🎯 Select Target Variable",
    valid_targets,
    index=valid_targets.index('loan_status') if 'loan_status' in valid_targets else 0
)

st.markdown("---")

# ── Tabs (exactly like the video top navigation) ──────────────
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "📊 Data & EDA",
    "🧹 Cleaning & Engineering",
    "🎯 Feature Selection",
    "🤖 Model Training",
    "📈 Performance",
    "🔮 Live Predictor"       # ← new
])

# ── TAB 1: Data & EDA ─────────────────────────────────────────
with tab1:
    st.subheader("📂 Raw Data Preview")
    st.dataframe(raw_df.head(20), use_container_width=True)
    run_eda(raw_df, target_col)

# ── TAB 2: Cleaning ───────────────────────────────────────────
with tab2:
    st.subheader("🧹 Data Engineering & Cleaning")

    # Outlier removal toggle (like in video!)
    remove_outliers = st.checkbox("Remove outliers using IQR?", value=True)
    missing_strategy = st.radio("Handle missing values:",
                                 ["Impute Median", "Delete Rows", "Keep as-is"],
                                 horizontal=True)
    if st.button("Apply Cleaning"):
        cleaned_df = clean_data(raw_df)
        st.session_state['cleaned_df'] = cleaned_df
        st.success(f"✅ Data cleaned! New shape: {cleaned_df.shape}")
        st.dataframe(cleaned_df.head(), use_container_width=True)
    elif 'cleaned_df' not in st.session_state:
        st.info("Click 'Apply Cleaning' to process the data.")

# ── TAB 3: Feature Selection ──────────────────────────────────
with tab3:
    st.subheader("🎯 Feature Engineering & Selection")

    if 'cleaned_df' not in st.session_state:
        st.warning("⚠️ Please complete the Cleaning step first.")
    else:
        method = st.radio("Select Method:",
                          ["All Features", "Variance Threshold", "Information Gain"],
                          horizontal=True)
        if st.button("Run Feature Selection"):
            top_features = select_features(
                st.session_state['cleaned_df'], target_col, method
            )
            st.session_state['features'] = top_features

# ── TAB 4: Model Training ─────────────────────────────────────
with tab4:
    st.subheader("🤖 Model Training & K-Fold Validation")

    if 'features' not in st.session_state or 'cleaned_df' not in st.session_state:
        st.warning("⚠️ Complete Cleaning & Feature Selection steps first.")
    else:
        df_clean = st.session_state['cleaned_df']
        features = st.session_state['features']

        X_train, X_test, y_train, y_test = split_data(
            df_clean, features, target_col, test_size
        )
        st.session_state['split'] = (X_train, X_test, y_train, y_test)

        if run_btn or st.button("▶️ Train Model"):
            with st.spinner(f"Training {model_choice}..."):
                trained_model = train_and_validate(X_train, y_train, model_choice, k_folds)
                st.session_state['model'] = trained_model
            st.success(f"✅ {model_choice} trained successfully!")

# ── TAB 5: Performance ────────────────────────────────────────
with tab5:
    st.subheader("📈 Model Evaluation Results")

    if 'model' not in st.session_state or 'split' not in st.session_state:
        st.warning("⚠️ Please train a model first.")
    else:
        _, X_test, _, y_test = st.session_state['split']
        show_metrics(st.session_state['model'], X_test, y_test)

with tab6:
    if 'model' not in st.session_state or 'features' not in st.session_state:
        st.markdown("""
            <div style='
                text-align:center; padding:60px 20px;
                background:#16213e; border-radius:18px;
                border: 1px dashed rgba(212,255,0,0.2);
            '>
                <div style='font-size:3rem;margin-bottom:16px;'>🔮</div>
                <div style='color:#d4ff00;font-family:Outfit,sans-serif;
                            font-size:1.2rem;font-weight:700;'>
                    Train a model first
                </div>
                <div style='color:#94a3b8;font-family:Outfit,sans-serif;
                            font-size:0.9rem;margin-top:8px;'>
                    Complete the Model Training tab to unlock the Live Predictor
                </div>
            </div>
        """, unsafe_allow_html=True)
    else:
        show_predictor(
            st.session_state['model'],
            st.session_state['features']
        )