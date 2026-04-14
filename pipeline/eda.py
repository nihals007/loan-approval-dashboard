import streamlit as st
import plotly.express as px

CHART_THEME = dict(
    paper_bgcolor='rgba(0,0,0,0)',
    plot_bgcolor='rgba(22,33,62,1)',
    font=dict(family='Outfit, sans-serif', color='#94a3b8'),
    colorway=['#d4ff00', '#00d4b4', '#7c5cbf', '#ff6b35', '#60a5fa']
)

def run_eda(df, target_col):
    st.subheader("📋 Dataset Summary")
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Rows", df.shape[0])
    col2.metric("Total Columns", df.shape[1])
    col3.metric("Missing Values", df.isnull().sum().sum())
    st.dataframe(df.describe())

    # Target distribution
    st.subheader("🎯 Loan Approval Distribution")
    counts = df[target_col].astype(str).str.strip().value_counts().reset_index()
    counts.columns = ['Value', 'Count']
    fig = px.pie(counts, names='Value', values='Count',
                title=f'Distribution of {target_col}',
                color_discrete_sequence=['#d4ff00', '#00d4b4'])
    fig.update_layout(**CHART_THEME)
    st.plotly_chart(fig, use_container_width=True)

    # Correlation heatmap
    st.subheader("🔥 Correlation Heatmap")
    numeric_df = df.select_dtypes(include='number')
    corr = numeric_df.corr()
    fig2 = px.imshow(corr, text_auto=True, color_continuous_scale='RdBu_r')
    fig.update_layout(**CHART_THEME) 
    st.plotly_chart(fig2, use_container_width=True)

    # Key feature distributions
    st.subheader("📊 Key Feature Distributions")
    key_cols = ['income_annum', 'loan_amount', 'cibil_score',
                'loan_term', 'residential_assets_value', 'bank_asset_value']
    for col in key_cols:
        if col in df.columns:
            fig3 = px.histogram(df, x=col, color=target_col,
                                barmode='overlay', title=f'{col} Distribution')
            fig.update_layout(**CHART_THEME) 
            st.plotly_chart(fig3, use_container_width=True)

    # Education & self_employed breakdown
    st.subheader("📌 Categorical Breakdowns")
    for col in ['education', 'self_employed']:
        if col in df.columns:
            fig4 = px.histogram(df, x=col, color=target_col,
                                barmode='group', title=f'Loan Status by {col}')
            fig.update_layout(**CHART_THEME) 
            st.plotly_chart(fig4, use_container_width=True)