import streamlit as st
import plotly.express as px
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import VarianceThreshold
from sklearn.inspection import permutation_importance

CHART_THEME = dict(
    paper_bgcolor='rgba(0,0,0,0)',
    plot_bgcolor='rgba(22,33,62,1)',
    font=dict(family='Outfit, sans-serif', color='#94a3b8'),
    colorway=['#d4ff00', '#00d4b4', '#7c5cbf', '#ff6b35', '#60a5fa']
)

def select_features(df, target_col, method='All Features'):
    X = df.drop(columns=[target_col])
    y = df[target_col]

    if method == 'Variance Threshold':
        selector = VarianceThreshold(threshold=0.1)
        selector.fit(X)
        selected = X.columns[selector.get_support()].tolist()

    elif method == 'Information Gain':
        from sklearn.feature_selection import mutual_info_classif
        scores = mutual_info_classif(X, y, random_state=42)
        importance_df = pd.Series(scores, index=X.columns).sort_values(ascending=False)
        fig = px.bar(importance_df, orientation='h', title='Information Gain per Feature')
        fig.update_layout(**CHART_THEME) 
        st.plotly_chart(fig, use_container_width=True)
        selected = importance_df[importance_df > 0].index.tolist()

    else:  # All Features
        selected = X.columns.tolist()

    # Feature importance chart using Random Forest
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X[selected], y)
    imp = pd.Series(rf.feature_importances_, index=selected).sort_values(ascending=True)
    fig = px.bar(imp, orientation='h', title='Feature Importance (Random Forest)',
                 color=imp.values, color_continuous_scale='Blues')
    fig.update_layout(**CHART_THEME) 
    st.plotly_chart(fig, use_container_width=True)

    st.success(f"✅ {len(selected)} features selected")
    st.json(selected)

    return selected