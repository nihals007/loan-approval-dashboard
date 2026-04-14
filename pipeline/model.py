import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score
from xgboost import XGBClassifier

CHART_THEME = dict(
    paper_bgcolor='rgba(0,0,0,0)',
    plot_bgcolor='rgba(22,33,62,1)',
    font=dict(family='Outfit, sans-serif', color='#94a3b8'),
    colorway=['#d4ff00', '#00d4b4', '#7c5cbf', '#ff6b35', '#60a5fa']
)

MODELS = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Random Forest":       RandomForestClassifier(n_estimators=100, random_state=42),
    "Gradient Boosting":   GradientBoostingClassifier(random_state=42),
    "XGBoost":             XGBClassifier(use_label_encoder=False,
                                         eval_metric='logloss', random_state=42)
}

def train_and_validate(X_train, y_train, model_name, k=5):
    model = MODELS[model_name]
    kf = StratifiedKFold(n_splits=k, shuffle=True, random_state=42)

    # Run K-Fold for all 4 metrics
    acc    = cross_val_score(model, X_train, y_train, cv=kf, scoring='accuracy')
    prec   = cross_val_score(model, X_train, y_train, cv=kf, scoring='precision')
    recall = cross_val_score(model, X_train, y_train, cv=kf, scoring='recall')
    f1     = cross_val_score(model, X_train, y_train, cv=kf, scoring='f1')

    # Show fold-by-fold accuracy chart (like in video — "Stability Across K-Folds")
    fold_df = pd.DataFrame({
        'Fold':     [f'Fold {i+1}' for i in range(k)],
        'Accuracy': acc
    })
    fig = px.area(fold_df, x='Fold', y='Accuracy',
                  title='Stability Across K-Folds',
                  color_discrete_sequence=['#2196F3'])
    fig.update_layout(**CHART_THEME) 
    st.plotly_chart(fig, use_container_width=True)

    # Show avg metrics (exactly like the video!)
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Avg Accuracy",  f"{acc.mean()*100:.2f}%")
    col2.metric("Avg Precision", f"{prec.mean()*100:.2f}%")
    col3.metric("Avg Recall",    f"{recall.mean()*100:.2f}%")
    col4.metric("Avg F1 Score",  f"{f1.mean()*100:.2f}%")

    # Final fit on full training data
    model.fit(X_train, y_train)
    return model