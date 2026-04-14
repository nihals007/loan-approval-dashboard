import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.figure_factory as ff
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                              f1_score, roc_auc_score, confusion_matrix, roc_curve)

CHART_THEME = dict(
    paper_bgcolor='rgba(0,0,0,0)',
    plot_bgcolor='rgba(22,33,62,1)',
    font=dict(family='Outfit, sans-serif', color='#94a3b8'),
    colorway=['#d4ff00', '#00d4b4', '#7c5cbf', '#ff6b35', '#60a5fa']
)

def show_metrics(model, X_test, y_test):
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    # Big metric cards
    st.subheader("📊 Model Evaluation Results")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Accuracy",  f"{accuracy_score(y_test, y_pred)*100:.2f}%")
    col2.metric("Precision", f"{precision_score(y_test, y_pred)*100:.2f}%")
    col3.metric("Recall",    f"{recall_score(y_test, y_pred)*100:.2f}%")
    col4.metric("F1 Score",  f"{f1_score(y_test, y_pred)*100:.2f}%")
    st.metric("ROC-AUC Score", f"{roc_auc_score(y_test, y_prob)*100:.2f}%")

    # Confusion Matrix
    st.subheader("🔢 Confusion Matrix")
    cm = confusion_matrix(y_test, y_pred)
    fig = ff.create_annotated_heatmap(
        cm,
        x=['Predicted: No', 'Predicted: Yes'],
        y=['Actual: No', 'Actual: Yes'],
        colorscale='Blues'
    )
    fig.update_layout(**CHART_THEME) 
    st.plotly_chart(fig, use_container_width=True)

    # ROC Curve
    st.subheader("📈 ROC Curve")
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    roc_df = pd.DataFrame({'False Positive Rate': fpr, 'True Positive Rate': tpr})
    fig2 = px.area(roc_df, x='False Positive Rate', y='True Positive Rate',
                   title=f'ROC Curve (AUC = {roc_auc_score(y_test, y_prob):.4f})')
    fig2.add_shape(type='line', line=dict(dash='dash', color='red'),
                   x0=0, x1=1, y0=0, y1=1)
    fig.update_layout(**CHART_THEME) 
    st.plotly_chart(fig2, use_container_width=True)