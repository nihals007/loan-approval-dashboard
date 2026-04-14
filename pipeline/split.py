import streamlit as st
from sklearn.model_selection import train_test_split

def split_data(df, features, target_col, test_size=0.2):
    X = df[features]
    y = df[target_col]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42, stratify=y
    )

    col1, col2, col3 = st.columns(3)
    col1.metric("Total Samples", len(X))
    col2.metric("Training Samples", len(X_train))
    col3.metric("Testing Samples", len(X_test))

    import plotly.express as px
    import pandas as pd
    split_df = pd.DataFrame({'Split': ['Train', 'Test'], 'Count': [len(X_train), len(X_test)]})
    fig = px.pie(split_df, names='Split', values='Count', title='Train / Test Split',
                 color_discrete_sequence=['#2196F3', '#FF5722'])
    st.plotly_chart(fig, use_container_width=True)

    return X_train, X_test, y_train, y_test