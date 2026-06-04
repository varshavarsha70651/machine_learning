import streamlit as st
import pandas as pd
import plotly.express as px
from model import train_model
from utils import preprocess_data

st.set_page_config(page_title="ML Classification Dashboard", layout="wide")

st.title("🤖 Machine Learning Classification Dashboard")

uploaded_file = st.file_uploader("Upload CSV File", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)

    st.subheader("📄 Dataset Preview")
    st.dataframe(df.head())

    # Select target column
    target = st.selectbox("🎯 Select Target Column", df.columns)

    if st.button("🚀 Run Analysis"):

        # Preprocess
        X_train, X_test, y_train, y_test, feature_names = preprocess_data(df, target)

        # Train model
        model, accuracy, cm, importance = train_model(X_train, X_test, y_train, y_test, feature_names)

        st.subheader("📊 Model Performance")

        col1, col2 = st.columns(2)
        col1.metric("Accuracy", f"{accuracy:.2f}")

        # Confusion Matrix
        st.subheader("🔍 Confusion Matrix")
        fig_cm = px.imshow(cm, text_auto=True, title="Confusion Matrix")
        st.plotly_chart(fig_cm, use_container_width=True)

        # Feature Importance
        st.subheader("📌 Feature Importance")
        imp_df = pd.DataFrame({
            "Feature": feature_names,
            "Importance": importance
        }).sort_values(by="Importance", ascending=False)

        fig_imp = px.bar(imp_df, x="Importance", y="Feature", orientation='h')
        st.plotly_chart(fig_imp, use_container_width=True)

        st.success("✅ Model Training Complete!")
