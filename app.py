from pathlib import Path
import streamlit as st
import pandas as pd
import pickle
import joblib
import plotly.express as px
from sklearn.preprocessing import LabelEncoder, StandardScaler

# ---------------------------
# Load Data
# ---------------------------
@st.cache_data
def load_data():
    df = pd.read_csv("cleaned_dataset_CSV")
    return df

# ---------------------------
# Load Model
# ---------------------------
@st.cache_resource
def load_model():
    with open("model_logistic_regression_smote.pkl", "rb") as f:
        model = pickle.load(f)
    return model

# ---------------------------
# Analysis Page
# ---------------------------
def analysis_page(df):
    st.title("Loan Dataset Analysis")

    st.subheader("Dataset Preview")
    st.dataframe(df.head())

    st.subheader("Dataset Info")
    st.write(df.describe())

    st.subheader("Loan Status Distribution")
    st.bar_chart(df["loan_status"].value_counts())

# ---------------------------
# Prediction Page
# ---------------------------
def prediction_page(model, df):
    st.title("Loan Approval Prediction")

    st.write("Enter customer information:")

    person_age = st.number_input("Age", min_value=18, max_value=100, value=25)
    person_gender = st.selectbox("Gender", df["person_gender"].unique())
    person_education = st.selectbox("Education", df["person_education"].unique())
    person_income = st.number_input("Income", min_value=0.0, value=50000.0)
    person_emp_exp = st.number_input("Employment Experience (Years)", min_value=0, value=2)
    person_home_ownership = st.selectbox(
        "Home Ownership", df["person_home_ownership"].unique()
    )

    loan_amnt = st.number_input("Loan Amount", min_value=0.0, value=10000.0)
    loan_intent = st.selectbox("Loan Intent", df["loan_intent"].unique())
    loan_int_rate = st.number_input("Interest Rate", min_value=0.0, value=10.0)
    loan_percent_income = st.number_input(
        "Loan Percent of Income", min_value=0.0, max_value=1.0, value=0.3
    )

    cb_person_cred_hist_length = st.number_input(
        "Credit History Length", min_value=0, value=3
    )
    credit_score = st.number_input("Credit Score", min_value=300, max_value=850, value=650)
    previous_loan_defaults_on_file = st.selectbox(
        "Previous Loan Default", df["previous_loan_defaults_on_file"].unique()
    )

    if st.button("Predict"):
        input_data = pd.DataFrame({
            "person_age": [person_age],
            "person_gender": [person_gender],
            "person_education": [person_education],
            "person_income": [person_income],
            "person_emp_exp": [person_emp_exp],
            "person_home_ownership": [person_home_ownership],
            "loan_amnt": [loan_amnt],
            "loan_intent": [loan_intent],
            "loan_int_rate": [loan_int_rate],
            "loan_percent_income": [loan_percent_income],
            "cb_person_cred_hist_length": [cb_person_cred_hist_length],
            "credit_score": [credit_score],
            "previous_loan_defaults_on_file": [previous_loan_defaults_on_file],
        })

        prediction = model.predict(input_data)[0]

        if prediction == 1:
            st.success("✅ Loan Approved")
        else:
            st.error("❌ Loan Rejected")

# ---------------------------
# Main App
# ---------------------------
def main():
    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Go to", ["Analysis", "Prediction"])

    df = load_data()
    model = load_model()

    if page == "Analysis":
        analysis_page(df)
    else:
        prediction_page(model, df)

if __name__ == "__main__":
    main()