import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
import numpy as np

# Load the dataset
@st.cache_data
def load_data():
    df = pd.read_csv("diabetes_prediction_dataset.csv")
    le = LabelEncoder()
    df['gender'] = le.fit_transform(df['gender'])
    df['smoking_history'] = le.fit_transform(df['smoking_history'])
    return df, le

# Train the model
@st.cache_data
def train_model(df):
    X = df.drop('diabetes', axis=1)
    y = df['diabetes']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.8, random_state=42)
    model = DecisionTreeClassifier(random_state=42)
    model.fit(X_train, y_train)
    accuracy = accuracy_score(y_test, model.predict(X_test))
    return model, accuracy

# Main Streamlit app
st.set_page_config(page_title="Diabetes Prediction App", layout="wide")

# Sidebar
with st.sidebar:
    st.title("About This App")
    st.markdown("""This app predicts whether a person has diabetes based on their health data.
    It uses a machine learning model trained on a diabetes prediction dataset.
    """)
    st.markdown("### Quick Links")
    st.markdown("[Learn More About Diabetes](https://www.halodoc.com/kesehatan/diabetes)")

# Load data and train model
df, le = load_data()
model, accuracy = train_model(df)

st.set_page_config(
    page_title="Diabetes Prediction App",
    page_icon="favicon.ico",  # Replace with your favicon file
    layout="wide"
)

# Header
st.title("Diasense : Diabetes Prediction App")
st.markdown("""### Enter your health data below to get a prediction.""")
#The model has an accuracy of **{:.2f}%** based on the training data.
#""".format(accuracy * 100))

# Input Form with Columns
st.subheader("Your Health Information")
col1, col2 = st.columns(2)

with col1:
    gender = st.selectbox("Gender", ["Male", "Female"], help="Select your gender.")
    age = st.slider("Age", min_value=0, max_value=120, value=25, help="Enter your age.")
    hypertension = st.radio("Do you have hypertension?", ["No", "Yes"], horizontal=True)
    heart_disease = st.radio("Do you have heart disease?", ["No", "Yes"], horizontal=True)

with col2:
    smoking_history = st.selectbox(
        "Smoking history",
        ["Never", "Former", "Ever", "Current", "Not Current"],
        help="Select your smoking history."
    )
    bmi = st.slider("BMI", min_value=0.0, max_value=100.0, value=25.0, help="Enter your BMI.")
    HbA1c_level = st.slider("HbA1c Level", min_value=0.0, max_value=20.0, value=5.5, help="Enter your HbA1c level.")
    blood_glucose_level = st.slider(
        "Blood Glucose Level", min_value=0.0, max_value=500.0, value=100.0, help="Enter your blood glucose level."
    )

# Convert inputs to the model format
if st.button("Predict"):
    with st.spinner("Predicting..."):
        gender_encoded = 0 if gender == "Male" else 1
        hypertension_encoded = 1 if hypertension == "Yes" else 0
        heart_disease_encoded = 1 if heart_disease == "Yes" else 0
        smoking_history_encoded = le.transform([smoking_history.lower()])[0]

        input_data = [
            gender_encoded, age, hypertension_encoded, heart_disease_encoded,
            smoking_history_encoded, bmi, HbA1c_level, blood_glucose_level
        ]

        # Prediction
        prediction = model.predict([input_data])[0]

        st.markdown("---")
        if prediction == 1:
            st.error("\ud83d\ude22 Oh no, you might have diabetes. Please consult a doctor.")
        else:
            st.success("\ud83c\udf89 Great! You are not likely to have diabetes.")

        # Show progress bar for user satisfaction
        st.progress(100)

# Footer
st.markdown("---")
st.markdown("### Disclaimer")
st.markdown("This app is for informational purposes only and not a substitute for professional medical advice.")
