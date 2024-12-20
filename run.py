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
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = DecisionTreeClassifier(random_state=42)
    model.fit(X_train, y_train)
    accuracy = accuracy_score(y_test, model.predict(X_test))
    return model, accuracy

# Main Streamlit app
st.title("Diasense Diabetes Prediction App")

st.markdown("[Diabetes Explanation](https://www.halodoc.com/kesehatan/diabetes?srsltid=AfmBOorsQ7vTvKtoIXr5Fc1nJ-KugmkCNNgyMdyeWlqZuNX_OoWAig0P)")


# Load data and train model
df, le = load_data()
model, accuracy = train_model(df)
#st.write(f"Model accuracy: {accuracy * 100:.2f}%")

# User input form
st.subheader("Enter your health data:")

gender = st.selectbox("Gender", ["Male", "Female"])
age = st.number_input("Age", min_value=0, max_value=120, value=25)
hypertension = st.selectbox("Do you have hypertension?", ["No", "Yes"])
heart_disease = st.selectbox("Do you have heart disease?", ["No", "Yes"])
smoking_history = st.selectbox(
    "Smoking history",
    ["Never", "Former", "Ever", "Current", "Not Current"]
)
bmi = st.number_input("BMI", min_value=0.0, max_value=100.0, value=25.0)
HbA1c_level = st.number_input("HbA1c Level", min_value=0.0, max_value=20.0, value=5.5)
blood_glucose_level = st.number_input("Blood Glucose Level", min_value=0.0, max_value=500.0, value=100.0)

# Convert inputs to the model format
if st.button("Predict"):
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
    if prediction == 1:
        st.error("Oh noooo, you got diabetes :(")
    else:
        st.success("You're safe :)")

