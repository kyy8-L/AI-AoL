import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
import numpy as np
import plotly.express as px

# Custom CSS untuk desain yang lebih baik dan smooth
st.markdown("""
    <style>
        /* Overall background and typography */
        body {
            font-family: 'Arial', sans-serif;
            background-color: #f3f3f3;
            transition: background-color 0.3s ease;
        }
        
        /* Title */
        .title {
            text-align: center;
            color: #FF6347;
            font-size: 50px;
            font-weight: bold;
            margin-top: 20px;
            transition: color 0.3s ease;
        }

        /* Subtitle */
        .subheader {
            text-align: center;
            color: #4B0082;
            font-size: 28px;
            margin-top: 20px;
            font-weight: bold;
            transition: color 0.3s ease;
        }

        /* Button Styling */
        .stButton>button {
            background-color: #FF6347;
            color: white;
            font-size: 18px;
            font-weight: bold;
            border-radius: 12px;
            padding: 15px;
            transition: background-color 0.3s ease, transform 0.3s ease;
        }
        
        .stButton>button:hover {
            background-color: #FF4500;
            transform: scale(1.1);
        }
        
        /* Input Fields */
        .stSelectbox>label, .stNumberInput>label {
            color: #4B0082;
            font-size: 16px;
        }

        /* Success/Error message */
        .stSuccess {
            background-color: #32CD32;
            color: white;
            font-weight: bold;
        }
        .stError {
            background-color: #FF6347;
            color: white;
            font-weight: bold;
        }

        /* Styling for links */
        .markdown-text a {
            color: #4B0082;
            text-decoration: none;
        }

        .markdown-text a:hover {
            text-decoration: underline;
        }
        
        /* Animation for loading spinner */
        .stSpinner {
            color: #FF6347;
            font-size: 20px;
        }
    </style>
""", unsafe_allow_html=True)

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
st.markdown('<h1 class="title">Diasense : Diabetes Prediction App</h1>', unsafe_allow_html=True)

# Information link
st.markdown("""
    <div class="markdown-text">
        <p><b>Learn more about diabetes:</b><br>
           <a href="https://www.halodoc.com/kesehatan/diabetes?srsltid=AfmBOorsQ7vTvKtoIXr5Fc1nJ-KugmkCNNgyMdyeWlqZuNX_OoWAig0P" target="_blank">Click here for more information.</a>
        </p>
    </div>
""", unsafe_allow_html=True)

# Load data and train model
df, le = load_data()
model, accuracy = train_model(df)

# User input form with smooth inputs
st.markdown('<h2 class="subheader">Enter your health data:</h2>', unsafe_allow_html=True)

# User input components
gender = st.selectbox("Gender", ["Male", "Female"])
age = st.number_input("Age", min_value=0, max_value=120, value=25, step=1)
hypertension = st.selectbox("Do you have hypertension?", ["No", "Yes"])
heart_disease = st.selectbox("Do you have heart disease?", ["No", "Yes"])
smoking_history = st.selectbox("Smoking history", ["Never", "Former", "Ever", "Current", "Not Current"])
bmi = st.number_input("BMI", min_value=0.0, max_value=100.0, value=25.0, step=0.1)
HbA1c_level = st.number_input("HbA1c Level", min_value=0.0, max_value=20.0, value=5.5, step=0.1)
blood_glucose_level = st.number_input("Blood Glucose Level", min_value=0.0, max_value=500.0, value=100.0, step=1)

# Add progress spinner for prediction
with st.spinner('Calculating your prediction...'):
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

# Visualize Data (Bonus: You can visualize the distribution of diabetes in the dataset)
fig = px.histogram(df, x='age', color='diabetes', title="Age Distribution of Diabetes Cases")
fig.update_layout(bargap=0.2)
st.plotly_chart(fig, use_container_width=True)
