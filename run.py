import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score

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

# Set page configuration
st.set_page_config(
    page_title="Diasense",
    page_icon="favicon.ico",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for Styling
st.markdown("""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Jost&family=Roboto:wght@300;400;700&display=swap');

        body {
            font-family: 'Jost', sans-serif;
        }

        /* Gradient Text for Headers */
        .gradient-text {
            font-weight: bold;
            background: -webkit-linear-gradient(left, #FF6F61, #32CD32, #6A5ACD);
            background: linear-gradient(to right, #FF6F61, #32CD32, #6A5ACD);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            font-size: 3em;
            line-height: 1.2;
            text-align: center;
        }

        /* Animated Gradient Header */
        @keyframes gradientMove {
            0% {background-position: 0%;}
            50% {background-position: 100%;}
            100% {background-position: 0%;}
        }

        .animated-gradient {
            font-weight: bold;
            font-size: 2.5em;
            text-align: center;
            background: linear-gradient(90deg, #FF5722, #FF4081, #6A5ACD, #32CD32);
            background-size: 300% 300%;
            animation: gradientMove 5s infinite;
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }

        /* Custom Subheaders */
        .subheader {
            font-size: 1.5em;
            color: #FF4081;
            margin-top: 10px;
            margin-bottom: 20px;
        }

        /* Buttons Styling */
        .stButton button {
            background: linear-gradient(to right, #FF4081, #6A5ACD);
            color: white;
            font-weight: bold;
            font-size: 16px;
            border: none;
            border-radius: 5px;
            padding: 12px 30px;
            transition: all 0.3s ease;
        }

        .stButton button:hover {
            background: linear-gradient(to right, #6A5ACD, #32CD32);
            transform: scale(1.05);
        }

        /* Green Color for Sliders */
        .stSlider > div[data-baseweb="slider"] > div {
            background: linear-gradient(to right, #32CD32, #98FB98);
        }
        .stSlider > div[data-baseweb="slider"] > div > div {
            color: white !important;
        }

        /* Sidebar Styling */
        .sidebar-header {
            font-size: 1.3em;
            color: #FF6F61;
            font-weight: bold;
            text-align: center;
        }
    </style>
""", unsafe_allow_html=True)


# Sidebar Navigation
st.sidebar.title("Navigation")
pages = ["Home", "Explore Data", "About"]
selected_page = st.sidebar.radio("Go to", pages)

# Load data and model
df, le = load_data()
model, accuracy = train_model(df)

# Home Page
if selected_page == "Home":
    st.markdown("<h1 class='gradient-text'>Welcome to Diasense</h1>", unsafe_allow_html=True)
    st.markdown("<h3 class='subheader'>Your go-to app for diabetes prediction and insights</h3>", unsafe_allow_html=True)

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

    if st.button("Predict"):
        with st.spinner("Analyzing your data..."):
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
                st.error("😢 Oh no, you might have diabetes. Please consult a doctor.")
            else:
                st.success("🎉 Great! You are not likely to have diabetes.")
                st.balloons()

# Explore Data Page
elif selected_page == "Explore Data":
    st.markdown("<h1 class='gradient-text'>Explore Diabetes Dataset</h1>", unsafe_allow_html=True)
    st.markdown("Here’s a sneak peek into the data used for prediction.")
    st.dataframe(df.head())

    st.subheader("Gender Distribution")
    gender_counts = df['gender'].value_counts()
    st.bar_chart(gender_counts)

    st.subheader("Smoking History")
    smoking_counts = df['smoking_history'].value_counts()
    st.bar_chart(smoking_counts)

    st.subheader("Diabetes Outcome")
    diabetes_counts = df['diabetes'].value_counts()
    diabetes_counts = diabetes_counts.sort_values()  # Sort for better visualization
    st.bar_chart(diabetes_counts)

# About Page
elif selected_page == "About":
    st.markdown("<h1 class='gradient-text'>About Diasense</h1>", unsafe_allow_html=True)
    st.markdown("""
    Diasense is a simple app that helps predict the likelihood of diabetes using user input data.
    
    - **Built with:** Python, Streamlit, and Scikit-learn
    - **Authors:** Jip Tyrone, Emanuel, Sofyan
    """, unsafe_allow_html=True)
    
    st.image("https://img.freepik.com/free-vector/healthy-lifestyle-concept-illustration_114360-11136.jpg", width=400)

    st.markdown("### Disclaimer")
    st.markdown("This app is for informational purposes only and should not be used as a substitute for medical advice.")

# Footer
st.markdown("---")
st.markdown("© 2024 Diasense | Built with 💖 and Python")
