import joblib
import numpy as np
import streamlit as st

# Load model and scaler
model = joblib.load('model.pkl')
scaler = joblib.load('scaler.pkl')

# Function to preprocess input data
def preprocess_input(input_data):
    return scaler.transform(input_data)

# Function to make a prediction
def make_prediction(input_data):
    processed_data = preprocess_input(input_data)
    prediction = model.predict(processed_data)
    return prediction

# Streamlit UI
st.title("Iris Flower Prediction")

# Input fields for the features
sepal_length = st.number_input("Sepal Length")
sepal_width = st.number_input("Sepal Width")
petal_length = st.number_input("Petal Length")
petal_width = st.number_input("Petal Width")

input_data = np.array([[sepal_length, sepal_width, petal_length, petal_width]])

if st.button("Predict"):
    prediction = make_prediction(input_data)
    print(f"Predicted Class: {prediction[0]}")
    # Display the prediction    
    st.write(f"Predicted Class: {prediction[0]}")
