import streamlit as st
import pandas as pd
import numpy as np
import pickle

# Load Model & Scaler
model = pickle.load(open("model.pkl","rb"))
scaler = pickle.load(open("scaler.pkl","rb"))

st.title("🩺 Diabetes Risk Prediction")

# User Inputs
preg = st.number_input("Pregnancies",0,20,0)
glucose = st.number_input("Glucose Level",0,300,120)
bp = st.number_input("Blood Pressure",0,200,80)
skin = st.number_input("Skin Thickness",0,100,20)
insulin = st.number_input("Insulin Level",0,900,80)
bmi = st.number_input("BMI",0.0,70.0,25.0)
dpf = st.number_input("Diabetes Pedigree Function",0.0,3.0,0.5)
age = st.number_input("Age",0,120,30)

if st.button("Predict"):
    data = np.array([[preg,glucose,bp,skin,insulin,bmi,dpf,age]])
    data_scaled = scaler.transform(data)
    result = model.predict(data_scaled)[0]
    st.success("Diabetic" if result else "Not Diabetic")
