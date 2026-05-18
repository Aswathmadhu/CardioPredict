from matplotlib import axis
import streamlit as st
import pickle
from PIL import Image
import pandas as pd
import numpy as np

# Variable Definitions 

# Demographics age (int): Age of the individual (25-90 years) gender (str): Gender of the individual (Male, Female) region (str): Living area (Urban, Rural) income_level (str): Socioeconomic status (Low, Middle, High)
# Clinical Risk Factors hypertension (int): High blood pressure (1 = Yes, 0 = No) diabetes (int): Diagnosed diabetes (1 = Yes, 0 = No) cholesterol_level (int): Total cholesterol level (mg/dL) obesity (int): BMI > 30 (1 = Yes, 0 = No) waist_circumference (int): Waist circumference in cm family_history (int): Family history of heart disease (1 = Yes, 0 = No)
# Lifestyle & Behavioral Factors smoking_status (str): Smoking habit (Never, Past, Current) alcohol_consumption (str): Alcohol intake (None, Moderate, High) physical_activity (str): Physical activity level (Low, Moderate, High) dietary_habits (str): Diet quality (Healthy, Unhealthy)
# Environmental & Social Factors air_pollution_exposure (str): Pollution exposure (Low, Moderate, High) stress_level (str): Stress level (Low, Moderate, High) sleep_hours (float): Average sleep hours per night (3-9 hours)
# Medical Screening & Health System Factors blood_pressure_systolic (int): Systolic BP (mmHg) blood_pressure_diastolic (int): Diastolic BP (mmHg) fasting_blood_sugar (int): Blood sugar level (mg/dL) cholesterol_hdl (int): HDL cholesterol level (mg/dL) cholesterol_ldl (int): LDL cholesterol level (mg/dL) triglycerides (int): Triglyceride level (mg/dL) EKG_results (str): Electrocardiogram result (Normal, Abnormal) previous_heart_disease (int): Prior heart disease (1 = Yes, 0 = No) medication_usage (int): Currently taking heart-related medications (1 = Yes, 0 = No) participated_in_free_screening (int): Attended Indonesia’s free health screening program (1 = Yes, 0 = No)
# Target Variable heart_attack (int): Heart attack occurrence (1 = Yes, 0 = No)


def main():
    st.title(':yellow[HEART ATTACK PREDICTION]')

    image = Image.open('image.png')
    st.image([image])

    age = st.text_input('Age')
    hypertension = st.selectbox('Hypertension',('','yes','no'))
    diabetes = st.selectbox('Diabetes',('','yes','no'))
    cholesterol_level = st.text_input('Cholesterol level')
    obesity = st.selectbox('Obesity (BMI > 30)',('','yes','no'))
    fasting_blood_sugar = st.text_input('Fasting blood sugar')
    previous_heart_disease = st.selectbox('Previous heart disease',('','yes','no'))
    smoking_status = st.selectbox('Smoking status',('','Never','Past','Current'))




    model = pickle.load(open('model.sav','rb'))
    ohe_smoking = pickle.load(open('ohe_smoking.sav','rb'))
    scaler = pickle.load(open('scaler.sav','rb'))

    pred = st.button('Predict')


    if pred:
        data = [[age,hypertension,diabetes,cholesterol_level,obesity,fasting_blood_sugar,previous_heart_disease]]
        data = pd.DataFrame(data=data,columns=('age','hypertension','diabetes','cholesterol_level','obesity','fasting_blood_sugar','previous_heart_disease'))

        smoking1 = ohe_smoking.transform([[smoking_status]])
        smoking = pd.DataFrame(smoking1,columns=ohe_smoking.get_feature_names_out())
        data = pd.concat((data,smoking),axis=1)
        data['age'] = data['age'].astype(float)
        data['cholesterol_level'] = data['cholesterol_level'].astype(float)
        data['fasting_blood_sugar'] = data['fasting_blood_sugar'].astype(float)
        data['hypertension'] = data['hypertension'].replace({'yes': 1,'no': 0}).astype(float)
        data['diabetes'] = data['diabetes'].replace({'yes': 1,'no': 0}).astype(float)
        data['obesity'] = data['obesity'].replace({'yes': 1,'no': 0}).astype(float)
        data['previous_heart_disease'] = data['previous_heart_disease'].replace({'yes': 1,'no': 0}).astype(float)
        data = scaler.transform(data)
        print(data)
        prediction = model.predict(data)
        print(prediction)
         
        if prediction == 1:
             st.write('Heart attack possibility')
        else :
             st.write('No Heart attack possibility ')

main()