import streamlit as st
import pandas as pd
import numpy as np
import pickle

# Helper functions for health metrics calculations
def convert_weight(weight, unit):
    if unit in ["lbs", "pounds"]:
        return weight * 0.45359237
    return weight

def convert_height(height, unit):
    if unit == "cm":
        return height / 100
    return height

def convert_waist_hip(measurement, unit):
    if unit == "inches":
        return measurement * 2.54
    return measurement

def calculate_bmi(weight, height):
    bmi = weight / (height * height)
    return round(bmi, 2)

def interpret_bmi(bmi, gender):
    if bmi < 18.5:
        return "Underweight"
    elif bmi < 24.9:
        return "Normal weight"
    elif bmi < 29.9:
        return "Overweight"
    else:
        return "Obese"

def calculate_bmr(weight, height, age, gender):
    if gender == "Male":
        bmr = 88.362 + (13.397 * weight) + (4.799 * height * 100) - (5.677 * age)
    else:
        bmr = 447.593 + (9.247 * weight) + (3.098 * height * 100) - (4.330 * age)
    return round(bmr)

def calculate_tdee(bmr, activity_level):
    activity_multipliers = {
        "Sedentary": 1.2,
        "Lightly Active": 1.375,
        "Moderately Active": 1.55,
        "Very Active": 1.725,
        "Extra Active": 1.9
    }
    tdee = bmr * activity_multipliers[activity_level]
    return round(tdee)

def calculate_body_fat(weight, height, age, gender):
    bmi = calculate_bmi(weight, height)
    if gender == "Male":
        body_fat = (1.20 * bmi) + (0.23 * age) - 16.2
    else:
        body_fat = (1.20 * bmi) + (0.23 * age) - 5.4
    return round(body_fat, 2)

def interpret_body_fat(bf_percentage, gender):
    if gender == "Male":
        if bf_percentage < 6:
            return "Essential Fat"
        elif bf_percentage < 14:
            return "Athletes"
        elif bf_percentage < 18:
            return "Fitness"
        elif bf_percentage < 25:
            return "Average"
        else:
            return "Obese"
    else:
        if bf_percentage < 13:
            return "Essential Fat"
        elif bf_percentage < 21:
            return "Athletes"
        elif bf_percentage < 25:
            return "Fitness"
        elif bf_percentage < 32:
            return "Average"
        else:
            return "Obese"

def calculate_lbm(weight, body_fat_percentage):
    lbm = weight * (1 - (body_fat_percentage / 100))
    return round(lbm, 2)

def calculate_wh_ratio(waist, hip):
    return round(waist / hip, 3)

def interpret_wh_ratio(whr, gender):
    if gender == "Male":
        if whr < 0.9:
            return "Low Risk"
        elif whr < 1.0:
            return "Moderate Risk"
        else:
            return "High Risk"
    else:
        if whr < 0.8:
            return "Low Risk"
        elif whr < 0.85:
            return "Moderate Risk"
        else:
            return "High Risk"

# Load the saved models
diabetes_model = pickle.load(open(r'D:\SACAIM\Digital Health Assistant\Saved Models\diabetesPrediction.sav', 'rb'))
heart_model = pickle.load(open(r'D:\SACAIM\Digital Health Assistant\Saved Models\heartDiseasePrediction.sav', 'rb'))
alzheimers_model = pickle.load(open(r'D:\SACAIM\Digital Health Assistant\Saved Models\alzheimersPrediction.sav', 'rb'))
parkinsons_model = pickle.load(open(r'D:\SACAIM\Digital Health Assistant\Saved Models\parkinsonsPrediction.sav', 'rb'))
kidney_model = pickle.load(open(r'D:\SACAIM\Digital Health Assistant\Saved Models\kidneyDiseasePrediction.sav', 'rb'))
breast_cancer_model = pickle.load(open(r'D:\SACAIM\Digital Health Assistant\Saved Models\breastCancerPrediction.sav', 'rb'))
cervical_cancer_model = pickle.load(open(r'D:\SACAIM\Digital Health Assistant\Saved Models\cervicalCancerPrediction.sav', 'rb'))

# Load symptom checker model and encoders
with open(r"D:\SACAIM\Digital Health Assistant\Saved Models\symptomChecker.sav", 'rb') as f:
    model_dict = pickle.load(f)

model = model_dict['model']
le_fever = model_dict['fever_encoder']
le_cough = model_dict['cough_encoder']
le_fatigue = model_dict['fatigue_encoder'] 
le_breathing = model_dict['breathing_encoder']
le_gender = model_dict['gender_encoder']
le_bp = model_dict['bp_encoder']
le_cholesterol = model_dict['cholesterol_encoder']
le_disease = model_dict['disease_encoder']

# Create tabs for different predictions
tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8, tab9 = st.tabs(["Diabetes Prediction", "Heart Disease Prediction", "Alzheimer's Prediction", 
                                             "Parkinson's Prediction", "Kidney Disease Prediction", "Breast Cancer Prediction",
                                             "Cervical Cancer Prediction", "Symptom Checker", "Health Metrics Calculator"])

with tab1:
    st.title('Diabetes Prediction System')
    st.write('Enter patient information to predict diabetes risk')

    # Create input fields for diabetes features
    pregnancies = st.number_input('Number of Pregnancies', min_value=0, max_value=20, value=0, key='diabetes_pregnancies')
    glucose = st.number_input('Glucose Level', min_value=0, max_value=300, value=0, key='diabetes_glucose')
    blood_pressure = st.number_input('Blood Pressure', min_value=0, max_value=200, value=0, key='diabetes_bp')
    skin_thickness = st.number_input('Skin Thickness', min_value=0, max_value=100, value=0, key='diabetes_skin')
    insulin = st.number_input('Insulin Level', min_value=0, max_value=1000, value=0, key='diabetes_insulin')
    bmi = st.number_input('BMI', min_value=0.0, max_value=70.0, value=0.0, key='diabetes_bmi')
    diabetes_pedigree = st.number_input('Diabetes Pedigree Function', min_value=0.0, max_value=3.0, value=0.0, key='diabetes_pedigree')
    age_diabetes = st.number_input('Age', min_value=0, max_value=120, value=0, key='diabetes_age')

    # Create a button for diabetes prediction
    if st.button('Predict Diabetes Risk'):
        # Create input data frame
        input_data = pd.DataFrame([[pregnancies, glucose, blood_pressure, skin_thickness, 
                                insulin, bmi, diabetes_pedigree, age_diabetes]], 
                                columns=['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness',
                                        'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age'])
        
        # Make prediction
        prediction = diabetes_model.predict(input_data)
        prediction_proba = diabetes_model.predict_proba(input_data)
        
        # Display results
        st.subheader('Prediction Result')
        if prediction[0] == 0:
            st.success('The patient is predicted to NOT have diabetes')
        else:
            st.error('The patient is predicted to have diabetes')
            
        st.write('Prediction Probability:')
        st.write(f'No Diabetes: {prediction_proba[0][0]:.2%}')
        st.write(f'Diabetes: {prediction_proba[0][1]:.2%}')

with tab2:
    st.title('Heart Disease Prediction System')
    st.write('Enter patient information to predict heart disease risk')

    # Create input fields for heart disease features
    age = st.number_input('Age', min_value=0, max_value=120, value=0, key='heart_age')
    sex = st.selectbox('Sex', ['Male', 'Female'], key='heart_sex')
    cp = st.selectbox('Chest Pain Type', ['Typical Angina', 'Atypical Angina', 'Non-anginal Pain', 'Asymptomatic'], key='heart_cp')
    trestbps = st.number_input('Resting Blood Pressure', min_value=0, max_value=300, value=0, key='heart_trestbps')
    chol = st.number_input('Cholesterol', min_value=0, max_value=600, value=0, key='heart_chol')
    fbs = st.selectbox('Fasting Blood Sugar > 120 mg/dl', ['Yes', 'No'], key='heart_fbs')
    restecg = st.selectbox('Resting ECG Results', ['Normal', 'ST-T Wave Abnormality', 'Left Ventricular Hypertrophy'], key='heart_restecg')
    thalach = st.number_input('Maximum Heart Rate', min_value=0, max_value=300, value=0, key='heart_thalach')
    exang = st.selectbox('Exercise Induced Angina', ['Yes', 'No'], key='heart_exang')
    oldpeak = st.number_input('ST Depression', min_value=0.0, max_value=10.0, value=0.0, key='heart_oldpeak')
    slope = st.selectbox('Slope of Peak Exercise ST Segment', ['Upsloping', 'Flat', 'Downsloping'], key='heart_slope')
    ca = st.number_input('Number of Major Vessels', min_value=0, max_value=4, value=0, key='heart_ca')
    thal = st.selectbox('Thalassemia', ['Normal', 'Fixed Defect', 'Reversible Defect'], key='heart_thal')

    # Create a button for heart disease prediction
    if st.button('Predict Heart Disease Risk'):
        # Convert categorical inputs to numerical
        sex_encoded = 1 if sex == 'Male' else 0
        cp_encoded = ['Typical Angina', 'Atypical Angina', 'Non-anginal Pain', 'Asymptomatic'].index(cp)
        fbs_encoded = 1 if fbs == 'Yes' else 0
        restecg_encoded = ['Normal', 'ST-T Wave Abnormality', 'Left Ventricular Hypertrophy'].index(restecg)
        exang_encoded = 1 if exang == 'Yes' else 0
        slope_encoded = ['Upsloping', 'Flat', 'Downsloping'].index(slope)
        thal_encoded = ['Normal', 'Fixed Defect', 'Reversible Defect'].index(thal) + 1

        # Create input data frame
        input_data = pd.DataFrame([[age, sex_encoded, cp_encoded, trestbps, chol, fbs_encoded, 
                                  restecg_encoded, thalach, exang_encoded, oldpeak, 
                                  slope_encoded, ca, thal_encoded]], 
                                columns=['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg',
                                       'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal'])
        
        # Make prediction
        prediction = heart_model.predict(input_data)
        prediction_proba = heart_model.predict_proba(input_data)
        
        # Display results
        st.subheader('Prediction Result')
        if prediction[0] == 0:
            st.success('The patient is predicted to NOT have heart disease')
        else:
            st.error('The patient is predicted to have heart disease')
            
        st.write('Prediction Probability:')
        st.write(f'No Heart Disease: {prediction_proba[0][0]:.2%}')
        st.write(f'Heart Disease: {prediction_proba[0][1]:.2%}')

with tab3:
    st.title("Alzheimer's Disease Prediction System")
    st.write('Enter patient information to predict Alzheimer\'s risk')

    # Create input fields for Alzheimer's features
    age_alz = st.number_input('Age', min_value=0, max_value=120, value=0, key='alz_age')
    gender = st.selectbox('Gender', ['Female', 'Male'], key='alz_gender')
    ethnicity = st.selectbox('Ethnicity', ['Caucasian', 'African American', 'Asian', 'Hispanic'], key='alz_ethnicity')
    education = st.selectbox('Education Level', ['Less than High School', 'High School', 'College', 'Graduate'], key='alz_education')
    bmi_alz = st.number_input('BMI', min_value=0.0, max_value=50.0, value=0.0, key='alz_bmi')
    alcohol = st.number_input('Alcohol Consumption (drinks per week)', min_value=0.0, max_value=50.0, value=0.0, key='alz_alcohol')
    physical_activity = st.number_input('Physical Activity (hours per week)', min_value=0.0, max_value=40.0, value=0.0, key='alz_activity')
    smoking = st.selectbox('Smoking Status', ['No', 'Yes'], key='alz_smoking')
    
    # Additional medical history
    family_history = st.selectbox('Family History of Alzheimer\'s', ['No', 'Yes'], key='alz_family')
    cardiovascular = st.selectbox('Cardiovascular Disease', ['No', 'Yes'], key='alz_cardio')
    diabetes_alz = st.selectbox('Diabetes', ['No', 'Yes'], key='alz_diabetes')
    depression = st.selectbox('Depression', ['No', 'Yes'], key='alz_depression')
    head_injury = st.selectbox('History of Head Injury', ['No', 'Yes'], key='alz_injury')
    hypertension = st.selectbox('Hypertension', ['No', 'Yes'], key='alz_hypertension')

    # Clinical measurements
    systolic_bp = st.number_input('Systolic Blood Pressure', min_value=0, max_value=250, value=0, key='alz_systolic')
    diastolic_bp = st.number_input('Diastolic Blood Pressure', min_value=0, max_value=150, value=0, key='alz_diastolic')
    
    # Cognitive assessments
    mmse = st.number_input('MMSE Score', min_value=0.0, max_value=30.0, value=0.0, key='alz_mmse')
    memory_complaints = st.selectbox('Memory Complaints', ['No', 'Yes'], key='alz_memory')
    confusion = st.selectbox('Confusion', ['No', 'Yes'], key='alz_confusion')
    disorientation = st.selectbox('Disorientation', ['No', 'Yes'], key='alz_disorientation')

    if st.button('Predict Alzheimer\'s Risk'):
        # Convert categorical inputs to numerical
        gender_encoded = 1 if gender == 'Male' else 0
        ethnicity_encoded = ['Caucasian', 'African American', 'Asian', 'Hispanic'].index(ethnicity)
        education_encoded = ['Less than High School', 'High School', 'College', 'Graduate'].index(education)
        smoking_encoded = 1 if smoking == 'Yes' else 0
        family_history_encoded = 1 if family_history == 'Yes' else 0
        cardiovascular_encoded = 1 if cardiovascular == 'Yes' else 0
        diabetes_encoded = 1 if diabetes_alz == 'Yes' else 0
        depression_encoded = 1 if depression == 'Yes' else 0
        head_injury_encoded = 1 if head_injury == 'Yes' else 0
        hypertension_encoded = 1 if hypertension == 'Yes' else 0
        memory_complaints_encoded = 1 if memory_complaints == 'Yes' else 0
        confusion_encoded = 1 if confusion == 'Yes' else 0
        disorientation_encoded = 1 if disorientation == 'Yes' else 0

        # Create input data frame with all required features
        input_data = pd.DataFrame([[age_alz, gender_encoded, ethnicity_encoded, education_encoded, 
                                  bmi_alz, smoking_encoded, alcohol, physical_activity, 
                                  family_history_encoded, cardiovascular_encoded, diabetes_encoded,
                                  depression_encoded, head_injury_encoded, hypertension_encoded,
                                  systolic_bp, diastolic_bp, mmse, memory_complaints_encoded,
                                  confusion_encoded, disorientation_encoded]], 
                                columns=['Age', 'Gender', 'Ethnicity', 'EducationLevel', 'BMI',
                                       'Smoking', 'AlcoholConsumption', 'PhysicalActivity',
                                       'FamilyHistoryAlzheimers', 'CardiovascularDisease', 'Diabetes',
                                       'Depression', 'HeadInjury', 'Hypertension', 'SystolicBP',
                                       'DiastolicBP', 'MMSE', 'MemoryComplaints', 'Confusion',
                                       'Disorientation'])

        # Make prediction
        prediction = alzheimers_model.predict(input_data)
        prediction_proba = alzheimers_model.predict_proba(input_data)

        # Display results
        st.subheader('Prediction Result')
        if prediction[0] == 0:
            st.success('The patient is predicted to NOT have Alzheimer\'s Disease')
        else:
            st.error('The patient is predicted to have Alzheimer\'s Disease')
            
        st.write('Prediction Probability:')
        st.write(f'No Alzheimer\'s Disease: {prediction_proba[0][0]:.2%}')
        st.write(f'Alzheimer\'s Disease: {prediction_proba[0][1]:.2%}')

with tab4:
    st.title("Parkinson's Disease Prediction System")
    st.write('Enter patient information to predict Parkinson\'s risk')

    # Create input fields for Parkinson's features
    fo = st.number_input('MDVP:Fo(Hz)', min_value=0.0, max_value=300.0, value=0.0, key='park_fo')
    fhi = st.number_input('MDVP:Fhi(Hz)', min_value=0.0, max_value=300.0, value=0.0, key='park_fhi') 
    flo = st.number_input('MDVP:Flo(Hz)', min_value=0.0, max_value=300.0, value=0.0, key='park_flo')
    jitter_percent = st.number_input('MDVP:Jitter(%)', min_value=0.0, max_value=1.0, value=0.0, key='park_jitter_pct')
    jitter_abs = st.number_input('MDVP:Jitter(Abs)', min_value=0.0, max_value=1.0, value=0.0, key='park_jitter_abs')
    rap = st.number_input('MDVP:RAP', min_value=0.0, max_value=1.0, value=0.0, key='park_rap')
    ppq = st.number_input('MDVP:PPQ', min_value=0.0, max_value=1.0, value=0.0, key='park_ppq')
    ddp = st.number_input('Jitter:DDP', min_value=0.0, max_value=1.0, value=0.0, key='park_ddp')
    shimmer = st.number_input('MDVP:Shimmer', min_value=0.0, max_value=1.0, value=0.0, key='park_shimmer')
    shimmer_db = st.number_input('MDVP:Shimmer(dB)', min_value=0.0, max_value=1.0, value=0.0, key='park_shimmer_db')
    apq3 = st.number_input('Shimmer:APQ3', min_value=0.0, max_value=1.0, value=0.0, key='park_apq3')
    apq5 = st.number_input('Shimmer:APQ5', min_value=0.0, max_value=1.0, value=0.0, key='park_apq5')
    apq = st.number_input('MDVP:APQ', min_value=0.0, max_value=1.0, value=0.0, key='park_apq')
    dda = st.number_input('Shimmer:DDA', min_value=0.0, max_value=1.0, value=0.0, key='park_dda')
    nhr = st.number_input('NHR', min_value=0.0, max_value=1.0, value=0.0, key='park_nhr')
    hnr = st.number_input('HNR', min_value=0.0, max_value=50.0, value=0.0, key='park_hnr')
    rpde = st.number_input('RPDE', min_value=0.0, max_value=1.0, value=0.0, key='park_rpde')
    dfa = st.number_input('DFA', min_value=0.0, max_value=1.0, value=0.0, key='park_dfa')
    spread1 = st.number_input('spread1', min_value=-10.0, max_value=10.0, value=0.0, key='park_spread1')
    spread2 = st.number_input('spread2', min_value=0.0, max_value=1.0, value=0.0, key='park_spread2')
    d2 = st.number_input('D2', min_value=0.0, max_value=10.0, value=0.0, key='park_d2')
    ppe = st.number_input('PPE', min_value=0.0, max_value=1.0, value=0.0, key='park_ppe')

    if st.button('Predict Parkinson\'s Risk'):
        # Create input data frame
        input_data = pd.DataFrame([[fo, fhi, flo, jitter_percent, jitter_abs, rap, ppq, ddp,
                                  shimmer, shimmer_db, apq3, apq5, apq, dda, nhr, hnr,
                                  rpde, dfa, spread1, spread2, d2, ppe]], 
                                columns=['MDVP:Fo(Hz)', 'MDVP:Fhi(Hz)', 'MDVP:Flo(Hz)', 'MDVP:Jitter(%)',
                                       'MDVP:Jitter(Abs)', 'MDVP:RAP', 'MDVP:PPQ', 'Jitter:DDP',
                                       'MDVP:Shimmer', 'MDVP:Shimmer(dB)', 'Shimmer:APQ3', 'Shimmer:APQ5',
                                       'MDVP:APQ', 'Shimmer:DDA', 'NHR', 'HNR', 'RPDE', 'DFA',
                                       'spread1', 'spread2', 'D2', 'PPE'])

        # Make prediction
        prediction = parkinsons_model.predict(input_data)
        prediction_proba = parkinsons_model.predict_proba(input_data)

        # Display results
        st.subheader('Prediction Result')
        if prediction[0] == 0:
            st.success('The patient is predicted to NOT have Parkinson\'s Disease')
        else:
            st.error('The patient is predicted to have Parkinson\'s Disease')
            
        st.write('Prediction Probability:')
        st.write(f'No Parkinson\'s Disease: {prediction_proba[0][0]:.2%}')
        st.write(f'Parkinson\'s Disease: {prediction_proba[0][1]:.2%}')

with tab5:
    st.title("Kidney Disease Prediction System")
    st.write('Enter patient information to predict kidney disease risk')

    # Create input fields for Kidney Disease features
    age_kidney = st.number_input('Age', min_value=0, max_value=120, value=0, key='kidney_age')
    bp = st.number_input('Blood Pressure', min_value=0, max_value=200, value=0, key='kidney_bp')
    sg = st.number_input('Specific Gravity', min_value=1.0, max_value=1.025, value=1.0, key='kidney_sg')
    al = st.number_input('Albumin', min_value=0, max_value=5, value=0, key='kidney_al')
    su = st.number_input('Sugar', min_value=0, max_value=5, value=0, key='kidney_su')
    rbc = st.selectbox('Red Blood Cells', ['normal', 'abnormal'], key='kidney_rbc')
    pc = st.selectbox('Pus Cell', ['normal', 'abnormal'], key='kidney_pc')
    pcc = st.selectbox('Pus Cell Clumps', ['present', 'notpresent'], key='kidney_pcc')
    ba = st.selectbox('Bacteria', ['present', 'notpresent'], key='kidney_ba')
    bgr = st.number_input('Blood Glucose Random', min_value=0, max_value=500, value=0, key='kidney_bgr')
    bu = st.number_input('Blood Urea', min_value=0, max_value=200, value=0, key='kidney_bu')
    sc = st.number_input('Serum Creatinine', min_value=0.0, max_value=15.0, value=0.0, key='kidney_sc')
    sod = st.number_input('Sodium', min_value=0, max_value=200, value=0, key='kidney_sod')
    pot = st.number_input('Potassium', min_value=0.0, max_value=10.0, value=0.0, key='kidney_pot')
    hemo = st.number_input('Hemoglobin', min_value=0.0, max_value=20.0, value=0.0, key='kidney_hemo')
    pcv = st.number_input('Packed Cell Volume', min_value=0, max_value=60, value=0, key='kidney_pcv')
    wc = st.number_input('White Blood Cell Count', min_value=0, max_value=20000, value=0, key='kidney_wc')
    rc = st.number_input('Red Blood Cell Count', min_value=0.0, max_value=8.0, value=0.0, key='kidney_rc')
    htn = st.selectbox('Hypertension', ['yes', 'no'], key='kidney_htn')
    dm = st.selectbox('Diabetes Mellitus', ['yes', 'no'], key='kidney_dm')
    cad = st.selectbox('Coronary Artery Disease', ['yes', 'no'], key='kidney_cad')
    appet = st.selectbox('Appetite', ['good', 'poor'], key='kidney_appet')
    pe = st.selectbox('Pedal Edema', ['yes', 'no'], key='kidney_pe')
    ane = st.selectbox('Anemia', ['yes', 'no'], key='kidney_ane')

    if st.button('Predict Kidney Disease Risk'):
        # Create input data frame
        input_data = pd.DataFrame([[age_kidney, bp, sg, al, su, rbc, pc, pcc, ba, bgr, bu, sc, sod, pot,
                                  hemo, pcv, wc, rc, htn, dm, cad, appet, pe, ane]], 
                                columns=['age', 'bp', 'sg', 'al', 'su', 'rbc', 'pc', 'pcc', 'ba', 'bgr',
                                       'bu', 'sc', 'sod', 'pot', 'hemo', 'pcv', 'wc', 'rc', 'htn', 'dm',
                                       'cad', 'appet', 'pe', 'ane'])

        # Make prediction using the pipeline
        prediction = kidney_model['pipeline'].predict(input_data)
        prediction_proba = kidney_model['pipeline'].predict_proba(input_data)

        # Display results
        st.subheader('Prediction Result')
        if prediction[0] == 0:
            st.success('The patient is predicted to NOT have Chronic Kidney Disease')
        else:
            st.error('The patient is predicted to have Chronic Kidney Disease')
            
        st.write('Prediction Probability:')
        st.write(f'No Chronic Kidney Disease: {prediction_proba[0][0]:.2%}')
        st.write(f'Chronic Kidney Disease: {prediction_proba[0][1]:.2%}')

with tab6:
    st.title("Breast Cancer Prediction System")
    st.write('Enter patient information to predict breast cancer risk')

    # Create input fields for breast cancer features
    radius_mean = st.number_input('Mean Radius', min_value=0.0, max_value=50.0, value=0.0, key='bc_radius')
    texture_mean = st.number_input('Mean Texture', min_value=0.0, max_value=50.0, value=0.0, key='bc_texture')
    perimeter_mean = st.number_input('Mean Perimeter', min_value=0.0, max_value=200.0, value=0.0, key='bc_perimeter')
    area_mean = st.number_input('Mean Area', min_value=0.0, max_value=2500.0, value=0.0, key='bc_area')
    smoothness_mean = st.number_input('Mean Smoothness', min_value=0.0, max_value=1.0, value=0.0, key='bc_smoothness')
    compactness_mean = st.number_input('Mean Compactness', min_value=0.0, max_value=1.0, value=0.0, key='bc_compactness')
    concavity_mean = st.number_input('Mean Concavity', min_value=0.0, max_value=1.0, value=0.0, key='bc_concavity')
    concave_points_mean = st.number_input('Mean Concave Points', min_value=0.0, max_value=1.0, value=0.0, key='bc_concave_points')
    symmetry_mean = st.number_input('Mean Symmetry', min_value=0.0, max_value=1.0, value=0.0, key='bc_symmetry')
    fractal_dimension_mean = st.number_input('Mean Fractal Dimension', min_value=0.0, max_value=1.0, value=0.0, key='bc_fractal')

    if st.button('Predict Breast Cancer Risk'):
        # Create input data frame
        input_data = pd.DataFrame([[radius_mean, texture_mean, perimeter_mean, area_mean,
                                  smoothness_mean, compactness_mean, concavity_mean,
                                  concave_points_mean, symmetry_mean, fractal_dimension_mean]], 
                                columns=['radius_mean', 'texture_mean', 'perimeter_mean', 'area_mean',
                                       'smoothness_mean', 'compactness_mean', 'concavity_mean',
                                       'concave points_mean', 'symmetry_mean', 'fractal_dimension_mean'])

        # Make prediction
        prediction = breast_cancer_model.predict(input_data)
        prediction_proba = breast_cancer_model.predict_proba(input_data)

        # Display results
        st.subheader('Prediction Result')
        if prediction[0] == 0:
            st.success('The tumor is predicted to be BENIGN')
        else:
            st.error('The tumor is predicted to be MALIGNANT')
            
        st.write('Prediction Probability:')
        st.write(f'Benign: {prediction_proba[0][0]:.2%}')
        st.write(f'Malignant: {prediction_proba[0][1]:.2%}')

with tab7:
    st.title("Cervical Cancer Prediction System")
    st.write('Enter patient information to predict cervical cancer risk')

    # Create input fields for cervical cancer features
    age_cervical = st.number_input('Age', min_value=0, max_value=120, value=0, key='cervical_age')
    sexual_partners = st.number_input('Number of Sexual Partners', min_value=0, max_value=50, value=0, key='cervical_partners')
    first_intercourse = st.number_input('Age at First Sexual Intercourse', min_value=0, max_value=50, value=0, key='cervical_intercourse')
    num_pregnancies = st.number_input('Number of Pregnancies', min_value=0, max_value=20, value=0, key='cervical_pregnancies')
    smokes = st.selectbox('Smoking Status', ['No', 'Yes'], key='cervical_smokes_status')
    smokes_years = st.number_input('Years of Smoking', min_value=0, max_value=50, value=0, key='cervical_smokes_years')
    smokes_packs = st.number_input('Packs per Year', min_value=0.0, max_value=50.0, value=0.0, key='cervical_packs')
    hormonal_contra = st.selectbox('Hormonal Contraceptives', ['No', 'Yes'], key='cervical_contra')
    sexual_partners = st.number_input('Number of Sexual Partners', min_value=0, max_value=50, value=0)
    first_intercourse = st.number_input('Age at First Sexual Intercourse', min_value=0, max_value=50, value=0)
    num_pregnancies = st.number_input('Number of Pregnancies', min_value=0, max_value=20, value=0)
    smokes = st.selectbox('Smoking Status', ['No', 'Yes'], key='cervical_smokes')
    smokes_years = st.number_input('Years of Smoking', min_value=0, max_value=50, value=0)
    smokes_packs = st.number_input('Packs per Year', min_value=0.0, max_value=50.0, value=0.0)
    hormonal_contra = st.selectbox('Hormonal Contraceptives', ['No', 'Yes'])
    hormonal_years = st.number_input('Years on Hormonal Contraceptives', min_value=0, max_value=50, value=0)
    iud = st.selectbox('IUD', ['No', 'Yes'])
    iud_years = st.number_input('Years with IUD', min_value=0, max_value=50, value=0)
    stds = st.selectbox('History of STDs', ['No', 'Yes'])
    stds_number = st.number_input('Number of STD Diagnoses', min_value=0, max_value=10, value=0)

    if st.button('Predict Cervical Cancer Risk'):
        # Convert categorical inputs to numerical
        smokes_encoded = 1 if smokes == 'Yes' else 0
        hormonal_contra_encoded = 1 if hormonal_contra == 'Yes' else 0
        iud_encoded = 1 if iud == 'Yes' else 0
        stds_encoded = 1 if stds == 'Yes' else 0

        # Create input data frame
        input_data = pd.DataFrame([[age_cervical, sexual_partners, first_intercourse, num_pregnancies,
                                  smokes_encoded, smokes_years, smokes_packs, hormonal_contra_encoded,
                                  hormonal_years, iud_encoded, iud_years, stds_encoded, stds_number]], 
                                columns=['Age', 'Number of sexual partners', 'First sexual intercourse',
                                       'Num of pregnancies', 'Smokes', 'Smokes (years)', 'Smokes (packs/year)',
                                       'Hormonal Contraceptives', 'Hormonal Contraceptives (years)',
                                       'IUD', 'IUD (years)', 'STDs', 'STDs (number)'])

        # Make prediction
        prediction = cervical_cancer_model.predict(input_data)
        prediction_proba = cervical_cancer_model.predict_proba(input_data)

        # Display results
        st.subheader('Prediction Result')
        if prediction[0] == 0:
            st.success('The patient is predicted to NOT have Cervical Cancer')
        else:
            st.error('The patient is predicted to have Cervical Cancer')
            
        st.write('Prediction Probability:')
        st.write(f'No Cervical Cancer: {prediction_proba[0][0]:.2%}')
        st.write(f'Cervical Cancer: {prediction_proba[0][1]:.2%}')
with tab8:
    st.title("Symptom Checker System")
    st.write('Enter symptoms to check possible conditions')

    # Create input fields
    fever = st.selectbox('Fever', ['Yes', 'No'])
    cough = st.selectbox('Cough', ['Yes', 'No'])
    fatigue = st.selectbox('Fatigue', ['Yes', 'No'])
    difficulty_breathing = st.selectbox('Difficulty Breathing', ['Yes', 'No'])
    age = st.number_input('Age', min_value=0, max_value=120, value=0)
    gender = st.selectbox('Gender', ['Male', 'Female'])
    blood_pressure = st.selectbox('Blood Pressure', ['Low', 'Normal', 'High'])
    cholesterol = st.selectbox('Cholesterol Level', ['Normal', 'High'])

    if st.button('Predict Disease'):
        # Transform the inputs
        fever_encoded = le_fever.transform([fever])[0]
        cough_encoded = le_cough.transform([cough])[0]
        fatigue_encoded = le_fatigue.transform([fatigue])[0]
        breathing_encoded = le_breathing.transform([difficulty_breathing])[0]
        gender_encoded = le_gender.transform([gender])[0]
        bp_encoded = le_bp.transform([blood_pressure])[0]
        cholesterol_encoded = le_cholesterol.transform([cholesterol])[0]
        
        # Create input array
        input_data = pd.DataFrame([[fever_encoded, cough_encoded, fatigue_encoded, 
                                   breathing_encoded, age, gender_encoded,
                                   bp_encoded, cholesterol_encoded]], 
                                 columns=['Fever', 'Cough', 'Fatigue', 'Difficulty Breathing',
                                        'Age', 'Gender', 'Blood Pressure', 'Cholesterol Level'])
        
        # Make prediction
        prediction = model.predict(input_data)
        predicted_disease = le_disease.inverse_transform(prediction)[0]
        
        st.success(f'Predicted Disease: {predicted_disease}')
        
        # Display confidence note
        st.info('Note: This is a preliminary prediction. Please consult a medical professional for accurate diagnosis.')
        
with tab9:
    st.title("Health Metrics Calculator")
    st.sidebar.title("Unit Preferences")
    weight_unit = st.sidebar.radio("Select weight unit:", ("kg", "lbs", "pounds"))
    height_unit = st.sidebar.radio("Select height unit:", ("cm", "meters"))
    waist_hip_unit = st.sidebar.radio("Select waist/hip unit:", ("cm", "inches"))

    # Select Metric to Calculate
    metric = st.selectbox("Select a Health Metric to Calculate:", 
        ["Body Mass Index (BMI)", 
         "Basal Metabolic Rate (BMR)",
         "Total Daily Energy Expenditure (TDEE)",
         "Body Fat Percentage",
         "Lean Body Mass (LBM)",
         "Waist-to-Hip Ratio (WHR)"
        ])

    # Inputs for Metrics
    gender = st.radio("Select your gender:", ("Male", "Female"))

    if metric == "Body Mass Index (BMI)":
        weight = st.number_input(f"Enter your weight ({weight_unit}):", min_value=0.0, format="%.2f")
        height = st.number_input(f"Enter your height ({height_unit}):", min_value=0.0, format="%.2f")
        if st.button("Calculate BMI"):
            if weight > 0 and height > 0:
                weight_kg = convert_weight(weight, weight_unit)
                height_m = convert_height(height, height_unit)
                bmi = calculate_bmi(weight_kg, height_m)
                category = interpret_bmi(bmi, gender)
                st.success(f"Your BMI is: {bmi} ({category})")

    elif metric == "Basal Metabolic Rate (BMR)":
        weight = st.number_input(f"Enter your weight ({weight_unit}):", min_value=0.0, format="%.2f")
        height = st.number_input(f"Enter your height ({height_unit}):", min_value=0.0, format="%.2f")
        age = st.number_input("Enter your age (years):", min_value=0, format="%d")
        if st.button("Calculate BMR"):
            if weight > 0 and height > 0 and age > 0:
                weight_kg = convert_weight(weight, weight_unit)
                height_m = convert_height(height, height_unit)
                bmr = calculate_bmr(weight_kg, height_m, age, gender)
                st.success(f"Your BMR is: {bmr} calories/day")

    elif metric == "Total Daily Energy Expenditure (TDEE)":
        weight = st.number_input(f"Enter your weight ({weight_unit}):", min_value=0.0, format="%.2f")
        height = st.number_input(f"Enter your height ({height_unit}):", min_value=0.0, format="%.2f")
        age = st.number_input("Enter your age (years):", min_value=0, format="%d")
        activity_level = st.selectbox("Select your activity level:", 
            ["Sedentary", "Lightly Active", "Moderately Active", "Very Active", "Extra Active"])
        if st.button("Calculate TDEE"):
            if weight > 0 and height > 0 and age > 0:
                weight_kg = convert_weight(weight, weight_unit)
                height_m = convert_height(height, height_unit)
                bmr = calculate_bmr(weight_kg, height_m, age, gender)
                tdee = calculate_tdee(bmr, activity_level)
                st.success(f"Your TDEE is: {tdee} calories/day")

    elif metric == "Body Fat Percentage":
        weight = st.number_input(f"Enter your weight ({weight_unit}):", min_value=0.0, format="%.2f")
        height = st.number_input(f"Enter your height ({height_unit}):", min_value=0.0, format="%.2f")
        age = st.number_input("Enter your age (years):", min_value=0, format="%d")
        if st.button("Calculate Body Fat Percentage"):
            if weight > 0 and height > 0 and age > 0:
                weight_kg = convert_weight(weight, weight_unit)
                height_m = convert_height(height, height_unit)
                body_fat = calculate_body_fat(weight_kg, height_m, age, gender)
                category = interpret_body_fat(body_fat, gender)
                st.success(f"Your Body Fat Percentage is: {body_fat}% ({category})")

    elif metric == "Lean Body Mass (LBM)":
        weight = st.number_input(f"Enter your weight ({weight_unit}):", min_value=0.0, format="%.2f")
        body_fat_percentage = st.number_input("Enter your body fat percentage (%):", min_value=0.0, max_value=100.0, format="%.2f")
        if st.button("Calculate LBM"):
            if weight > 0 and body_fat_percentage >= 0:
                weight_kg = convert_weight(weight, weight_unit)
                lbm = calculate_lbm(weight_kg, body_fat_percentage)
                st.success(f"Your Lean Body Mass is: {lbm} kg")

    elif metric == "Waist-to-Hip Ratio (WHR)":
        waist = st.number_input(f"Enter your waist circumference ({waist_hip_unit}):", min_value=0.0, format="%.2f")
        hip = st.number_input(f"Enter your hip circumference ({waist_hip_unit}):", min_value=0.0, format="%.2f")
        if st.button("Calculate WHR"):
            if waist > 0 and hip > 0:
                waist_cm = convert_waist_hip(waist, waist_hip_unit)
                hip_cm = convert_waist_hip(hip, waist_hip_unit)
                whr = calculate_wh_ratio(waist_cm, hip_cm)
                category = interpret_wh_ratio(whr, gender)
                st.success(f"Your Waist-to-Hip Ratio is: {whr} ({category})")

# Add information about the models
st.sidebar.header('About')
st.sidebar.info('''
This health prediction system uses machine learning to assess:
1. Diabetes risk based on the Pima Indians Diabetes Database
2. Heart Disease risk based on the UCI Heart Disease Dataset
3. Alzheimer's Disease risk based on clinical and behavioral factors
4. Parkinson's Disease risk based on voice recording measurements
5. Chronic Kidney Disease risk based on clinical measurements and medical history
6. Breast Cancer risk based on tumor characteristics from the Wisconsin Breast Cancer Dataset
7. Cervical Cancer risk based on patient history and clinical factors
8. General symptom checking for common conditions
9. Health metrics calculations including BMI, BMR, TDEE, Body Fat %, LBM, and WHR
''')