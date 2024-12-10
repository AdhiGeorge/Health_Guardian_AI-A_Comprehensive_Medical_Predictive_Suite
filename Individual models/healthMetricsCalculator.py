import streamlit as st

# Unit Conversion Functions
def convert_weight(weight, unit):
    if unit in ["lbs", "pounds"]:
        return weight * 0.453592  # lbs/pounds to kg
    return weight  # kg

def convert_height(height, unit):
    if unit == "cm":
        return height / 100  # cm to meters
    return height  # meters

def convert_waist_hip(value, unit):
    if unit == "inches":
        return value * 2.54  # inches to cm
    return value  # cm

# Result Interpretation Functions
def interpret_bmi(bmi, gender):
    if gender == "Male":
        if bmi < 18.5:
            return "Underweight"
        elif 18.5 <= bmi < 24.9:
            return "Normal weight"
        elif 25 <= bmi < 29.9:
            return "Overweight"
        else:
            return "Obesity"
    else:
        if bmi < 18.5:
            return "Underweight"
        elif 18.5 <= bmi < 24.9:
            return "Normal weight"
        elif 25 <= bmi < 29.9:
            return "Overweight"
        else:
            return "Obesity"

def interpret_body_fat(body_fat, gender):
    if gender == "Male":
        if body_fat < 6:
            return "Essential fat"
        elif 6 <= body_fat < 24:
            return "Athletic/Fit"
        elif 24 <= body_fat < 31:
            return "Acceptable"
        else:
            return "Obesity"
    else:
        if body_fat < 14:
            return "Essential fat"
        elif 14 <= body_fat < 31:
            return "Athletic/Fit"
        elif 31 <= body_fat < 36:
            return "Acceptable"
        else:
            return "Obesity"

def interpret_wh_ratio(whr, gender):
    if gender == "Male":
        return "High Risk" if whr > 0.9 else "Low Risk"
    else:
        return "High Risk" if whr > 0.85 else "Low Risk"

# Calculation Functions
def calculate_bmi(weight, height):
    return round(weight / (height ** 2), 2)

def calculate_bmr(weight, height, age, gender):
    if gender == "Male":
        return round(10 * weight + 6.25 * (height * 100) - 5 * age + 5, 2)  # height in cm
    else:
        return round(10 * weight + 6.25 * (height * 100) - 5 * age - 161, 2)

def calculate_tdee(bmr, activity_level):
    activity_multipliers = {
        "Sedentary": 1.2,
        "Lightly Active": 1.375,
        "Moderately Active": 1.55,
        "Very Active": 1.725,
        "Extra Active": 1.9
    }
    return round(bmr * activity_multipliers[activity_level], 2)

def calculate_body_fat(weight, height, age, gender):
    if gender == "Male":
        return round((1.20 * (weight / (height ** 2)) + 0.23 * age - 16.2), 2)
    else:
        return round((1.20 * (weight / (height ** 2)) + 0.23 * age - 5.4), 2)

def calculate_lbm(weight, body_fat_percentage):
    return round(weight * (1 - body_fat_percentage / 100), 2)

def calculate_wh_ratio(waist, hip):
    return round(waist / hip, 2)

# Streamlit UI
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