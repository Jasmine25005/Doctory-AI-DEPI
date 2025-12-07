import streamlit as st
import pandas as pd
# Import helpers from our central utils file
from utils import load_css, load_all_models, ask_medbot, MEDICAL_PROMPT, render_sidebar

# --- 1. Page Config & Theme ---
st.set_page_config(page_title="Heart Risk", page_icon="❤️", layout="wide")
load_css() # Apply the Medical Blue Theme
render_sidebar("Heart Risk")

# --- 3. Load Models ---
MODELS = load_all_models()

if MODELS is None:
    st.error("Model initialization failed. Check 'models/' folder.")
    st.stop()

# --- 4. Helper Functions (Specific to Heart Model) ---
def calculate_bmi(height_cm, weight_kg):
    if height_cm == 0: return 0
    return weight_kg / ((height_cm / 100) ** 2)

def get_age_category(age):
    age = int(age)
    if 18 <= age <= 24: return 'Young'
    if 25 <= age <= 39: return 'Adult'
    if 40 <= age <= 54: return 'Mid-Aged'
    if 55 <= age <= 64: return 'Senior-Adult'
    if age >= 65: return 'Elderly'
    return 'Adult'

def prepare_heart_features(data):
    scaler = MODELS['heart_scaler']
    
    height = data.get('Height')
    weight = data.get('Weight')
    age = data.get('Age')
    bmi = calculate_bmi(height, weight)
    
    # Mappings
    general_health_map = {'Excellent': 0, 'Fair': 1, 'Good': 2, 'Poor': 3, 'Very Good': 4}
    checkup_map = {'More than 5 years': 0, 'Never': 1, 'Past 1 year': 2, 'Past 2 years': 3, 'Past 5 years': 4}
    binary_map = {'No': 0, 'Yes': 1} 
    diabetes_map = {'No': 0, 'No Pre Diabetes': 1, 'Only during pregnancy': 2, 'Yes': 3}
    age_category_map = {'Adult': 0, 'Elderly': 1, 'Mid-Aged': 2, 'Senior-Adult': 3, 'Young': 4}
    bmi_group_map = {'Normal weight': 0, 'Obese I': 1, 'Obese II': 2, 'Overweight': 3, 'Underweight': 4}

    # BMI Group Calculation
    bmi_bins = [12.02, 18.3, 26.85, 31.58, 37.8, 100]
    bmi_labels = ['Underweight', 'Normal weight', 'Overweight', 'Obese I', 'Obese II']
    try:
        bmi_group_str = pd.cut([bmi], bins=bmi_bins, labels=bmi_labels, right=False)[0]
    except (ValueError, IndexError):
        bmi_group_str = 'Normal weight'

    # Lifestyle Mappers
    def map_smoking(val): return 1 if val in ['Former', 'Current'] else 0 
    def map_alcohol(val):
        if val == 'Never': return 0
        if val == 'Occasionally': return 4
        if val == 'Weekly': return 8
        if val == 'Daily': return 30
        return 0
    def map_consumption(val):
        if val == '0': return 0
        if val == '1–2': return 12 
        if val == '3–5': return 20 
        if val == '6–7': return 30 
        return 0
    def map_fried(val):
        if val == 'Rarely': return 2
        if val == 'Weekly': return 4
        if val == 'Several times per week': return 8
        return 0
        
    age_cat_str = get_age_category(age)

    feature_dict = {
        'general_health': general_health_map.get(data.get('General_Health')),
        'checkup': checkup_map.get(data.get('Checkup')),
        'exercise': binary_map.get(data.get('Exercise')),
        'skin_cancer': binary_map.get(data.get('Skin_Cancer')),
        'other_cancer': binary_map.get(data.get('Other_Cancer')),
        'depression': binary_map.get(data.get('Depression')),
        'diabetes': diabetes_map.get(data.get('Diabetes')),
        'arthritis': binary_map.get(data.get('Arthritis')),
        'age_category': age_category_map.get(age_cat_str),
        'height': height,
        'weight': weight,
        'bmi': bmi,
        'bmi_group': bmi_group_map.get(bmi_group_str, 0), 
        'alcohol_consumption': map_alcohol(data.get('Alcohol_Consumption')),
        'fruit_consumption': map_consumption(data.get('Fruit_Consumption')),
        'vegetables_consumption': map_consumption(data.get('Vegetables_Consumption')),
        'potato_consumption': map_fried(data.get('FriedPotato_Consumption')),
        'sex_Female': 1 if data.get('Sex') == 'Female' else 0,
        'sex_Male': 1 if data.get('Sex') == 'Male' else 0,
        'smoking_history_No': 1 if map_smoking(data.get('Smoking_History')) == 0 else 0,
        'smoking_history_Yes': 1 if map_smoking(data.get('Smoking_History')) == 1 else 0,
    }

    final_feature_order = [
        'general_health', 'checkup', 'exercise', 'skin_cancer', 'other_cancer',
        'depression', 'diabetes', 'arthritis', 'age_category', 'height', 'weight',
        'bmi', 'bmi_group', 'alcohol_consumption', 'fruit_consumption', 'vegetables_consumption',
        'potato_consumption', 'sex_Female', 'sex_Male',
        'smoking_history_No', 'smoking_history_Yes'
    ]
    
    features = pd.DataFrame([feature_dict], columns=final_feature_order)
    return scaler.transform(features)

# --- 5. UI Layout ---

st.title("❤️ Heart Disease Risk Assessment")

st.markdown('<div class="css-card">', unsafe_allow_html=True)
st.write("Provide your lifestyle and health inputs for a 10-year risk assessment.")

# Input form 
with st.form("heart_form"):
    col1, col2, col3 = st.columns(3)
    
    # COLUMN 1: Basic Biometrics
    with col1:
        st.markdown("###  Biometrics")
        age = st.number_input("Age (Years)", min_value=18, max_value=120, value=45)
        sex = st.selectbox("Sex", ['Female', 'Male'])
        height = st.number_input("Height (cm)", min_value=100.0, max_value=250.0, value=170.0, format="%.1f")
        weight = st.number_input("Weight (kg)", min_value=30.0, max_value=200.0, value=80.0, format="%.1f")

    # COLUMN 2: Medical History
    with col2:
        st.markdown("###  Health History")
        general_health = st.selectbox("General Health Status", ['Very Good', 'Good', 'Fair', 'Poor', 'Excellent'])
        checkup = st.selectbox("Last Health Checkup", ['Past 1 year', 'Past 2 years', 'Past 5 years', 'More than 5 years', 'Never'])
        diabetes = st.selectbox("Diabetes Status", ['No', 'No Pre Diabetes', 'Only during pregnancy', 'Yes'])
        arthritis = st.selectbox("Have Arthritis?", ['No', 'Yes'])
        depression = st.selectbox("Have Depression?", ['No', 'Yes'])

    # COLUMN 3: Lifestyle
    with col3:
        st.markdown("###  Lifestyle")
        smoking = st.selectbox("Smoking History", ['Never', 'Former', 'Current'])
        exercise = st.selectbox("Any physical exercise in past 30 days?", ['No', 'Yes'])
        alcohol = st.selectbox("Avg. Alcoholic drinks per day", ['Never', 'Occasionally', 'Weekly', 'Daily'], index=0)
        fruit = st.selectbox("Fruit servings per day", ['0', '1–2', '3–5', '6–7'])
        vegetables = st.selectbox("Vegetable servings per day", ['0', '1–2', '3–5', '6–7'])
        fried_potato = st.selectbox("Fried Potato consumption", ['Rarely', 'Weekly', 'Several times per week'])

    st.markdown("<br>", unsafe_allow_html=True)
    submitted = st.form_submit_button("🔍 Predict Heart Risk")

st.markdown('</div>', unsafe_allow_html=True)

# --- 6. Prediction Logic ---
if submitted:
    if height == 0 or weight == 0:
        st.error("Height and Weight must be greater than zero.")
    else:
        data = {
            'Age': age, 'Sex': sex, 'Height': height, 'Weight': weight,
            'General_Health': general_health, 'Checkup': checkup, 'Diabetes': diabetes,
            'Arthritis': arthritis, 'Depression': depression,
            'Smoking_History': smoking, 'Exercise': exercise,
            'Alcohol_Consumption': alcohol, 'Fruit_Consumption': fruit,
            'Vegetables_Consumption': vegetables, 'FriedPotato_Consumption': fried_potato,
        }
        
        with st.spinner('Analyzing cardiovascular health...'):
            try:
                # Prepare features
                features = prepare_heart_features(data)
                
                # Predict
                prob = MODELS['heart_model'].predict_proba(features)[0][1]
                risk_percent = prob * 100
                
                prediction_label = "High Risk" if prob > 0.5 else "Low Risk"
                
                if prob > 0.5:
                    color = "#D32F2F" # Red
                else:
                    color = "#388E3C" # Green
                
                # Display Result
                st.markdown(f"### Result: <span style='color:{color}'>{prediction_label}</span> (Risk: {risk_percent:.1f}%)", unsafe_allow_html=True)

                # AI Explanation
                ai_prompt = f"Patient Heart Risk Analysis: {prediction_label} ({risk_percent:.1f}%). Age: {age}, Smoking: {smoking}, Exercise: {exercise}. Explain this result."
                explanation = ask_medbot(ai_prompt, MEDICAL_PROMPT)
                
                st.info(f"👨‍⚕️ **Dr. AI Analysis:**\n\n{explanation}")
                
            except Exception as e:
                st.error(f"Prediction failed: {e}")
