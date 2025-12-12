import streamlit as st
import pandas as pd
from utils import load_css, load_all_models, ask_medbot, MEDICAL_PROMPT, render_sidebar

# --- Config & Helpers ---
st.set_page_config(page_title="Heart Risk", page_icon="❤️", layout="wide")
load_css()
render_sidebar("Heart Risk")
MODELS = load_all_models()

# (Include your helper functions here: calculate_bmi, get_age_category, prepare_heart_features)
# ... [Paste the helper functions from your original file here] ...
# For brevity, I am assuming the helper functions are defined here exactly as in your original file.

# --- UI Layout ---
st.title("❤️ Heart Disease Risk Assessment")
st.markdown("### 10-Year Cardiovascular Risk Prediction")

if MODELS is None:
    st.error("⚠️ Models could not be loaded.")
    st.stop()

# --- Form Start ---
with st.form("heart_form"):
    
    # Section 1: Biometrics
    st.subheader("1️⃣ Biometrics & Vitals")
    c1, c2, c3, c4 = st.columns(4)
    with c1: age = st.number_input("Age", 18, 120, 45)
    with c2: sex = st.selectbox("Sex", ['Female', 'Male'])
    with c3: height = st.number_input("Height (cm)", 100.0, 250.0, 170.0)
    with c4: weight = st.number_input("Weight (kg)", 30.0, 200.0, 80.0)

    st.markdown("---")

    # Section 2: Medical History
    st.subheader("2️⃣ Medical History")
    m1, m2, m3 = st.columns(3)
    with m1: 
        general_health = st.selectbox("General Health", ['Excellent', 'Very Good', 'Good', 'Fair', 'Poor'], index=2)
        diabetes = st.selectbox("Diabetes History", ['No', 'No Pre Diabetes', 'Only during pregnancy', 'Yes'])
    with m2:
        checkup = st.selectbox("Last Checkup", ['Past 1 year', 'Past 2 years', 'Past 5 years', 'More than 5 years', 'Never'])
        arthritis = st.selectbox("Arthritis?", ['No', 'Yes'])
    with m3:
        depression = st.selectbox("Depression?", ['No', 'Yes'])
        skin_cancer = st.selectbox("Skin Cancer?", ['No', 'Yes'])
        other_cancer = st.selectbox("Other Cancer?", ['No', 'Yes'])

    st.markdown("---")

    # Section 3: Lifestyle
    st.subheader("3️⃣ Lifestyle Habits")
    l1, l2, l3 = st.columns(3)
    with l1:
        smoking = st.selectbox("Smoking Status", ['Never', 'Former', 'Current'])
        alcohol = st.selectbox("Alcohol Consumption", ['Never', 'Occasionally', 'Weekly', 'Daily'])
    with l2:
        exercise = st.selectbox("Exercise (Past 30 Days)", ['Yes', 'No'])
        fried_potato = st.selectbox("Fried Food", ['Rarely', 'Weekly', 'Several times per week'])
    with l3:
        fruit = st.selectbox("Fruit Intake", ['0', '1–2', '3–5', '6–7'])
        vegetables = st.selectbox("Vegetable Intake", ['0', '1–2', '3–5', '6–7'])

    st.markdown("<br>", unsafe_allow_html=True)
    submitted = st.form_submit_button("💓 Analyze Heart Health", use_container_width=True, type="primary")

# --- Logic ---
if submitted:
    data = {
        'Age': age, 'Sex': sex, 'Height': height, 'Weight': weight,
        'General_Health': general_health, 'Checkup': checkup, 'Diabetes': diabetes,
        'Arthritis': arthritis, 'Depression': depression, 'Skin_Cancer': skin_cancer, 'Other_Cancer': other_cancer,
        'Smoking_History': smoking, 'Exercise': exercise,
        'Alcohol_Consumption': alcohol, 'Fruit_Consumption': fruit,
        'Vegetables_Consumption': vegetables, 'FriedPotato_Consumption': fried_potato,
    }
    
    with st.spinner('Calculating risk factors...'):
        try:
            # Prepare & Predict
            # NOTE: Ensure prepare_heart_features handles Skin_Cancer/Other_Cancer if added to inputs
            features = prepare_heart_features(data) 
            prob = MODELS['heart_model'].predict_proba(features)[0][1]
            risk_percent = prob * 100
            
            # UI Display
            st.divider()
            
            col_res, col_ai = st.columns([1, 1.5])
            
            with col_res:
                st.subheader("Risk Score")
                if risk_percent > 50:
                    st.error(f"High Risk: {risk_percent:.1f}%")
                    st.progress(min(int(risk_percent), 100), text="Risk Probability")
                elif risk_percent > 20:
                    st.warning(f"Moderate Risk: {risk_percent:.1f}%")
                    st.progress(min(int(risk_percent), 100), text="Risk Probability")
                else:
                    st.success(f"Low Risk: {risk_percent:.1f}%")
                    st.progress(min(int(risk_percent), 100), text="Risk Probability")
            
            with col_ai:
                st.subheader("Dr. AI Analysis")
                prediction_label = "High Risk" if prob > 0.5 else "Low Risk"
                ai_prompt = f"Heart Risk: {prediction_label} ({risk_percent:.1f}%). Age: {age}, Smoking: {smoking}. Advice?"
                explanation = ask_medbot(ai_prompt, MEDICAL_PROMPT)
                st.info(explanation)
                
        except Exception as e:
            st.error(f"Prediction failed: {e}")
