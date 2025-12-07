import streamlit as st
from utils import load_css, load_all_models, prepare_diabetes_features, calculate_bmi, ask_medbot, MEDICAL_PROMPT, render_sidebar

st.set_page_config(page_title="Diabetes", page_icon="🩸", layout="wide")
load_css()
MODELS = load_all_models()
render_sidebar("Diabetes")

st.title("🩸 Diabetes Risk Assessment")

st.markdown('<div class="css-card">', unsafe_allow_html=True)
col1, col2, col3 = st.columns(3)
with col1:
    age = st.number_input("Age", 1, 100, 30)
    glucose = st.number_input("Glucose", 0, 500, 100)
with col2:
    weight = st.number_input("Weight (kg)", 10, 300, 70)
    bp = st.number_input("Blood Pressure", 0, 200, 70)
with col3:
    height = st.number_input("Height (cm)", 50, 250, 170)
    pregnancies = st.number_input("Pregnancies", 0, 20, 0)
st.markdown('</div>', unsafe_allow_html=True)

if st.button("Analyze Result"):
    if MODELS:
        try:
            bmi = calculate_bmi(height, weight)
            data = {'Age': age, 'Glucose': glucose, 'BP': bp, 'Pregnancies': pregnancies, 'BMI': bmi}
            
            features = prepare_diabetes_features(data, MODELS['diabetes_scaler'])
            prediction = MODELS['diabetes_model'].predict(features)[0]
            
            result = "Diabetic" if prediction == 1 else "Healthy"
            color = "#D32F2F" if prediction == 1 else "#388E3C"
            
            st.markdown(f"### Result: <span style='color:{color}'>{result}</span>", unsafe_allow_html=True)
            ai_exp = ask_medbot(f"Result: {result}. Glucose: {glucose}, BMI: {bmi:.1f}. Explain.", MEDICAL_PROMPT)
            st.info(f"👨‍⚕️ Analysis: {ai_exp}")
        except Exception as e:
            st.error(f"Error: {e}")
