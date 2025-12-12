import streamlit as st
from utils import load_css, load_all_models, prepare_diabetes_features, calculate_bmi, ask_medbot, MEDICAL_PROMPT, render_sidebar

st.set_page_config(page_title="Diabetes Risk", page_icon="🩸", layout="wide")
load_css()
render_sidebar("Diabetes")
MODELS = load_all_models()

st.title("🩸 Diabetes Risk Assessment")
st.markdown("Fill in the clinical data below to estimate the risk.")

# --- Input Form (Prevents auto-reload) ---
with st.form("diabetes_form"):
    st.subheader("📝 Clinical Data")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("**👤 Personal Info**")
        age = st.number_input("Age", 1, 100, 30)
        pregnancies = st.number_input("Pregnancies", 0, 20, 0)
    
    with col2:
        st.markdown("**📏 Body Metrics**")
        weight = st.number_input("Weight (kg)", 10, 300, 70)
        height = st.number_input("Height (cm)", 50, 250, 170)
        
    with col3:
        st.markdown("**🩺 Vitals**")
        glucose = st.number_input("Glucose Level (mg/dL)", 0, 500, 100)
        bp = st.number_input("Blood Pressure (mm Hg)", 0, 200, 70)

    st.markdown("---")
    submit_btn = st.form_submit_button("🔍 Calculate Risk", use_container_width=True, type="primary")

# --- Result Section ---
if submit_btn:
    if MODELS:
        try:
            bmi = calculate_bmi(height, weight)
            data = {'Age': age, 'Glucose': glucose, 'BP': bp, 'Pregnancies': pregnancies, 'BMI': bmi}
            
            # Prediction
            features = prepare_diabetes_features(data, MODELS['diabetes_scaler'])
            prediction = MODELS['diabetes_model'].predict(features)[0]
            
            # Display
            st.divider()
            r_col1, r_col2 = st.columns([1, 2])
            
            with r_col1:
                st.metric("Calculated BMI", f"{bmi:.1f}")
                if prediction == 1:
                    st.error("## Result: Diabetic")
                    st.write("⚠️ High likelihood of Diabetes.")
                else:
                    st.success("## Result: Healthy")
                    st.write("✅ Metrics are within normal range.")
            
            with r_col2:
                with st.spinner("Generating medical advice..."):
                    result_text = "Diabetic" if prediction == 1 else "Healthy"
                    ai_exp = ask_medbot(f"Diabetes Check: {result_text}. Glucose: {glucose}, BMI: {bmi:.1f}, BP: {bp}. Advice?", MEDICAL_PROMPT)
                    st.info(f"👨‍⚕️ **Dr. AI Advice:**\n\n{ai_exp}")

        except Exception as e:
            st.error(f"Error: {e}")
