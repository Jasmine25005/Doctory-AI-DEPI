import streamlit as st
import numpy as np
from PIL import Image
from utils import load_css, load_all_models, process_image, ask_medbot, MEDICAL_PROMPT, render_sidebar

# --- Page Config ---
st.set_page_config(page_title="Pneumonia Check", page_icon="🫁", layout="wide")
load_css()
render_sidebar("Pneumonia")

# --- Load Models ---
MODELS = load_all_models()

# --- Header ---
st.title("🫁 Pneumonia X-Ray Detection")
st.markdown("### AI-Powered Chest X-Ray Analysis")
st.info("Upload a standard Chest X-Ray (JPEG/PNG) to detect signs of Pneumonia.")

# --- Input Section ---
st.markdown('<div class="css-card">', unsafe_allow_html=True)
uploaded_file = st.file_uploader("📂 Upload X-Ray Image", type=["jpg", "png", "jpeg"])
st.markdown('</div>', unsafe_allow_html=True)
# --- Analysis ---
if uploaded_file:
    # 1. نقرأ الصورة كـ Bytes مرة واحدة بس هنا عشان الـ Buffer ميفضاش
    image_bytes = uploaded_file.getvalue()
    
    col_img, col_res = st.columns([1, 2])
    
    with col_img:
        # 2. نمرر الـ image_bytes بدل uploaded_file ونستخدم use_container_width
        st.image(image_bytes, caption="Uploaded X-Ray", use_container_width=True)
        run_btn = st.button("🫁 Analyze X-Ray", use_container_width=True, type="primary")

    with col_res:
        if run_btn:
            if MODELS and MODELS.get('pneumonia_sess'):
                try:
                    with st.spinner("Analyzing lung scans..."):
                        # 3. نمرر نفس الـ image_bytes للموديل بدل uploaded_file.read()
                        img_input = process_image(image_bytes)
                        session = MODELS['pneumonia_sess']
                        result = session.run([MODELS['pneu_out']], {MODELS['pneu_in']: img_input})
                        
                        prediction = result[0][0][0]
                        
                        # Threshold Logic (افترضي إن 0.5 هو الحد الفاصل)
                        if prediction > 0.5:
                            label = "Pneumonia"
                            confidence = prediction
                            is_healthy = False
                        else:
                            label = "Normal"
                            confidence = 1 - prediction
                            is_healthy = True

                        # Display Result
                        st.subheader("Analysis Results")
                        
                        if is_healthy:
                            st.success(f"## ✅ {label}")
                            st.progress(float(confidence), text=f"Confidence: {confidence:.2%}")
                        else:
                            st.error(f"## 🫁 {label}")
                            st.progress(float(confidence), text=f"Confidence: {confidence:.2%}")
                            st.warning("⚠️ High Risk: Signs of lung infection detected.")

                        # AI Explanation
                        st.divider()
                        ai_prompt = f"Chest X-Ray analysis result: {label}. Explain this result briefly."
                        explanation = ask_medbot(ai_prompt, MEDICAL_PROMPT)
                        st.caption("Dr. AI Analysis:")
                        st.write(explanation)

                except Exception as e:
                    st.error(f"Error: {e}")
            else:
                st.error("Model not loaded.")
