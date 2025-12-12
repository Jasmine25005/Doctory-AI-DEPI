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

# --- Analysis Logic ---
if uploaded_file:
    # Layout: Image on Left, Results on Right
    col1, col2 = st.columns([1, 1.5])
    
    with col1:
        st.image(uploaded_file, caption="Uploaded X-Ray", use_column_width=True)
        analyze_btn = st.button("🔍 Analyze Image", use_container_width=True, type="primary")

    with col2:
        if analyze_btn:
            if MODELS and MODELS.get('pneumonia_sess'):
                try:
                    with st.spinner("Scanning lung opacities..."):
                        # 1. Preprocess
                        image_bytes = uploaded_file.read()
                        img_input = process_image(image_bytes)
                        
                        # 2. Inference
                        session = MODELS['pneumonia_sess']
                        result = session.run([MODELS['pneu_out']], {MODELS['pneu_in']: img_input})
                        
                        # 3. Process Result
                        probs = result[0][0]
                        idx = np.argmax(probs)
                        classes = ["Normal", "Pneumonia (Bacterial)", "Pneumonia (Viral)"]
                        
                        final_result = classes[idx] if idx < len(classes) else "Unknown"
                        
                        # 4. Display Logic
                        if "Normal" in final_result:
                            st.success(f"### ✅ Result: {final_result}")
                            st.balloons()
                        else:
                            st.error(f"### ⚠️ Result: {final_result}")
                        
                        # 5. AI Explanation
                        st.markdown("---")
                        st.subheader("🤖 Dr. AI Insights")
                        ai_prompt = f"Chest X-Ray analysis result: {final_result}. Explain this medical condition briefly to a patient."
                        explanation = ask_medbot(ai_prompt, MEDICAL_PROMPT)
                        st.write(explanation)

                except Exception as e:
                    st.error(f"Analysis Error: {e}")
            else:
                st.warning("⚠️ Model not loaded. Please check system configuration.")
