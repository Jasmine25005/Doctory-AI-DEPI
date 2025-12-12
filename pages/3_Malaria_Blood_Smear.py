import streamlit as st
import numpy as np
from PIL import Image
import io
from utils import load_css, load_all_models, ask_medbot, MEDICAL_PROMPT, render_sidebar

# --- Page Config ---
st.set_page_config(page_title="Malaria Check", page_icon="🦟", layout="wide")
load_css()
render_sidebar("Malaria")

MODELS = load_all_models()

st.title("🦟 Malaria Cell Analysis")
st.markdown("### Microscopic Blood Smear Detection")

# --- Helper ---
def process_malaria_image(image_bytes):
    img = Image.open(io.BytesIO(image_bytes)).convert('RGB')
    img = img.resize((224, 224))
    img_np = np.array(img).astype(np.float32) / 255.0
    return np.expand_dims(img_np, axis=0)

# --- Input ---
col_up, col_info = st.columns([1, 1])
with col_up:
    st.markdown('<div class="css-card">', unsafe_allow_html=True)
    uploaded_file = st.file_uploader("🔬 Upload Cell Image", type=["jpg", "png", "jpeg"])
    st.markdown('</div>', unsafe_allow_html=True)
with col_info:
    st.info("💡 **Tip:** Ensure the image is a clear, single-cell crop from a blood smear slide.")

# --- Analysis ---
if uploaded_file:
    col_img, col_res = st.columns([1, 2])
    
    with col_img:
        st.image(uploaded_file, caption="Microscopic View", use_column_width=True)
        run_btn = st.button("🔬 Analyze Sample", use_container_width=True, type="primary")

    with col_res:
        if run_btn:
            if MODELS and MODELS.get('malaria_sess'):
                try:
                    with st.spinner("Analyzing cellular structure..."):
                        # Inference
                        img_input = process_malaria_image(uploaded_file.read())
                        session = MODELS['malaria_sess']
                        result = session.run([MODELS['mal_out']], {MODELS['mal_in']: img_input})
                        
                        prediction = result[0][0][0]
                        
                        # Threshold Logic
                        if prediction > 0.5:
                            label = "Uninfected"
                            confidence = prediction
                            is_healthy = True
                        else:
                            label = "Parasitized"
                            confidence = 1 - prediction
                            is_healthy = False

                        # Display Result
                        st.subheader("Analysis Results")
                        
                        if is_healthy:
                            st.success(f"## ✅ {label}")
                            st.progress(float(confidence), text=f"Confidence: {confidence:.2%}")
                        else:
                            st.error(f"## 🦠 {label}")
                            st.progress(float(confidence), text=f"Confidence: {confidence:.2%}")
                            st.warning("⚠️ High Risk: Plasmodium parasite detected.")

                        # AI Explanation
                        st.divider()
                        ai_prompt = f"Malaria cell analysis result: {label}. Explain this result."
                        explanation = ask_medbot(ai_prompt, MEDICAL_PROMPT)
                        st.caption("Dr. AI Analysis:")
                        st.write(explanation)

                except Exception as e:
                    st.error(f"Error: {e}")
            else:
                st.error("Model not loaded.")
