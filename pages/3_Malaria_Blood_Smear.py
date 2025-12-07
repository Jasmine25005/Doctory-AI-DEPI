import streamlit as st
import numpy as np
from PIL import Image
import io
from utils import load_css, load_all_models, ask_medbot, MEDICAL_PROMPT, render_sidebar

# --- 1. Page Config ---
st.set_page_config(page_title="Malaria Check", page_icon="🦟", layout="wide")
load_css() # Loads the Blue/White Theme
render_sidebar("Malaria")

# --- 3. Load Models ---
MODELS = load_all_models()

st.title("🦟 Malaria Blood Cell Analysis")
st.markdown("Upload a microscopic image of a blood cell to detect if it is Parasitized or Uninfected.")

# --- 4. Helper: Preprocessing for Malaria (Keras-style) ---
def process_malaria_image(image_bytes):
    """
    Malaria model expects: (1, 224, 224, 3) 
    No Transpose (HWC format)
    """
    img = Image.open(io.BytesIO(image_bytes)).convert('RGB')
    img = img.resize((224, 224))
    img_np = np.array(img).astype(np.float32) / 255.0 # Normalize
    img_np = np.expand_dims(img_np, axis=0) # Add batch dimension -> (1, 224, 224, 3)
    return img_np

# --- 5. Input Section (Medical Blue Card) ---
# --- 5. Input Section (Medical Blue Card) ---
st.markdown('<div class="css-card">', unsafe_allow_html=True)
col1, col2, col3 = st.columns([1, 2, 1])

with col2:  # Centered column
    uploaded_file = st.file_uploader("Upload Cell Image", type=["jpg", "png", "jpeg"])

st.markdown('</div>', unsafe_allow_html=True)

# --- 6. Analysis Logic ---
if uploaded_file:
    # Center Image
    col_img1, col_img2, col_img3 = st.columns([1, 1, 1])
    with col_img2:
        st.image(uploaded_file, caption="Microscopic View", width=300)

    if st.button("Analyze Cell"):
        if MODELS and MODELS.get('malaria_sess'):
            try:
                with st.spinner("Analyzing cellular structure..."):
                    # 1. Preprocess
                    image_bytes = uploaded_file.read()
                    img_input = process_malaria_image(image_bytes)
                    
                    # 2. Inference
                    session = MODELS['malaria_sess']
                    input_name = MODELS['mal_in']
                    output_name = MODELS['mal_out']
                    
                    # ONNX Run
                    result = session.run([output_name], {input_name: img_input})
                    
                    # 3. Process Result
                    # Assuming output is a probability [0-1] where <0.5 is Parasitized (check your model training)
                    # Or output is [Uninfected_Prob, Parasitized_Prob]
                    # Let's assume standard single neuron output:
                    prediction = result[0][0][0] 
                    
                    # Logic: Usually 0 = Parasitized, 1 = Uninfected OR vice versa. 
                    # Adjust this threshold logic based on your specific training.
                    # Commonly: < 0.5 = Parasitized, > 0.5 = Uninfected
                    if prediction > 0.5:
                        label = "Uninfected (Healthy)"
                        color = "#388E3C" # Green
                        risk = "Low"
                    else:
                        label = "Parasitized (Infected)"
                        color = "#D32F2F" # Red
                        risk = "High"

                    # 4. Display Result
                    st.markdown(f"### Result: <span style='color:{color}'>{label}</span>", unsafe_allow_html=True)
                    
                    # 5. AI Explanation
                    ai_prompt = f"Malaria cell analysis result: {label}. Risk Level: {risk}. Explain this result."
                    explanation = ask_medbot(ai_prompt, MEDICAL_PROMPT)
                    
                    st.info(f"👨‍⚕️ **Dr. AI Analysis:**\n\n{explanation}")

            except Exception as e:
                st.error(f"Analysis Error: {e}")
        else:
            st.warning("⚠️ Malaria model not loaded. Check 'models/malaria_model.onnx'.")
