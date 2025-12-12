import streamlit as st
from utils import ask_medbot, load_css, MEDICAL_PROMPT, render_sidebar

# --- Page Config ---
st.set_page_config(page_title="Chat with Doctory", page_icon="💬", layout="wide")
load_css()
render_sidebar("AI Chat")

# --- Header ---
col1, col2 = st.columns([1, 8])
with col1:
    st.image("https://cdn-icons-png.flaticon.com/512/4712/4712009.png", width=70) # Generic bot icon
with col2:
    st.title("Doctory AI Assistant")
    st.markdown("ask me anything about your symptoms or medical reports.")

st.divider()

# --- Session State ---
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "Hello! I am Doctory. How can I assist you today?"}]

# --- Sidebar Action ---
with st.sidebar:
    if st.button("🗑️ Clear Chat History", use_container_width=True):
        st.session_state.messages = [{"role": "assistant", "content": "Hello! I am Doctory. How can I assist you today?"}]
        st.rerun()

# --- Display Chat ---
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

# --- User Input ---
if prompt := st.chat_input("Type your health question here..."):
    # User message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.write(prompt)
    
    # AI Response
    with st.chat_message("assistant"):
        with st.spinner("Dr. AI is thinking..."):
            reply = ask_medbot(prompt, MEDICAL_PROMPT)
            st.write(reply)
            st.session_state.messages.append({"role": "assistant", "content": reply})
