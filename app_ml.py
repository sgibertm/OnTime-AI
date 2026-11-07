import streamlit as st

from Navegation.home import home_page
from Navegation.prediction import prediction
from Navegation.notifications import notifications
from Logic.utils_ml import *

# Page Configuration
st.set_page_config(
    page_title="OnTime AI", 
    page_icon="Img/logo.png", 
    layout="wide"
)

st.logo("Img/logo.png")

# Global Styles (CSS)
st.markdown("""
    <style>

    .stApp {
        background: linear-gradient(135deg, #f8fafc 0%, #eef2ff 100%);
        font-family: 'Inter', sans-serif;
    }

    /* Sidebar */
    [data-testid="stSidebar"] {
        background-color: #d3e4ff;
    }
    
    /* Button base */
    .stButton > button {
        font-weight: 600;
        transition: all 0.2s;
        background-color: #d3e4ff;
    }
    
    /* Primary (dark blue -> active) */
    .stButton > button[kind="primary"] {
        background-color: #1d4ed8 !important;
        color: white !important;
        border: none;
    }
    
    .stButton > button[kind="primary"]:hover {
        background-color: #1e40af !important;
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(29, 78, 216, 0.3);
    }
    
    /* Secondary (light blue -> inactive) */
    .stButton > button[kind="secondary"] {
        background-color: #3b82f6 !important;
        color: white !important;
        opacity: 0.85;
    }
    
    .stButton > button[kind="secondary"]:hover {
        background-color: #2563eb !important;
        transform: translateY(-1px);
    }
        
    [data-testid="stChatMessage"] {
        border-radius: 20px;
        padding: 15px;
        margin-bottom: 10px;
        max-width: 80%;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.1);
    }
    
    [data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) {
        background-color: #d3e4ff;
        margin-left: auto; 
        flex-direction: row-reverse; 
        border-bottom-right-radius: 2px; 
        color: #000000;
    }
    </style>
""", unsafe_allow_html=True)


# Session State Initialization
if "page" not in st.session_state:
    st.session_state.page = "home"

if "model_choice" not in st.session_state:
    st.session_state.model_choice = "Linear Regression"



# Sidebar Information
with st.sidebar:
    st.image("Img/logotype.png", use_container_width=True)
    st.divider()

    if "page" not in st.session_state:
        st.session_state.page = "home"

    st.subheader("Current page")
    st.write("Page:", st.session_state.page)

    st.divider()

    st.subheader("Model Selected")
    # Display whichever algorithm is currently chosen in home.py
    st.info(st.session_state.get("model_choice", "Linear Regression"))
    
    st.divider()

    st.subheader("Dataset")
    st.write("Rows loaded:", len(load_data()))
    


col1, col2, col3 = st.columns(3)

with col1:
    if st.button("Home", use_container_width=True, type = "primary" if st.session_state.page == "home" else "secondary"):
        st.session_state.page = "home"
        st.rerun()
with col2:
    if st.button("Prediction", use_container_width=True, type = "primary" if st.session_state.page == "prediction" else "secondary"):
        st.session_state.page = "prediction"
        st.rerun()
with col3:
    if st.button("Notifications", use_container_width=True, type = "primary" if st.session_state.page == "notifications" else "secondary"):
        st.session_state.page = "notifications"
        st.rerun()


if st.session_state.page == "home":
    home_page()
elif st.session_state.page == "prediction":
    prediction()
elif st.session_state.page == "notifications":
    notifications()


st.markdown("""
    <p style="text-align: center; color: #94a3b8; font-size: 0.8rem; margin-top: 2rem;">
        OnTime Transport Logic © 2026 | Version 1.0
    </p>
""", unsafe_allow_html=True)