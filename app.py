import streamlit as st
import pandas as pd
from src.data_loader import load_data
from src.eda_visualization import show_eda_dashboard
from src.ml_model import show_ml_section

# --- PAGE CONFIG ---
st.set_page_config(
    page_title="Digital Balance Dashboard",
    page_icon="📱",
    layout="wide"
)

# --- SIDEBAR ---
st.sidebar.title("📊 Dashboard Menu")
menu = st.sidebar.radio(
    "Chọn phần hiển thị:",
    ["Giới thiệu", "EDA Dashboard", "Dự đoán Happiness"]
)

# --- MAIN CONTENT ---
if menu == "Giới thiệu":
    st.title("🧠 Mental Health & Social Media Balance")
    st.markdown("""
    ### 🎯 Mục tiêu
    Phân tích mối quan hệ giữa việc sử dụng mạng xã hội, chất lượng giấc ngủ, stress, vận động và mức độ hạnh phúc.
    
    **Dữ liệu gồm các trường:**
    - Age, Gender, Daily_Screen_Time, Sleep_Quality, Stress_Level, Days_Without_Social_Media, Exercise_Frequency, Platform, Happiness_Index
    """)
    
    df = load_data("data/Mental_Health_and_Social_Media_Balance_Dataset.csv")
    st.dataframe(df.head())
    st.write(f"📦 Tổng số dòng: {df.shape[0]}, Cột: {df.shape[1]}")

elif menu == "EDA Dashboard":
    show_eda_dashboard()

elif menu == "Dự đoán Happiness":
    show_ml_section()
