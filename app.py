import streamlit as st
import pandas as pd
from src.data_loader import load_data
from src.eda_visualization import show_eda_dashboard
from src.ml_model import show_ml_section
from src.regression_analysis import show_regression_analysis

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
    ["Giới thiệu", "EDA Dashboard", "Phân tích hồi quy", "Dự đoán Happiness", "Insight & Recommendation"]
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

elif menu == "Phân tích hồi quy":
    show_regression_analysis()

elif menu == "EDA Dashboard":
    show_eda_dashboard()

elif menu == "Dự đoán Happiness":
    show_ml_section()
elif menu == "Insight & Recommendation":
    st.title("💡 Insight & Recommendation")
    st.markdown("""
    ### 🔍 **Những phát hiện chính từ phân tích dữ liệu**
    1️⃣ Người có **thời gian sử dụng mạng xã hội > 8 giờ/ngày** thường có **mức stress cao hơn** và **chỉ số hạnh phúc thấp**.  
    2️⃣ **Giấc ngủ chất lượng cao (≥7/10)** có mối tương quan dương mạnh với **chỉ số hạnh phúc**.  
    3️⃣ **Tập thể dục ≥3 lần/tuần** giúp giảm stress đáng kể.  
    4️⃣ Người dùng **Instagram / YouTube** có xu hướng hạnh phúc cao hơn nhóm dùng **TikTok / Twitter**.  

    ### 🧭 **Khuyến nghị đề xuất**
    - ⏳ Giới hạn thời gian dùng mạng xã hội còn **<5 giờ/ngày**.  
    - 😴 Giữ **giấc ngủ ổn định 7–8 tiếng/đêm**.  
    - 🏃 Tăng **hoạt động thể chất 3–4 lần/tuần** để giảm stress.  
    - 🧘 Thực hành **digital detox** (ngừng dùng mạng xã hội vài ngày/tháng).  
    - ❤️ Ưu tiên dùng mạng xã hội tích cực, tránh tin tiêu cực.  

    ---
    **📌 Kết luận:**  
    Mối quan hệ giữa **thời gian sử dụng mạng xã hội – giấc ngủ – stress – vận động** ảnh hưởng trực tiếp đến **hạnh phúc tinh thần**.  
    Việc cân bằng các yếu tố này giúp người trẻ **nâng cao sức khỏe tâm lý và hiệu suất sống.**
    """)
