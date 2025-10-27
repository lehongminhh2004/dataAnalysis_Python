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
menu = st.sidebar.radio("📍 Chọn nội dung:", 
                        [
        "Dataset Overview",
        "EDA Dashboard",
        "Machine Learning",
        "Regression Analysis",
        "Insight & Recommendation"
    ]
)

# --- MAIN CONTENT ---
if menu == "Dataset Overview":
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

elif menu == "Regression Analysis":
    show_regression_analysis()

elif menu == "Machine Learning":
    show_ml_section()
elif menu == "Insight & Recommendation":
    st.title("📄 Tổng kết & Khuyến nghị")

    st.header("🔍 Những phát hiện chính từ phân tích dữ liệu")
    st.markdown("""
    1️⃣ Người có **thời gian sử dụng mạng xã hội > 8 giờ/ngày** thường có **mức stress cao hơn** và **chỉ số hạnh phúc thấp**.  
    2️⃣ **Giấc ngủ chất lượng cao (≥7/10)** có mối tương quan dương mạnh với **chỉ số hạnh phúc**.  
    3️⃣ **Tập thể dục ≥3 lần/tuần** giúp giảm stress đáng kể.  
    4️⃣ Người dùng **Instagram / YouTube** có xu hướng hạnh phúc cao hơn nhóm dùng **TikTok / Twitter**.
    """)

    st.header("🧭 Khuyến nghị đề xuất")
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("""
        - ⏳ Giới hạn thời gian dùng mạng xã hội còn **<5 giờ/ngày**  
        - 😴 Duy trì **giấc ngủ 7–8 tiếng mỗi đêm**  
        - 🏃 Tăng **hoạt động thể chất 3–4 lần/tuần**
        """)
    with c2:
        st.markdown("""
        - 🧘 Thực hành **digital detox** định kỳ  
        - ❤️ Ưu tiên sử dụng mạng xã hội tích cực  
        - 🚀 Tăng cường tương tác ngoài đời thực  
        """)

    st.markdown("---")
    st.subheader("📌 Kết luận cuối cùng")
    st.success("""
    Mối quan hệ giữa **Screen Time – Sleep – Stress – Exercise** ảnh hưởng trực tiếp đến **Happiness**.  
    Cân bằng các yếu tố này giúp nâng cao **sức khỏe tinh thần** và **chất lượng cuộc sống**.
    """)

