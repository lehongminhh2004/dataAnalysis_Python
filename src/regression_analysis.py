import streamlit as st
import statsmodels.api as sm
import plotly.express as px
from src.data_loader import load_data

def show_regression_analysis():
    st.title("📈 Phân tích hồi quy – Mối liên hệ giữa Screen Time và Stress")

    df = load_data("data/Mental_Health_and_Social_Media_Balance_Dataset.csv")

    # Biến độc lập và phụ thuộc
    X = sm.add_constant(df["Daily_Screen_Timehrs"])
    y = df["Stress_Level1_10"]

    model = sm.OLS(y, X).fit()

    st.subheader("📘 Kết quả hồi quy tuyến tính")
    st.write(model.summary())

    # Hiển thị biểu đồ scatter + đường hồi quy
    fig = px.scatter(df, x="Daily_Screen_Timehrs", y="Stress_Level1_10", trendline="ols",
                     title="Ảnh hưởng của thời gian dùng mạng xã hội đến mức độ stress")
    st.plotly_chart(fig, use_container_width=True)

    # Tóm tắt phương trình
    intercept = model.params[0]
    slope = model.params[1]
    r2 = model.rsquared

    st.markdown(f"""
    ### 🔍 Phương trình hồi quy:
    **Stress = {intercept:.2f} + {slope:.2f} × Screen_Time**
    
    ### 📊 Hệ số tương quan (R²): {r2:.3f}
    """)
