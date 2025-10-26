import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
from src.data_loader import load_data

def show_eda_dashboard():
    st.title("📊 EDA Dashboard – Phân tích dữ liệu")
    st.markdown("""
    Phần này hiển thị các biểu đồ và thống kê mô tả để tìm hiểu mối quan hệ giữa **sử dụng mạng xã hội**, 
    **giấc ngủ**, **stress**, **vận động** và **chỉ số hạnh phúc**.
    """)

    # --- LOAD DATA ---
    df = load_data("data/Mental_Health_and_Social_Media_Balance_Dataset.csv")

    # --- SIDEBAR FILTERS ---
    st.sidebar.subheader("⚙️ Bộ lọc tương tác")
    age_range = st.sidebar.slider("Chọn độ tuổi", int(df["Age"].min()), int(df["Age"].max()), (20, 40))
    platform = st.sidebar.selectbox("Chọn nền tảng mạng xã hội", df["Social_Media_Platform"].unique())
    df_filtered = df[(df["Age"] >= age_range[0]) & (df["Age"] <= age_range[1]) & (df["Social_Media_Platform"] == platform)]

    st.write(f"Hiển thị {len(df_filtered)} bản ghi phù hợp với bộ lọc.")

    # --- 1️⃣ Phân bố độ tuổi & giới tính ---
    st.subheader("1️⃣ Phân bố độ tuổi và giới tính")
    col1, col2 = st.columns(2)

    with col1:
        fig1 = px.histogram(df_filtered, x="Age", nbins=15, color="Gender", title="Phân bố độ tuổi")
        st.plotly_chart(fig1, use_container_width=True)
    with col2:
        gender_counts = df_filtered["Gender"].value_counts()
        fig2 = px.pie(values=gender_counts.values, names=gender_counts.index, title="Tỷ lệ giới tính")
        st.plotly_chart(fig2, use_container_width=True)

    # --- 2️⃣ Tương quan giữa thời gian dùng mạng & stress ---
    st.subheader("2️⃣ Tương quan giữa thời gian dùng mạng và mức độ stress")
    fig3 = px.scatter(df_filtered,
                      x="Daily_Screen_Timehrs",
                      y="Stress_Level1_10",
                      color="Gender",
                      trendline="ols",
                      title="Ảnh hưởng của thời gian sử dụng mạng xã hội đến stress")
    st.plotly_chart(fig3, use_container_width=True)

    # --- 3️⃣ Mối quan hệ giữa giấc ngủ & hạnh phúc ---
    st.subheader("3️⃣ Mối quan hệ giữa giấc ngủ và chỉ số hạnh phúc")
    fig4 = px.line(df_filtered.sort_values("Sleep_Quality1_10"),
                   x="Sleep_Quality1_10",
                   y="Happiness_Index1_10",
                   markers=True,
                   title="Giấc ngủ ảnh hưởng thế nào đến hạnh phúc?")
    st.plotly_chart(fig4, use_container_width=True)

    # --- 4️⃣ Ảnh hưởng vận động đến stress ---
    st.subheader("4️⃣ Ảnh hưởng của tần suất vận động đến mức độ stress")
    fig5 = px.box(df_filtered,
                  x="Exercise_Frequencyweek",
                  y="Stress_Level1_10",
                  color="Gender",
                  title="Phân bố stress theo tần suất vận động")
    st.plotly_chart(fig5, use_container_width=True)

    # --- 5️⃣ So sánh nền tảng mạng xã hội ---
    st.subheader("5️⃣ So sánh giữa các nền tảng mạng xã hội")
    avg_df = df.groupby("Social_Media_Platform")[["Happiness_Index1_10", "Stress_Level1_10"]].mean().reset_index()
    fig6 = px.bar(avg_df,
                  x="Social_Media_Platform",
                  y=["Happiness_Index1_10", "Stress_Level1_10"],
                  barmode="group",
                  title="So sánh trung bình Stress & Happiness giữa các nền tảng")
    st.plotly_chart(fig6, use_container_width=True)

    # --- Correlation Heatmap ---
    st.subheader("🔍 Ma trận tương quan (Heatmap)")
    corr_cols = ["Sleep_Quality1_10", "Stress_Level1_10", "Daily_Screen_Timehrs", "Happiness_Index1_10"]
    corr = df[corr_cols].corr()
    fig, ax = plt.subplots(figsize=(6,4))
    sns.heatmap(corr, annot=True, cmap="coolwarm", ax=ax)
    st.pyplot(fig)
