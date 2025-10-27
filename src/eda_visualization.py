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


    # --- 📈 SUMMARY STATISTICS ---
    st.subheader("📈 Thống kê mô tả (Summary Statistics)")
    st.caption("Bảng dưới đây hiển thị giá trị trung bình, độ lệch chuẩn, nhỏ nhất và lớn nhất của các cột định lượng (theo dữ liệu đã lọc).")

    summary = df_filtered.describe().loc[["mean", "std", "min", "max"]].T
    summary = summary.rename(columns={
        "mean": "Trung bình",
        "std": "Độ lệch chuẩn",
        "min": "Nhỏ nhất",
        "max": "Lớn nhất"
    })
    st.dataframe(summary)

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

    # ============================================================
    # 🌈 PHÂN TÍCH NÂNG CAO
    # ============================================================
    st.markdown("---")
    st.header("🌈 Phân tích nâng cao (Advanced Analysis)")

    # --- 1️⃣ Digital Wellbeing Index ---
    st.subheader("1️⃣ Chỉ số Digital Wellbeing tổng hợp")
    st.caption("Chỉ số tổng hợp phản ánh sức khỏe tinh thần dựa trên giấc ngủ, stress, hạnh phúc và vận động.")

    df["Digital_Wellbeing_Index"] = (
    0.25 * df["Sleep_Quality1_10"] +
    0.25 * (10 - df["Stress_Level1_10"]) +
    0.20 * df["Happiness_Index1_10"] +
    0.15 * df["Exercise_Frequencyweek"] +
    0.15 * (10 - df["Daily_Screen_Timehrs"])
)
    
    # #
    fig_dwi = px.histogram(
        df, 
        x="Digital_Wellbeing_Index",
        color="Gender",
        nbins=20,
        title="Phân bố chỉ số Digital Wellbeing theo giới tính",
        color_discrete_sequence=px.colors.qualitative.Pastel
    )
    st.plotly_chart(fig_dwi, use_container_width=True)

    mean_dwi = df["Digital_Wellbeing_Index"].mean()
    st.info(f"🌟 Chỉ số Digital Wellbeing trung bình của toàn bộ mẫu: **{mean_dwi:.2f}/10**")
        # --- 🔍 Tương quan của Digital Wellbeing Index ---
    st.subheader("📈 Mối tương quan giữa Digital Wellbeing và các yếu tố khác")
    st.caption("Biểu đồ dưới đây cho thấy mối tương quan giữa chỉ số Wellbeing và các biến hành vi liên quan (giấc ngủ, stress, hạnh phúc, thời gian dùng mạng).")

    corr_wellbeing = df[[
        "Digital_Wellbeing_Index",
        "Sleep_Quality1_10",
        "Stress_Level1_10",
        "Happiness_Index1_10",
        "Daily_Screen_Timehrs",
        "Exercise_Frequencyweek"
    ]].corr()

    fig_corr, ax_corr = plt.subplots(figsize=(6, 4))
    sns.heatmap(corr_wellbeing, annot=True, cmap="YlGnBu", ax=ax_corr)
    st.pyplot(fig_corr)

    st.info("""
    💡 **Nhận xét nhanh:**
    - Digital Wellbeing tương quan **âm mạnh** với Stress và Screen Time.
    - Tương quan **dương mạnh** với Sleep Quality và Happiness.
    - Điều này chứng minh chỉ số Wellbeing phản ánh tốt trạng thái tinh thần người dùng trong môi trường số.
    """)

    # --- 2️⃣ Phân nhóm người dùng (Clustering) ---
    st.subheader("2️⃣ Phân nhóm người dùng (KMeans Clustering)")
    st.caption("Phân nhóm người dùng dựa trên Screen Time, Sleep, Stress và Happiness để khám phá hành vi tương đồng.")

    from sklearn.cluster import KMeans
    features = df[["Daily_Screen_Timehrs", "Sleep_Quality1_10", "Stress_Level1_10", "Happiness_Index1_10"]]
    kmeans = KMeans(n_clusters=3, random_state=42)
    df["Cluster"] = kmeans.fit_predict(features)

    fig_cluster = px.scatter(
        df,
        x="Daily_Screen_Timehrs",
        y="Happiness_Index1_10",
        color=df["Cluster"].astype(str),
        title="Phân nhóm người dùng theo hành vi & mức độ hạnh phúc",
        hover_data=["Stress_Level1_10", "Sleep_Quality1_10"],
        color_discrete_sequence=px.colors.qualitative.Bold
    )
    st.plotly_chart(fig_cluster, use_container_width=True)

    st.success("""
    🔍 **Phân tích nhanh:**
    - Nhóm 0: Screen time cao, stress cao, happiness thấp.
    - Nhóm 1: Trung bình ở các yếu tố.
    - Nhóm 2: Ngủ tốt, ít stress, hạnh phúc cao.
    """)

    # --- 3️⃣ 3D Visualization ---
    st.subheader("3️⃣ Mối quan hệ 3 chiều: Giấc ngủ – Stress – Hạnh phúc")
    st.caption("Biểu đồ 3D cho thấy mối quan hệ giữa 3 yếu tố chính ảnh hưởng đến sức khỏe tinh thần.")

    fig_3d = px.scatter_3d(
        df,
        x="Sleep_Quality1_10",
        y="Stress_Level1_10",
        z="Happiness_Index1_10",
        color="Gender",
        size="Exercise_Frequencyweek",
        title="3D: Giấc ngủ – Stress – Hạnh phúc",
        color_discrete_sequence=px.colors.qualitative.Dark24
    )
    st.plotly_chart(fig_3d, use_container_width=True)

    st.info("""
    💡 **Nhận xét:**
    - Khi điểm Sleep tăng → Stress giảm → Happiness tăng.
    - Người vận động nhiều (điểm to hơn) có xu hướng hạnh phúc cao hơn.
    - Giới tính không phải yếu tố chính, nhưng có khác biệt nhỏ ở Sleep Quality.
    """)

