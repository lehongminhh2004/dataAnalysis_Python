import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from src.data_loader import load_data

def show_ml_section():
    st.title("🤖 Dự đoán Chỉ số Hạnh phúc (ML Model)")
    df = load_data("data/Mental_Health_and_Social_Media_Balance_Dataset.csv")

    # --- FEATURE & TARGET ---
    X = df[["Daily_Screen_Timehrs", "Sleep_Quality1_10", "Stress_Level1_10", "Exercise_Frequencyweek"]]
    y = df["Happiness_Index1_10"]

    # --- SPLIT DATA ---
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # --- MODEL SELECTION ---
    model_type = st.radio("Chọn mô hình:", ["Linear Regression", "Random Forest"])
    if model_type == "Linear Regression":
        model = LinearRegression()
    else:
        model = RandomForestRegressor(random_state=42, n_estimators=100)

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # --- METRICS ---
    st.write("### 📊 Kết quả đánh giá mô hình")
    st.write(f"**MAE:** {mean_absolute_error(y_test, y_pred):.3f}")
    st.write(f"**R²:** {r2_score(y_test, y_pred):.3f}")

    # --- USER INPUT ---
    st.write("### 🧩 Nhập giá trị để dự đoán Happiness")
    c1, c2, c3, c4 = st.columns(4)
    screen_time = c1.slider("Screen Time (hrs)", 0.0, 12.0, 4.0, 0.1)
    sleep_quality = c2.slider("Sleep Quality (1-10)", 1, 10, 7)
    stress = c3.slider("Stress Level (1-10)", 1, 10, 5)
    exercise = c4.slider("Exercise Frequency (per week)", 0, 7, 3)

    if st.button("🔮 Dự đoán"):
        input_data = pd.DataFrame({
            "Daily_Screen_Timehrs": [screen_time],
            "Sleep_Quality1_10": [sleep_quality],
            "Stress_Level1_10": [stress],
            "Exercise_Frequencyweek": [exercise]
        })
        pred = model.predict(input_data)[0]
        st.success(f"💡 Dự đoán Happiness_Index ≈ **{pred:.2f}/10**")
        # --- Hiển thị tầm quan trọng yếu tố nếu dùng Random Forest ---
    if model_type == "Random Forest":
        import plotly.graph_objects as go
        importance = model.feature_importances_
        fig_imp = go.Figure(go.Bar(
            x=importance,
            y=X.columns,
            orientation='h'
        ))
        fig_imp.update_layout(title="🎯 Tầm quan trọng của các yếu tố ảnh hưởng đến Happiness")
        st.plotly_chart(fig_imp, use_container_width=True)
    