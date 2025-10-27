import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from src.data_loader import load_data
import numpy as np

def show_ml_section():
    st.title("🤖 Machine Learning – Dự đoán chỉ số Happiness")

    # --- Load dữ liệu ---
    df = load_data("data/Mental_Health_and_Social_Media_Balance_Dataset.csv")

    # --- Chọn các đặc trưng (features) và nhãn (target) ---
    X = df[["Daily_Screen_Timehrs", "Sleep_Quality1_10", "Stress_Level1_10", "Exercise_Frequencyweek"]]
    y = df["Happiness_Index1_10"]

    # --- Chuẩn hóa dữ liệu ---
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # --- Chia dữ liệu ---
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

    # --- Chọn mô hình ---
    model_type = st.radio("Chọn mô hình dự đoán:", ["Linear Regression", "Random Forest"])
    if model_type == "Linear Regression":
        model = LinearRegression()
    else:
        model = RandomForestRegressor(random_state=42, n_estimators=200)

    # --- Huấn luyện ---
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # --- Đánh giá mô hình ---
    mae = mean_absolute_error(y_test, y_pred)
    rmse = mean_squared_error(y_test, y_pred) ** 0.5
    r2 = r2_score(y_test, y_pred)
    n = len(y_test)
    p = X_test.shape[1]
    adj_r2 = 1 - (1 - r2) * (n - 1) / (n - p - 1)  # Adjusted R²

    st.markdown("### 📊 Hiệu suất mô hình")
    col1, col2, col3 = st.columns(3)
    col1.metric("MAE", f"{mae:.3f}")
    col2.metric("RMSE", f"{rmse:.3f}")
    col3.metric("R²", f"{r2:.3f}")

    col4, col5 = st.columns(2)
    col4.metric("Adj R²", f"{adj_r2:.3f}")
    col5.metric("Samples", f"{n}")

    # --- So sánh Linear vs Random Forest ---
    st.markdown("### ⚖️ So sánh hiệu suất các mô hình")
    models = {
        "Linear Regression": LinearRegression(),
        "Random Forest": RandomForestRegressor(random_state=42, n_estimators=200)
    }

    results = []
    for name, m in models.items():
        m.fit(X_train, y_train)
        y_pred_model = m.predict(X_test)
        results.append({
            "Mô hình": name,
            "R²": r2_score(y_test, y_pred_model),
            "RMSE": mean_squared_error(y_test, y_pred_model) ** 0.5,
            "MAE": mean_absolute_error(y_test, y_pred_model)
        })

    results_df = pd.DataFrame(results)
    fig_compare = px.bar(
        results_df,
        x="Mô hình",
        y="R²",
        color="Mô hình",
        text=results_df["R²"].apply(lambda x: f"{x:.2f}"),
        title="So sánh R² giữa Linear Regression và Random Forest",
        color_discrete_sequence=px.colors.qualitative.Set2
    )
    fig_compare.update_traces(textposition='outside')
    st.plotly_chart(fig_compare, use_container_width=True)

    # --- Biểu đồ so sánh thực tế vs dự đoán ---
    st.markdown("### 🔍 So sánh giá trị thực tế và dự đoán")
    compare_df = pd.DataFrame({"Thực tế": y_test, "Dự đoán": y_pred})
    fig1 = px.scatter(compare_df, x="Thực tế", y="Dự đoán", trendline="ols",
                      title="Biểu đồ: Thực tế vs Dự đoán Happiness")
    fig1.add_trace(go.Scatter(x=[y.min(), y.max()], y=[y.min(), y.max()],
                              mode='lines', name='Đường lý tưởng', line=dict(color='red', dash='dash')))
    st.plotly_chart(fig1, use_container_width=True)

    # --- Biểu đồ phân bố lỗi ---
    st.markdown("### ⚠️ Phân tích lỗi dự đoán (Error Distribution)")
    errors = y_test - y_pred
    fig2 = px.histogram(errors, nbins=20, title="Phân bố sai số giữa thực tế và dự đoán",
                        color_discrete_sequence=['#E45756'])
    st.plotly_chart(fig2, use_container_width=True)

    # --- Biểu đồ Error vs Predicted ---
    st.markdown("### ⚖️ Phân tích lỗi so với giá trị dự đoán (Error vs Predicted)")
    error_df = pd.DataFrame({"Predicted": y_pred, "Error": errors})
    fig_err = px.scatter(
        error_df, x="Predicted", y="Error", trendline="ols",
        title="Mối quan hệ giữa sai số và giá trị dự đoán",
        color_discrete_sequence=["#4C78A8"]
    )
    st.plotly_chart(fig_err, use_container_width=True)
    st.caption("""
    💡 Nếu các điểm phân bố đều quanh trục 0 → mô hình **không bị bias**.  
    Nếu có xu hướng rõ rệt → mô hình **thiếu biến hoặc cần phi tuyến**.
    """)

    # --- Dự đoán thử với input người dùng ---
    st.markdown("### 🎯 Dự đoán chỉ số Happiness cho người dùng mới")
    c1, c2, c3, c4 = st.columns(4)
    screen_time = c1.slider("Screen Time (giờ/ngày)", 0.0, 12.0, 4.0, 0.1)
    sleep_quality = c2.slider("Sleep Quality (1-10)", 1, 10, 7)
    stress = c3.slider("Stress Level (1-10)", 1, 10, 5)
    exercise = c4.slider("Exercise Frequency (lần/tuần)", 0, 7, 3)

    if st.button("🔮 Dự đoán"):
        input_scaled = scaler.transform([[screen_time, sleep_quality, stress, exercise]])
        prediction = model.predict(input_scaled)[0]
        st.success(f"💡 Happiness dự đoán: **{prediction:.2f}/10**")

        if prediction < 5:
            st.warning("⚠️ Cảnh báo: Chỉ số hạnh phúc thấp – có thể chịu tác động tiêu cực từ stress hoặc thiếu ngủ.")
        elif prediction < 7:
            st.info("🙂 Mức hạnh phúc trung bình – có thể cải thiện bằng giảm stress hoặc tăng vận động.")
        else:
            st.success("🌈 Mức hạnh phúc cao – lối sống cân bằng và tích cực!")

    # --- Tầm quan trọng yếu tố (chỉ với Random Forest) ---
    if model_type == "Random Forest":
        st.markdown("### 🔬 Tầm quan trọng của các yếu tố (Feature Importance)")
        importance = model.feature_importances_
        fig_imp = go.Figure(go.Bar(
            x=importance,
            y=X.columns,
            orientation='h',
            text=[f"{v:.2%}" for v in importance],
            textposition="auto"
        ))
        fig_imp.update_layout(title="🎯 Mức ảnh hưởng của từng yếu tố đến Happiness")
        st.plotly_chart(fig_imp, use_container_width=True)
