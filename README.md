# 🧠 Mental Health & Social Media Balance Dashboard

### 🎓 Python Data Analysis Final Project  
**Tools:** Streamlit · Plotly · Scikit-learn · Statsmodels  

---

## 🌍 Project Overview

This project analyzes the **relationship between social media usage, sleep quality, stress level, physical activity, and overall happiness**.  
The main goal is to explore behavioral factors affecting **mental wellbeing** and to develop a synthetic indicator called the **Digital Wellbeing Index (DWI)** that quantifies one’s digital-life balance.  

The dashboard provides an interactive, data-driven interface for visual exploration and predictive modeling of happiness levels based on lifestyle habits.

---

## 🧩 Dashboard Sections

| Section | Description | Main Tools |
|----------|--------------|-------------|
| **1️⃣ Dataset Overview** | Preview of dataset, rows/columns, data summary | `pandas`, `streamlit` |
| **2️⃣ EDA Dashboard** | Exploratory analysis, distributions, correlations, clustering, DWI | `plotly.express`, `seaborn` |
| **3️⃣ Machine Learning** | Predicting *Happiness Index* using Linear Regression and Random Forest | `scikit-learn` |
| **4️⃣ Regression Analysis** | Simple OLS regression: *Screen Time → Stress* | `statsmodels` |
| **5️⃣ Insight & Recommendation** | Summarized findings and actionable recommendations | `streamlit markdown` |


---

## ⚙️ Installation & Execution

### 1️⃣ Install required packages
```bash
pip install -r requirements.txt

```
### 2️⃣ Run the Streamlit app
```bash
python -m streamlit run app.py

```

### 3️⃣ Open in browser

Streamlit will open automatically or show a URL like:

Local URL: http://localhost:8501

## 📊 Key Features
**🧭 1. EDA Dashboard**

    Interactive filters by age range and social media platform

    Main visualizations:

    Age and Gender distribution

    Screen Time vs Stress (scatter + regression line)

    Sleep Quality vs Happiness (line chart)

    Exercise Frequency vs Stress (box plot)

    Platform comparison of average Stress & Happiness

    Correlation Heatmap

    Advanced analysis:

    Digital Wellbeing Index (DWI) creation

    Mini heatmap of DWI correlations

    KMeans clustering (3 behavioral groups)

    3D visualization (Sleep – Stress – Happiness)

    Insight summary section

**🤖 2. Machine Learning – Happiness Prediction**

    Data normalization using StandardScaler

    Models implemented:

    Linear Regression

    Random Forest Regressor

    Evaluation metrics:

    MAE, RMSE, R², and Adjusted R²

    Model comparison bar chart

    Error Distribution and Error vs Predicted Plot

    Additional features:

    Interactive user input sliders for live prediction

    Feature importance (Random Forest)

**📈 3. Regression Analysis (OLS)**

    Linear regression model: Stress = β₀ + β₁ × Screen_Time

    Output includes:

    Full statistical summary (coefficients, p-values, R²)

    Regression equation and interpretation

    Scatter plot with fitted line

**💡 4. Insight & Recommendation**

    Summarized insights and recommendations for improving mental wellbeing:

    🔍 Key Findings

    1️⃣ Users spending >8 hours/day on social media have higher stress and lower happiness.
    2️⃣ Good sleep quality (≥7/10) strongly correlates with happiness.
    3️⃣ Exercising ≥3 times/week significantly reduces stress.
    4️⃣ Users of Instagram/YouTube tend to be happier than heavy TikTok/Twitter users.

    🧭 Recommendations

        ⏳ Limit daily social media use to <5 hours.

        😴 Maintain 7–8 hours of sleep each night.

        🏃 Engage in physical activity 3–4 times/week.

        🧘 Practice digital detox periodically.

        ❤️ Follow positive content, avoid toxic media.

    📌 Conclusion

    A balanced lifestyle between Screen Time, Sleep, Stress, and Exercise
    directly enhances happiness and psychological wellbeing.

**📚 Main Insights**

    1️⃣ Excessive screen time is directly linked to higher stress.
    2️⃣ Sleep quality and exercise are strong predictors of happiness.
    3️⃣ Cluster 2 represents the healthiest, most balanced user group.
    4️⃣ Random Forest achieved R² = 0.82, outperforming Linear Regression (R² = 0.69).
    5️⃣ The Error vs Predicted plot confirms no major model bias.

**🧠 Conclusion**

    The analysis confirms that Screen Time, Sleep, Stress, and Exercise
    jointly shape an individual’s Happiness Index.

    The proposed Digital Wellbeing Index provides a meaningful, quantifiable framework
    to assess and promote healthier digital habits for better mental health.
    

### 🛠 Technologies Used
    | Library                           | Purpose                               |
    | --------------------------------- | ------------------------------------- |
    | **Streamlit**                     | Interactive web dashboard             |
    | **Pandas / NumPy**                | Data preprocessing                    |
    | **Plotly / Seaborn / Matplotlib** | Visualization & analytics             |
    | **Scikit-learn**                  | Machine learning, scaling, clustering |
    | **Statsmodels**                   | Statistical regression (OLS)          |


### 🧩 Expected Outcome
    | Aspect            | Strength                          |
    | ----------------- | --------------------------------- |
    | Data Storytelling | Clear, insightful                 |
    | Visualization     | Aesthetic, interactive            |
    | Model Performance | Reliable, explainable             |
    | Statistical Rigor | Includes Adjusted R² + Regression |
    | Practical Impact  | Actionable wellbeing insights     |
