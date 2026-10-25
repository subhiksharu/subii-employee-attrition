# 🧭 subii

### 🤖 Employee Turnover & Early Risk Prediction System

**subii** uses **Machine Learning** to predict whether an employee may leave an organization and to identify their **turnover risk**.

---

## ✨ Features

👤 **Employee Prediction** → Predict turnover risk for an employee.

⚠️ **Risk Assessment** → Shows the employee's turnover risk level.

🔍 **Risk Factors** → Identifies important factors related to turnover.

📁 **Bulk Analysis** → Upload a CSV file and predict multiple employees.

📊 **Data Insights** → Explore employee data using charts.

📈 **Model Performance** → View the performance of the machine learning models.

---

## 🤖 Machine Learning Models

🌲 Random Forest
🌳 Decision Tree
📍 K-Nearest Neighbors (KNN)
📈 Logistic Regression

🔗 These models are combined using a **Stacking Classifier** for the final prediction.

---

## ⚙️ Data Processing

🧹 Data Cleaning → 📊 Outlier Handling (IQR) → 🔤 Categorical Encoding → 📏 Feature Scaling → 🤖 Model Training → 🎯 Prediction

---

## 📏 Evaluation Metrics

✅ Accuracy ✅ Precision ✅ Recall ✅ F1-Score ✅ ROC-AUC ✅ Confusion Matrix

---

## 📂 Dataset

📄 **File:** `HR-Employee-Attrition-New.csv`
📌 **Records:** 1,470 · **Columns:** 35
🎯 **Target:** `Attrition` (🟢 No → stays, 🔴 Yes → leaves)

---

## 🛠️ Technologies Used

🐍 Python · 🎈 Streamlit · 🤖 Scikit-learn · 🐼 Pandas · 🔢 NumPy · 📊 Plotly

---

## 📁 Project Structure

```text
📦 subii
│
├── 🐍 subii_app.py
├── 🤖 turnover_model.py
├── 📊 HR-Employee-Attrition-New.csv
├── 📦 requirements.txt
├── 📖 README.md
│
└── ⚙️ .streamlit/
    └── config.toml
```

---

## ▶️ Run the Project

```bash
pip install -r requirements.txt
streamlit run subii_app.py
```

---

## 🎯 Project Goal

💡 **Predict → Analyze → Assess → Support Early Intervention**

Identify potential employee turnover risk early and provide useful insights through a simple web application.

---

**subii — Made with ❤️ using Python, Machine Learning & Streamlit**
