# 📉 Customer Churn Prediction System (End-to-End ML Project)

An end-to-end **Machine Learning customer churn prediction system** built using  
**Python, scikit-learn, SHAP, and Streamlit**.

This project covers the **full ML lifecycle**:
data exploration → feature engineering → model training → explainability → deployment.

---

## 🚀 Live Features

✔ Upload customer data (CSV)  
✔ Predict churn probability  
✔ Batch predictions  
✔ Model explainability using **SHAP**  
✔ Interactive **Streamlit web app**

---

## 🧠 Problem Statement

Customer churn is a major challenge for subscription-based businesses.  
This project predicts **whether a customer is likely to churn**, enabling businesses to take **preventive actions**.

---

## 🏗️ Project Architecture

customer-churn-ml/
│
├── app/ # Streamlit application
│ └── app.py
│
├── notebooks/ # ML pipeline notebooks
│ ├── 01_eda.ipynb
│ ├── 02_feature_engineering.ipynb
│ ├── 03_model_training.ipynb
│ └── 04_shap_explainability.ipynb
│
├── data/ # Dataset (CSV)
├── artifacts/ # Models, preprocessors, plots
├── screenshots/ # App screenshots
├── README.md
└── requirements.txt


---

## 📊 Dataset

- **Source:** Telecom Customer Churn dataset  
- **Target:** `Churn` (Yes / No)
- **Features include:**
  - Demographics (gender, senior citizen)
  - Account info (tenure, contract)
  - Services (Internet, Streaming, Tech Support)
  - Billing (MonthlyCharges, PaymentMethod)

⚠️ **Input CSV must contain the same feature columns used during training.**

---

## 🧪 Models Used

- Logistic Regression  
- Random Forest Classifier (final selected model)

### Model Evaluation
- ROC-AUC
- Precision / Recall
- Confusion Matrix

---

## 🔍 Model Explainability (SHAP)

The project uses **SHAP (SHapley Additive exPlanations)** to explain:
- Global feature importance
- Local predictions per customer

This makes predictions **transparent and trustworthy**.

---

## 🖥️ Streamlit Web App

### Features
- Upload CSV file
- Preview data
- Predict churn probability
- Download predictions

### Example Screenshots

![Upload CSV](screenshots/Screenshot%20(623).png)
![Predictions](screenshots/Screenshot%20(624).png)
![SHAP Explainability](screenshots/Screenshot%20(625).png)

---

## ▶️ How to Run Locally

### 1️⃣ Install dependencies

pip install -r requirements.txt
### 2️⃣ Run Streamlit app
streamlit run app/app.py
###3️⃣ Open browser
http://localhost:8501
🛠️ Tech Stack

Python
pandas / numpy
scikit-learn
SHAP
matplotlib / seaborn
Streamlit
Git & GitHub
