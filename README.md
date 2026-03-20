#  Customer Churn Prediction System

##  Project Overview
This project is an end-to-end Machine Learning system that predicts whether a customer is likely to churn or not. It includes data preprocessing, model training, API development, and deployment.

The model is exposed through a FastAPI-based REST API and deployed on Render for real-time predictions.

---

##  Problem Statement
Customer churn is a major challenge for telecom companies. This project aims to identify customers who are likely to leave the service so that proactive retention strategies can be applied.

---

##  Features
- Data Cleaning and Preprocessing
- Feature Engineering (One-hot encoding)
- Handling Class Imbalance
- Model Training using Random Forest
- Model Evaluation (Accuracy, Precision, Recall)
- REST API using FastAPI
- Deployment on Render (Live API)

---

##  Tech Stack
- Python
- Pandas, NumPy
- Scikit-learn
- FastAPI
- Uvicorn
- Git & GitHub
- Render (Deployment)

---

##  Dataset
- Telecom Customer Churn Dataset
- ~7000+ records
- Features include demographics, services, billing, and tenure

---

##  How It Works

1. Data preprocessing and feature engineering
2. Model training using Random Forest
3. Model saved as `.pkl` file
4. FastAPI loads model and serves predictions
5. Deployed on Render for real-time access

---

##  Live API

👉 Try it here:  
https://customer-churn-api-b37p.onrender.com/docs

---

##  Sample Input

```json
{
  "SeniorCitizen": 1,
  "MonthlyCharges": 100,
  "TotalCharges": 500,
  "gender": "Female",
  "Partner": "No",
  "Contract": "Month-to-month",
  "PaymentMethod": "Electronic check"
}
