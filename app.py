from fastapi import FastAPI
from pydantic import BaseModel
import pickle
import pandas as pd

# Initialize app
app = FastAPI()

# Load model and columns
model = pickle.load(open("churn_model.pkl", "rb"))
columns = pickle.load(open("columns.pkl", "rb"))

# Input schema
class CustomerData(BaseModel):
    SeniorCitizen: int
    MonthlyCharges: float
    TotalCharges: float
    gender: str
    Partner: str
    Contract: str
    PaymentMethod: str

# Home route
@app.get("/")
def home():
    return {"message": "Customer Churn API Running"}

# Safe feature setter (prevents crash)
def set_feature(input_dict, feature):
    if feature in input_dict:
        input_dict[feature] = 1

# Prediction route
@app.post("/predict")
def predict(data: CustomerData):

    # Create input dictionary with all 0s
    input_dict = dict.fromkeys(columns, 0)

    # Numeric features
    input_dict["SeniorCitizen"] = data.SeniorCitizen
    input_dict["MonthlyCharges"] = data.MonthlyCharges
    input_dict["TotalCharges"] = data.TotalCharges

    # User input features (safe)
    set_feature(input_dict, f"gender_{data.gender}")
    set_feature(input_dict, f"Partner_{data.Partner}")
    set_feature(input_dict, f"Contract_{data.Contract}")
    set_feature(input_dict, f"PaymentMethod_{data.PaymentMethod}")

    # Default important features (to improve prediction)
    set_feature(input_dict, "InternetService_Fiber optic")
    set_feature(input_dict, "TechSupport_No")
    set_feature(input_dict, "OnlineSecurity_No")
    set_feature(input_dict, "PaperlessBilling_Yes")
    set_feature(input_dict, "tenure_group_1 - 12")

    # Convert to DataFrame
    df = pd.DataFrame([input_dict])

    # Prediction
    prediction = model.predict(df)[0]

    return {"prediction": int(prediction)}