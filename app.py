from fastapi import FastAPI
from pydantic import BaseModel
import pickle
import pandas as pd

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


@app.get("/")
def home():
    return {"message": "Customer Churn API Running"}


@app.post("/predict")
def predict(data: CustomerData):

    # Create empty input dictionary
    input_dict = {col: 0 for col in columns}

    # Numeric values
    input_dict["SeniorCitizen"] = data.SeniorCitizen
    input_dict["MonthlyCharges"] = data.MonthlyCharges
    input_dict["TotalCharges"] = data.TotalCharges

    # Categorical columns
    gender_col = f"gender_{data.gender}"
    partner_col = f"Partner_{data.Partner}"
    contract_col = f"Contract_{data.Contract}"
    payment_col = f"PaymentMethod_{data.PaymentMethod}"

    # Safely assign values
    if gender_col in input_dict:
        input_dict[gender_col] = 1

    if partner_col in input_dict:
        input_dict[partner_col] = 1

    if contract_col in input_dict:
        input_dict[contract_col] = 1

    if payment_col in input_dict:
        input_dict[payment_col] = 1

    # Convert to DataFrame
    df = pd.DataFrame([input_dict])

    # Prediction
    prediction = model.predict(df)[0]

    return {"prediction": int(prediction)}