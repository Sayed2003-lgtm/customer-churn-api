@app.post("/predict")
def predict(data: InputData):

    input_dict = dict.fromkeys(columns, 0)

    # numeric
    input_dict["SeniorCitizen"] = data.SeniorCitizen
    input_dict["MonthlyCharges"] = data.MonthlyCharges
    input_dict["TotalCharges"] = data.TotalCharges

    # gender
    input_dict[f"gender_{data.gender}"] = 1

    # partner
    input_dict[f"Partner_{data.Partner}"] = 1

    # contract
    input_dict[f"Contract_{data.Contract}"] = 1

    # payment method
    input_dict[f"PaymentMethod_{data.PaymentMethod}"] = 1

    # ADD DEFAULT IMPORTANT FEATURES (VERY IMPORTANT)
    input_dict["InternetService_Fiber optic"] = 1
    input_dict["TechSupport_No"] = 1
    input_dict["OnlineSecurity_No"] = 1
    input_dict["PaperlessBilling_Yes"] = 1
    input_dict["tenure_group_1 - 12"] = 1

    df = pd.DataFrame([input_dict])

    prediction = model.predict(df)[0]

    return {"prediction": int(prediction)}