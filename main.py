import joblib
import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()

model = joblib.load("model.joblib")

class DiabetesFeatures(BaseModel):
    age: int
    gender: int
    polyuria: int
    polydipsia: int
    sudden_weight_loss: int
    weakness: int
    polyphagia: int
    genital_thrush: int
    visual_blurring: int
    itching: int
    irritability: int
    delayed_healing: int
    partial_paresis: int
    muscle_stiffness: int
    alopecia: int
    obesity: int

@app.post("/predict")
def predict(data: DiabetesFeatures):

    features = np.array([[
        data.age,
        data.gender,
        data.polyuria,
        data.polydipsia,
        data.sudden_weight_loss,
        data.weakness,
        data.polyphagia,
        data.genital_thrush,
        data.visual_blurring,
        data.itching,
        data.irritability,
        data.delayed_healing,
        data.partial_paresis,
        data.muscle_stiffness,
        data.alopecia,
        data.obesity
    ]])

    prediction = model.predict(features)[0]
    probability = model.predict_proba(features)[0][1]

    return {
        "prediction": int(prediction),
        "probability": float(probability),
        "result": "Diabetic" if prediction == 1 else "Not Diabetic"
    }