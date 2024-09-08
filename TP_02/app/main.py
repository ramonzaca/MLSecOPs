from classes import CombinedAttributesAdder
from fastapi import FastAPI
from model import load_model, predict
from pydantic import BaseModel

import __main__

__main__.CombinedAttributesAdder = CombinedAttributesAdder

model = None

app = FastAPI()


class InputData(BaseModel):
    features: list[list[float | str]]


@app.on_event("startup")
async def load_predictive_model():
    global model
    model = load_model("./models/TP_01_model.pkl")


@app.post("/predict")
async def get_prediction(data: InputData):
    prediction = predict(model, data.features)
    return {"prediction": prediction.tolist()}


@app.get("/")
async def root():
    return {"message": "Welcome to the TP2 ML Model API"}
