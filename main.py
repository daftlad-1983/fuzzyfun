from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
from logistic import LogisticReg
import json
from typing import Union
import numpy as np

app = FastAPI()

with open('raisin_model.json', 'r') as f:
    reloaded_data = json.load(f)

re_loaded_mod = LogisticReg()
re_loaded_mod.load(**reloaded_data)

class Xdata(BaseModel):
    Area: list[float]
    MajorAxisLength: list[float]
    MinorAxisLength: list[float]
    Eccentricity: list[float]
    ConvexArea: list[float]
    Extent: list[float]
    Perimeter: list[float]

class PredictionResponse(BaseModel):
    predictions: list[list[float]]
    q_values: list[list[float]]

@app.get("/")
def read_root():
    return {"Hello": "World"}

@app.post("/predict/")
def predict_data(data: Xdata) -> PredictionResponse:

    df = pd.DataFrame(data.model_dump())

    predictions = re_loaded_mod.predict(df.to_numpy(), return_q=True, binary=False)

    response = PredictionResponse(predictions=predictions['predictions'].tolist(),
                                  q_values=predictions['q_values'].tolist())

    return response

