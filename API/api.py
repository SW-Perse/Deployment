import uvicorn
from fastapi import FastAPI
app = FastAPI(debug=True)
import mlflow.sklearn
import mlflow.pyfunc
import pandas as pd
from pydantic import BaseModel
import requests
import joblib

description = """
Welcome to the GetAround API.
"""

tags_metadata = [
    {
        "name": "Default",
        "description": "Default endpoint"
    },

    {
        "name": "Trained linear regression",
        "description": "Predict optimum rental price",
    }
]



app = FastAPI(
    title="GetAround",
    description=description,
    openapi_tags=tags_metadata
)

@app.get("/", tags=["Default"])
async def index():

    message = "Hello world! This `/` is the most simple and default endpoint. If you want to learn more, check out documentation of the api at `/docs`"

    return message

class predictionFeatures(BaseModel):
    model_key: str
    mileage: int
    engine_power: int
    fuel: str
    paint_color: str
    car_type: str
    private_parking_available: bool
    has_gps: bool
    has_air_conditioning: bool
    automatic_car: bool
    has_getaround_connect: bool
    has_speed_regulator: bool
    winter_tires: bool

@app.post("/predict", tags=["Machine-Learning"])
async def predict(input_pred: predictionFeatures):
     """
    Predicts daily rental price based on car features

    Example request : 
    {
    "model_key": "Audi",  
    "mileage": 48735,  
    "engine_power": 160,  
    "fuel": "diesel",  
    "paint_color": "silver",  
    "car_type": "sedan",  
    "private_parking_available": False,  
    "has_gps": False,  
    "has_air_conditioning": False,  
    "automatic_car": False,  
    "has_getaround_connect": False,  
    "has_speed_regulator": False,  
    "winter_tires": False  
    }

    """
     car_data = pd.DataFrame(dict(input_pred), index=[0])
     
     loaded_model = joblib.load("model.pkl")
     
     prediction = loaded_model.predict(car_data)
     response = {"prediction": prediction.tolist()[0]}
     return response

if __name__=="__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)