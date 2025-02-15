from fastapi import FastAPI
import uvicorn
from pydantic import BaseModel
import numpy as np
import torch
import pandas as pd
import joblib
from fastapi.responses import JSONResponse
import config

app = FastAPI()

@app.get("/")
async def root():
    return {"message": "Hello World"}

# Model loading (assuming these paths are set in your config.py)
from model import load_model

model = load_model(model_path=config.MODEL_PATH)

target_encoder = joblib.load(config.ENCODER_PATH)
scaler_x = joblib.load(config.SCALER_X_PATH)
scaler_y = joblib.load(config.SCALER_Y_PATH)


class CarInput(BaseModel):
    year: int
    manufacturer: str
    model: str
    fuel: str
    odometer: int
    title_status: str
    transmission: str
    type: str
    paint_color: str
    state: str


@app.post("/predict")
async def predict_price(input_data: CarInput):

    try:
        # Preprocessing
        input_df = pd.DataFrame([input_data.dict()])
        input_data_encoded = target_encoder.transform(input_df)
        input_data_scaled = scaler_x.transform(input_data_encoded)
        input_tensor = torch.from_numpy(input_data_scaled.astype(np.float32))

        # Prediction
        with torch.no_grad():
            predictions = model(input_tensor).numpy()
        predicted_price = scaler_y.inverse_transform(predictions)

        # Return JSON response
        return JSONResponse(content={"predicted_price": predicted_price.flatten().tolist()})

    except Exception as e:
        # Handle exceptions gracefully (e.g., missing data, model errors)
        return JSONResponse(content={"error": str(e)}, status_code=400)