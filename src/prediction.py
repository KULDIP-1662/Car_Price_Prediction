import torch
import numpy as np
import joblib
import config
import pandas as pd
from model import Model1  
from logger import get_logger
import streamlit as st

class CarPricePredictor:
    def __init__(self):
        self.logger = get_logger(__name__)
        # Load the model, scaler, and encoder from the fixed paths in the config file
        self.model = Model1(10)  # Ensure the input size matches your data
        # Load the state_dict (weights only)
        self.model.load_state_dict(torch.load(config.MODEL_PATH,weights_only=True))  
        self.model.eval() 
        self.scaler_x = joblib.load(config.SCALER_X_PATH)
        self.scaler_y = joblib.load(config.SCALER_Y_PATH)
        self.target_encoder = joblib.load(config.ENCODER_PATH)

    def preprocess_data(self, input_data):
        self.logger.info("--------------Prediction Phase-----------")
        self.logger.info("Preprocessing started....")

        if isinstance(input_data, list):
            input_data = pd.DataFrame([input_data])  # Convert list to DataFrame
        elif isinstance(input_data, dict):
            input_data = pd.DataFrame([input_data])  # Convert dictionary to DataFrame

        # Apply target encoding to categorical columns (transform with target encoder)
        input_data = self.target_encoder.transform(input_data)

        # Scale the input data using scaler_x
        input_data_scaled = self.scaler_x.transform(input_data)

        return input_data_scaled

    def predict(self, input_data):
        self.logger.info("Prediction running....")
        # Preprocess the input data
        processed_data = self.preprocess_data(input_data)

        # Convert to tensor
        input_tensor = torch.from_numpy(processed_data.astype(np.float32))

        # Make prediction
        self.model.eval()
        with torch.no_grad():
            prediction = self.model(input_tensor)

        # Rescale prediction to original scale using scaler_y
        predicted_price = self.scaler_y.inverse_transform(prediction.numpy())
        self.logger.info(f"Predicted Price is : {predicted_price[0][0]}")
        return predicted_price[0][0]

st.title('Car Price Predictor')

state = st.selectbox(
        "# Which states's car you want to buy?",
        ['al', 'ak', 'az', 'ar', 'ca', 'co', 'ct', 'dc', 'de', 'fl', 'ga',
       'hi', 'id', 'il', 'in', 'ia', 'ks', 'ky', 'la', 'me', 'md', 'ma',
       'mi', 'mn', 'ms', 'mo', 'mt', 'nc', 'ne', 'nv', 'nj', 'nm', 'ny',
       'nh', 'nd', 'oh', 'ok', 'or', 'pa', 'ri', 'sc', 'sd', 'tn', 'tx',
       'ut', 'vt', 'va', 'wa', 'wv', 'wi', 'wy'],index=None,placeholder="Select state..."
            )

    # Three elements below
col11, col12, col13 = st.columns(3)

with col11:
    # First checkbox
    manufacturer = st.selectbox("Select Car Brand",['gmc', 'chevrolet', 'toyota', 'ford', 'jeep', 'nissan', 'ram',
    'cadillac', 'honda', 'dodge', 'lexus', 'jaguar', 'buick',
    'chrysler', 'volvo', 'infiniti', 'lincoln', 'Unknown', 'acura',
    'hyundai', 'mercedes-benz', 'audi', 'bmw', 'mitsubishi', 'subaru',
    'alfa-romeo', 'volkswagen', 'mazda', 'porsche', 'kia', 'pontiac',
    'fiat', 'rover', 'mini', 'tesla', 'saturn', 'mercury',
    'harley-davidson', 'datsun', 'land rover', 'aston-martin',
    'ferrari'])

with col12:
    # Second checkbox
    model = st.selectbox("Select Car Model",['f-150', 'silverado 1500', '1500', 'accord', 'camry', 'civic', 'silverado', 'escape', 'altima', 'tacoma', 'wrangler', 'grand cherokee', 'mustang', 'explorer', 'corolla', 'cr-v', '2500', 'focus', 'equinox', 'rav4', 'fusion', 'corvette', 'sonata', 'impala', 'malibu', 'odyssey', 'outback', 'jetta', 'prius', 'elantra', 'cruze', 'sierra 1500', 'tahoe', 'forester', 'rogue', 'tundra', 'grand caravan', 'sentra', 'f-250', 'charger', 'f150 supercrew cab', 'sierra', 'edge', 'silverado 2500hd', 'highlander', 'sienna', 'camaro', 'wrangler unlimited', '4runner', 'cherokee', 'pilot', 'soul', 'f250 super duty', 'c-class', 'suburban', 'f150', 'impreza', '1500 crew cab', '3500', 'x5', 'town & country', 'acadia', 'passat', 'traverse', 'a4', '3 series', 'ranger', 'f-350', 'optima', 'expedition', 'santa fe', 'mdx', 'taurus', 'silverado 1500 lt', 'challenger', '200', 'legacy', 'f-250 super duty', 'super duty f-250', 'fusion se', 'versa', 'colorado', 'tacoma access cab', 'maxima', 'murano', 'grand cherokee laredo', 'e-class', 'pathfinder', 'terrain', 'f150 super cab', 'compass', 'liberty', 'sorento', 'yukon', 'journey', 'durango', 'sierra 2500hd', 'escalade', 'silverado 1500 crew', 'frontier'])

with col13:
    # Numeric input with a range from 1900 to 2012
    year = st.number_input("Enter a year", min_value=1900, max_value=2012, step=1)
# Three elements below
col21, col22, col23 = st.columns(3)

with col21:
    # First checkbox
    fuel = st.selectbox("fuel type",['gas', 'diesel', 'hybrid', 'electric', 'other'])

with col22:
    title_status = st.selectbox("Title status",['clean', 'rebuilt', 'salvage', 'lien', 'missing', 'parts only'])

with col23:
    odometer = st.number_input("Select odometer", min_value=0, max_value=299999, step=1)

# Three elements below
col31, col32, col33 = st.columns(3)

with col31:
    # First checkbox
    transmission = st.selectbox("Select Car Transmisssion",['automatic', 'other', 'manual'])

with col32:
    # Second checkbox
    type = st.selectbox("Select Car type",['sedan', 'SUV', 'pickup', 'truck', 'coupe', 'hatchback', 'wagon', 'convertible', 'van', 'mini-van', 'offroad', 'bus', 'other'])

with col33:
    # Numeric input with a range from 1900 to 2012
    paint_color = st.selectbox("Select Car color",['white', 'black', 'silver', 'blue', 'red', 'grey', 'green', 'custom', 'brown', 'yellow', 'orange', 'purple'])

submit = st.button("SUBMIT", type="primary")

if __name__ == "__main__":
    predictor = CarPricePredictor()

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

    # Example input data in dictionary or list form
    input_data = {
        'year': year,
        'manufacturer': manufacturer,
        'model': model,
        'fuel': fuel,
        'odometer': odometer,
        'title_status': title_status,
        'transmission': transmission,
        'type': type,
        'paint_color': paint_color,        
        'state': state,
    }

    # Make a prediction
    predicted_price = predictor.predict(input_data)

if submit:
    st.write(f'Predicted Price is : {predicted_price}')

