# import requests
# from pydantic import BaseModel

# # Define the CarInput data structure (same as in app.py)
# class CarInput(BaseModel):
#     year: int
#     manufacturer: str
#     model: str
#     fuel: str
#     odometer: int
#     title_status: str
#     transmission: str
#     type: str
#     paint_color: str
#     state: str


# class CarPredictor:
#     """
#     A class to handle car prediction tasks.
#     """

#     def __init__(self, url="http://localhost:8000/predict"):
#         """
#         Initializes the CarPredictor object with the API endpoint URL.

#         Args:
#             url (str, optional): The URL of the prediction endpoint in your app.py file. Defaults to "http://localhost:8000/predict".
#         """
#         self.url = url

#     def predict(self, input_data: CarInput):
#         """
#         Sends the input data to the prediction endpoint and returns the predicted price.

#         Args:
#             input_data (CarInput): An object containing the car data for prediction.

#         Returns:
#             float: The predicted price of the car.
#         """

#         # Convert the input data to a dictionary using model_dump
#         data = input_data.model_dump()

#         # Send a POST request to the prediction endpoint
#         response = requests.post(self.url, json=data)

#         # Check for successful response
#         if response.status_code == 200:
#             # Get the predicted price from the JSON response
#             prediction = response.json()
#             return prediction["predicted_price"][0]  # Assuming a single price in the list
#         else:
#             # Handle error (e.g., print an error message)
#             print(f"Error: API call failed with status code {response.status_code}")
#             return None

# # Example usage
# if __name__ == "__main__":

#     car_data = CarInput(
#         year=2020,
#         manufacturer="gmc",
#         model="sierra 1500 crew",
#         fuel="gas",
#         odometer=79230,
#         title_status="clean",
#         transmission="other",
#         type="pickup",
#         paint_color="white",
#         state="al",
#     )

#     # Create a CarPredictor object (replace URL if your API runs on a different address/port)
#     predictor = CarPredictor()

#     # Make the prediction and display the result
#     predicted_price = predictor.predict(car_data)
#     if predicted_price:
#         print(f"Predicted price for the car: {predicted_price:.2f}")
#     else:
#         print("Prediction failed.")
import requests

# Define the car data structure as a dictionary
car_data = {
    "year": 2020,
    "manufacturer": "gmc",
    "model": "sierra 1500 crew",
    "fuel": "gas",
    "odometer": 79230,
    "title_status": "clean",
    "transmission": "other",
    "type": "pickup",
    "paint_color": "white",
    "state": "al"
}

# Define the API endpoint URL
url = "http://localhost:8000/predict"  

# Send a POST request to the prediction endpoint
response = requests.post(url, json=car_data)

# Check for successful response
if response.status_code == 200:
    # print(response.json())
    # Parse the predicted price from the response
    prediction = response.json()
    # print(prediction)
    predicted_price = prediction["predicted_price"][0]  # Assuming the response contains a list
    print(f"Predicted price for the car: {predicted_price:.2f}")
else:
    print(f"Error: API call failed with status code {response.status_code}")
