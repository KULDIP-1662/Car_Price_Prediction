import pandas as pd
import torch
import config
import joblib
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from category_encoders import TargetEncoder
from torch.utils.data import Dataset
from logger import get_logger
import logging

# import sys
# import os

# print(f"Python executable: {sys.executable}")
# print(f"Environment name: {os.path.basename(sys.prefix)}")
# print(f"Python version: {sys.version}")

class DataHandler:
    def __init__(self, file_path):
        self.file_path = file_path
        self.logger = get_logger(__name__)
        

    def load_and_preprocess(self):
        self.logger.info("--------------Handling Phase-----------")
        dataset = pd.read_csv(self.file_path)
        # print("dataset loaded in file data_handling")

        x = dataset.drop(columns=['price'])
        y = dataset['price']

        x_train_full, x_test, y_train_full, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
        # print("data succsesfully splited")

        scaler_y = StandardScaler()
        y_train_full = scaler_y.fit_transform(pd.DataFrame(y_train_full))
        y_test = scaler_y.transform(pd.DataFrame(y_test))
        joblib.dump(scaler_y,config.SCALER_Y_PATH)

        target_encoder = TargetEncoder(smoothing=10000)
        x_train_full = target_encoder.fit_transform(x_train_full, y_train_full)
        x_test = target_encoder.transform(x_test)
        joblib.dump(target_encoder,config.ENCODER_PATH)

        scaler_x = StandardScaler()
        x_train_full = scaler_x.fit_transform(x_train_full)
        x_test = scaler_x.transform(x_test)
        joblib.dump(scaler_x,config.SCALER_X_PATH)
        self.logger.info("scaling and encoding done......")


        x_train, x_val, y_train, y_val = train_test_split(x_train_full, y_train_full, test_size=0.2, random_state=42)

        pd.DataFrame(x_train).to_csv(config.XTRAIN_PATH, index=False)
        pd.DataFrame(x_test).to_csv(config.XTEST_PATH, index=False)
        pd.DataFrame(y_train).to_csv(config.YTRAIN_PATH, index=False)
        pd.DataFrame(y_test).to_csv(config.YTEST_PATH, index=False)
        pd.DataFrame(x_val).to_csv(config.XVAL_PATH, index=False)
        pd.DataFrame(y_val).to_csv(config.YVAL_PATH, index=False)
        self.logger.info("Data splited and files saved.........")

        
        return (self.to_tensor(x_train), self.to_tensor(y_train), 
                self.to_tensor(x_val), self.to_tensor(y_val), 
                self.to_tensor(x_test), self.to_tensor(y_test))
    
    @staticmethod
    def to_tensor(array):
        return torch.from_numpy(array.astype(np.float32))
    
# if __name__ == "__main__":
    # data_path = r"C:\Users\Kulde\OneDrive\Desktop\OFFICE\CAR_PREDICT\data\processed\cleaned_car_csv.csv"  # Replace with your actual file path



    # data_handler = DataHandler(config.CLEANED_DATAPATH)
    # x_train, y_train, x_val, y_val, x_test, y_test = data_handler.load_and_preprocess()



    # Check the shapes of the tensors to verify the data processing
    # print(f"x_train shape: {x_train.shape}")
    # print(f"y_train shape: {y_train.shape}")
    # print(f"x_val shape: {x_val.shape}")
    # print(f"y_val shape: {y_val.shape}")
    # print(f"x_test shape: {x_test.shape}")
    # print(f"y_test shape: {y_test.shape}")
    # print("tensors created")

