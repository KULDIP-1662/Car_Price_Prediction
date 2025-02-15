import torch
from torch.utils.data import DataLoader
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from data_handling import DataHandler  # Ensure this points to your data handling class
import config
from model import Model1  # Ensure you import the correct model architecture
from logger import get_logger
import logging

# Dataset class to handle features and labels
class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, features, labels):
        self.features = features
        self.labels = labels

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]

# Model evaluation class
class ModelEvaluator:
    def __init__(self, model, test_dataset):
        self.model = model
        self.test_loader = DataLoader(test_dataset, batch_size=config.BATCH_SIZE, shuffle=False)
        self.logger = get_logger(__name__)
    def evaluate(self):
        self.logger.info("--------------Model Evaluation Phase-----------")
        self.model.eval()  # Set the model to evaluation mode

        y_true = []
        y_pred = []

        # No gradient computation required for evaluation
        with torch.no_grad():
            for features, labels in self.test_loader:
                predictions = self.model(features)
                y_true.extend(labels.numpy())
                y_pred.extend(predictions.numpy())

        # Convert lists to numpy arrays
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)

        # Calculate performance metrics
        mse = mean_squared_error(y_true, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)
        self.logger.info(f'''Performance Metrics : 
                            Mean Squared Error (MSE): {mse:.4f}
                            Root Mean Squared Error (RMSE): {rmse:.4f}
                            Mean Absolute Error (MAE): {mae:.4f}
                            R² Score: {r2:.4f}''')
        # Print evaluation metrics
        # print("\nTest Set Performance Metrics:")
        # print(f"Mean Squared Error (MSE): {mse:.4f}")
        # print(f"Root Mean Squared Error (RMSE): {rmse:.4f}")
        # print(f"Mean Absolute Error (MAE): {mae:.4f}")
        # print(f"R² Score: {r2:.4f}")
        self.logger.info("----------Model Evaluation Completed-------------")

        return mse, rmse, mae, r2

if __name__ == "__main__":
    # Load preprocessed data using DataHandler


    data_handler = DataHandler(file_path=config.CLEANED_DATAPATH)
    _,_,_,_, x_test, y_test = data_handler.load_and_preprocess()

    # Create test dataset
    test_dataset = CustomDataset(x_test, y_test)

    # Load the trained model
    model = Model1(10)  # Ensure the number of features is correct
    model.load_state_dict(torch.load(config.MODEL_PATH,weights_only=True))  # Load the saved model weights
    # model = torch.load(config.MODEL_PATH)
    model.eval()  # Set the model to evaluation mode

    # Initialize evaluator with the model and test dataset
    evaluator = ModelEvaluator(model, test_dataset)

    # Evaluate the model and print metrics
    mse, rmse, mae, r2 = evaluator.evaluate()
