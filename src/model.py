import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
import torch.optim as optim
import os
from data_preprocessing import CarDataPreprocessor
import numpy as np
from sklearn.metrics import r2_score
import config
from data_handling import DataHandler
from logger import get_logger
import logging

# Dataset class to handle features and labels
class CustomDataset(Dataset):
    def __init__(self, features, labels):
        self.features = features
        self.labels = labels

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]

# Model definition (as per the new logic)
class Model1(nn.Module):
    def __init__(self, num_features):
        super().__init__()
        self.linear = nn.Linear(num_features, 128)
        self.relu = nn.ReLU()
        self.linear1 = nn.Linear(128, 1)

    def forward(self, features):
        out = self.linear(features)
        out = self.relu(out)
        out = self.linear1(out)
        return out

def load_model(model_path):
        model = Model1(10)  # Create the model instance
        model.load_state_dict(torch.load(model_path, weights_only=True))
        model.eval()  # Set to evaluation mode
        return model

class ModelTrainer:
    def __init__(self, train_dataset, val_dataset, model_save_path):
        self.train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=True)
        self.val_loader = DataLoader(val_dataset, batch_size=config.BATCH_SIZE, shuffle=False)
        self.model = Model1(train_dataset.features.shape[1])
        self.optimizer = optim.Adam(self.model.parameters(), lr=config.LEARNING_RATE)
        self.criterion = nn.MSELoss()
        self.model_save_path = model_save_path
        self.logger = get_logger(__name__)

    def train(self, epochs=config.EPOCHS, patience=config.PATIENCE):
        best_val_loss = float('inf')  # Start with a very large value for validation loss
        best_model_wts = None
        patience_counter = 0  # Counter to track epochs without improvement
        self.logger.info("--------------Model Training Phase-----------")

        for epoch in range(epochs):
            self.model.train()
            total_loss = 0.0
            val_loss = 0.0
            val_true = []
            val_pred = []

            # Combine training and validation in a single loop using zip
            for (train_features, train_labels), (val_features, val_labels) in zip(self.train_loader, self.val_loader):
                # Training phase
                self.optimizer.zero_grad()
                train_predictions = self.model(train_features)
                train_loss = self.criterion(train_predictions, train_labels.view(-1, 1))

                train_loss.backward()
                self.optimizer.step()
                total_loss += train_loss.item()

                # Validation phase
                with torch.no_grad():
                    val_predictions = self.model(val_features)
                    val_loss_item = self.criterion(val_predictions, val_labels.view(-1, 1))
                    val_loss += val_loss_item.item()

                    val_true.extend(val_labels.numpy())
                    val_pred.extend(val_predictions.numpy())

            # Calculate average losses
            avg_loss = total_loss / len(self.train_loader)
            avg_val_loss = val_loss / len(self.val_loader)

            # Calculate R² score for validation
            val_true = np.array(val_true)
            val_pred = np.array(val_pred)
            val_r2 = r2_score(val_true, val_pred)

            print(f"Epoch [{epoch + 1}/{epochs}], Loss: {avg_loss:.4f}, Validation Loss: {avg_val_loss:.4f}, Validation R²: {val_r2:.4f}")

            # Early stopping logic
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                best_model_wts = self.model.state_dict()  # Save the best model
                patience_counter = 0  # Reset patience counter
            else:
                patience_counter += 1

            if patience_counter >= patience:
                print(f"Early stopping after epoch {epoch + 1}.")
                break
            
        self.logger.info("Model trained ......")

        # Save the best model after all epochs
        if best_model_wts is not None:
            self.save_model(best_model_wts)

    def save_model(self, best_model_wts):
        os.makedirs(os.path.dirname(self.model_save_path), exist_ok=True)
        torch.save(best_model_wts, self.model_save_path)
        self.logger.info("Best Model saved......")
        self.logger.info("-------------Model trained sucessfully-----------")


if __name__ == "__main__":
    
    #-------------preprocessing phase---------------

    # Use the data path from the config file
    preprocessor = CarDataPreprocessor(config.RAW_DATAPATH)
    # Load the dataset
    preprocessor.load_data()
    # Apply preprocessing steps
    preprocessor.preprocess()
    # Save the cleaned dataset
    preprocessor.save_data(config.CLEANED_DATAPATH)

    #---------data handling phase------------

    data_handler = DataHandler(file_path=config.CLEANED_DATAPATH)
    x_train, y_train, x_val, y_val, _, _ = data_handler.load_and_preprocess()

    #---------training phase-------------

    # Create datasets for training and validation
    train_dataset = CustomDataset(x_train, y_train)
    val_dataset = CustomDataset(x_val, y_val)

    # Initialize the model trainer using the model save path from the config file
    trainer = ModelTrainer(train_dataset, val_dataset, model_save_path=config.MODEL_PATH)

    # Train the model
    trainer.train()


