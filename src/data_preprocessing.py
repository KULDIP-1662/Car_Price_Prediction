import pandas as pd
import numpy as np
# import src.config
from logger import get_logger
import logging
import config

class CarDataPreprocessor:
    def __init__(self, dataset_path):
        """
        Initialize the preprocessor with the dataset path.
        """
        self.dataset_path = dataset_path
        self.dataset = None
        self.logger = get_logger(__name__)

    def load_data(self):
        """
        Load the dataset from the given path.
        """
        self.logger.info("--------------Preprocessing Phase-----------")
        self.dataset = pd.read_csv(self.dataset_path)
        self.logger.info("Data loaded succesfully......")
        # print("Dataset loaded successfully.")

    def save_data(self, save_path):
        """
        Save the cleaned dataset to the specified path.
        """
        self.dataset.to_csv(save_path, index=False)
        self.logger.info("Cleaned data saved succesfully......")
        # print(f"Dataset saved to {save_path}.")
# 
    def remove_high_missing_columns(self, threshold=0.4):
        """
        Remove columns with more than a specified percentage of missing values.
        """
        self.logger.info("Data Preprocessing started......")
        dataset_rows = len(self.dataset)
        removed_columns = [
            col for col in self.dataset.columns if self.dataset[col].isnull().sum() > dataset_rows * threshold
        ]
        self.dataset.drop(columns=removed_columns, inplace=True)
        # print(f"Removed columns with high missing values: {removed_columns}")

    def fill_missing_values(self):
        """
        Fill missing values based on specified rules.
        """
        # Remove rows where any of the specified columns have missing values
        self.dataset.dropna(subset=['model', 'type', 'year', 'paint_color'], inplace=True)

        # Fill missing values in 'manufacturer' with 'Unknown'
        self.dataset['manufacturer'] = self.dataset['manufacturer'].fillna('Unknown')

        # Fill missing values in 'fuel', 'title_status', and 'transmission' with the most frequent category
        for col in ['fuel', 'title_status', 'transmission']:
            self.dataset[col] = self.dataset[col].fillna(self.dataset[col].mode()[0])

        # Simplify 'model' to its first three words
        self.dataset['model'] = self.dataset['model'].str.split().str.slice(0, 3).str.join(' ')

        # print("Filled missing values.")

    def remove_outliers(self):
        """
        Remove outliers based on 'price' and 'odometer' values.
        """
        self.dataset = self.dataset[(self.dataset['price'] > 100) & (self.dataset['price'] < 60000)]
        self.dataset = self.dataset[self.dataset['odometer'] < 300000]
        # print("Removed outliers based on price and odometer.")

    def drop_columns(self):
        """
        Drop irrelevant columns.
        """
        columns_to_drop = [
            'id', 'region', 'posting_date', 'url', 'region_url', 
            'VIN', 'image_url', 'description', 'lat', 'long', 'drive'
        ]
        self.dataset.drop(columns=columns_to_drop, inplace=True, errors='ignore')
        # print(f"Dropped columns: {columns_to_drop}")

    def remove_duplicates(self):
        """
        Remove duplicates and redundant rows.
        """
        # Remove duplicates
        self.dataset.drop_duplicates(inplace=True)

        # Remove rows with NaN values except 'price' and 'state'
        self.dataset = self.dataset[~self.dataset.drop(['price', 'state'], axis=1).isnull().all(axis=1)]

        # Remove duplicates except for the 'price' column
        self.dataset = self.dataset[~self.dataset.drop('price', axis=1).duplicated()]
        self.dataset.reset_index(drop=True, inplace=True)

        # print("Removed duplicates.")

    def convert_column_types(self):
        """
        Convert columns to appropriate types.
        """
        self.dataset['year'] = self.dataset['year'].astype(int)
        self.dataset['odometer'] = self.dataset['odometer'].astype(int)
        # print("Converted column types.")

    def preprocess(self):
        """
        Execute all preprocessing steps in sequence.
        """
        if self.dataset is None:
            self.logger.warning("Dataset is None. Skipping preprocessing.")
            return

        self.remove_high_missing_columns()
        self.drop_columns()
        self.remove_duplicates()
        self.remove_outliers()
        self.fill_missing_values()
        self.convert_column_types()
        # self.logger.info("--------------Preprocessing Phase ended-----------")
 # One log at the end

# Example usage
if __name__ == "__main__":
    # Initialize the preprocessor
    preprocessor = CarDataPreprocessor(config.RAW_DATAPATH)

    # Load the dataset
    preprocessor.load_data()

    # Apply preprocessing steps
    preprocessor.preprocess()

    # Save the cleaned dataset
    preprocessor.save_data(config.CLEANED_DATAPATH)
