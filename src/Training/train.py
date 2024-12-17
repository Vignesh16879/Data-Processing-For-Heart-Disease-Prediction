import os
import sys
import time
import joblib
import logging
import traceback
import pandas as pd

from typing import Optional
from sklearn.model_selection import train_test_split

from ..Helper.dataloader import load_data


class Train:
    def __init__(self, logger: Optional[logging.Logger] = None, model = None, model_path: Optional[str] = None):
        # Setup logging
        if logger is None:
            logging.basicConfig(level=logging.INFO)
            self.logger = logging.getLogger(__name__)
        else:
            self.logger = logger

        self.data = None
        
        # If model is provided, use it; otherwise try to load from path
        if model:
            self.clf = model
        elif model_path:
            try:
                self.clf = joblib.load(model_path)
                self.logger.info(f"[Training] Loaded model from {model_path}")
            except Exception as e:
                raise ValueError(f"[Training] Error loading model: {str(e)}")
        else:
            raise ValueError("[Training] Either a model or a model path must be provided")
        
    
    def load_data(self, file: str) -> dict:
        response = load_data("Training", file, self.logger)
        
        if response["success"]:
            self.data = response["data"]
            
        return response

    
    def train_model(self):
        try:
            if self.data is None:
                raise ValueError("No data loaded. Please load data before training the model.")

            # Load data into a DataFrame
            df = pd.DataFrame(self.data)

            # Check and preprocess non-numeric columns
            for col in df.columns:
                if df[col].dtype == 'object':  # Identify non-numeric columns
                    self.logger.info(f"[Training] Encoding non-numeric column: {col}")
                    df[col] = df[col].astype('category').cat.codes  # Encode categorical data as integers

            # Separate features and target
            X = df.drop(columns=["target"])
            y = df["target"]
            
            # Handle missing values in the target column
            if y.isnull().any():
                self.logger.warning("[Training] Missing values detected in the target column. Dropping rows with NaN in the target.")
                df = df.dropna(subset=["target"])
                X = df.drop(columns=["target"])
                y = df["target"]

            # Train-test split
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

            # Train the model
            self.logger.info("[Training] Training the model...")
            self.clf.fit(X_train, y_train)
            self.logger.info(f"[Training] Model training completed.")

            return {
                "success": True,
                "message": "Model training completed successfully.",
                "data": {
                     "model": self.clf
                },
                "error": None
            }

        except Exception as e:
            error = traceback.format_exc()
            self.logger.error(f"[Training] Error in train_model: {str(e)}")
            self.logger.error(f"[Training] {error}")

            return {
                "success": False,
                "message": str(e),
                "data": None,
                "error": error
            }