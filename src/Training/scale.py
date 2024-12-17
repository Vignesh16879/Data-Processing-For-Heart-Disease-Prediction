import os
import sys
import time
import joblib
import logging
import traceback
import numpy as np
import pandas as pd

from typing import Optional

from ..Helper.dataloader import load_data
from ..Helper.datasaver import save_file


class Scale:
    def __init__(self, logger: Optional[logging.Logger] = None):
        # Setup logging
        if logger is None:
            logging.basicConfig(level=logging.INFO)
            self.logger = logging.getLogger(__name__)
        else:
            self.logger = logger

        self.data = None
        
    
    def load_data(self, file: str) -> dict:
        response = load_data("Scaling", file, self.logger)
        
        if response["success"]:
            self.data = response["data"]
            
        return response

    def save_file(self, output_file: str, index: bool = False, **kwargs) -> dict:
        response = save_file("Scaling", self.data, output_file)
        
        return response

    
    def scale_data(self):
        try:
            if self.data is None:
                raise ValueError("No data loaded. Please load data first.")
            
            # Ensure that only numeric columns are used for scaling
            numeric_data = self.data.select_dtypes(include=[np.number])

            if numeric_data.empty:
                raise ValueError("No numeric columns found in the data.")
            
            # Calculate mean and std for numeric columns only
            self.mean = numeric_data.mean(axis=0)
            self.std = numeric_data.std(axis=0)

            # Normalize the data (standardization)
            scaled_data = (numeric_data - self.mean) / self.std
            self.scaled_data = scaled_data
            
            # Logging the scaling process
            self.logger.info("Data scaling completed: mean and std calculated.")
            
            return {
                "success": True,
                "message": "Data scaled successfully.",
                "scaled_data": self.scaled_data,
                "mean": self.mean,
                "std": self.std,
                "error": None
            }
        except Exception as e:
            error = traceback.format_exc()
            self.logger.error(f"[Scaling] Error in scale_data: {str(e)}")
            self.logger.error(f"[Scaling] {error}")
            
            return {
                "success": False,
                "message": str(e),
                "data": None,
                "error": error
            }
    
    
    def mini_batch_gradient_descent(self, batch_size: int = 32, learning_rate: float = 0.01, epochs: int = 100):
        try:
            if self.scaled_data is None:
                raise ValueError("No scaled data available. Please scale data first.")
            
            # Convert scaled data into numpy array
            X = np.array(self.scaled_data)
            
            # Ensure target data 'y' is numeric and aligned with X
            y = self.data.iloc[:, -1].values  # Assuming the target is the last column
            y = pd.to_numeric(y, errors='coerce')  # Coerce non-numeric values to NaN
            valid_indices = ~np.isnan(y)
            X = X[valid_indices]
            y = y[valid_indices]

            # Initialize weights and biases
            m, n = X.shape
            weights = np.zeros(n)
            bias = 0

            # Mini-batch gradient descent
            for epoch in range(epochs):
                # Shuffle data for each epoch
                indices = np.random.permutation(m)
                X_shuffled = X[indices]
                y_shuffled = y[indices]

                for i in range(0, m, batch_size):
                    # Select mini-batch
                    X_batch = X_shuffled[i:i + batch_size]
                    y_batch = y_shuffled[i:i + batch_size]

                    # Ensure proper shapes for broadcasting
                    y_batch = y_batch.reshape(-1)  # (batch_size,)
                    y_pred = np.dot(X_batch, weights) + bias  # (batch_size,)

                    # Compute gradients
                    error = y_batch - y_pred
                    dw = -2 * np.dot(X_batch.T, error) / batch_size  # (n_features,)
                    db = -2 * np.sum(error) / batch_size  # Scalar

                    # Update weights and bias
                    weights -= learning_rate * dw
                    bias -= learning_rate * db

                if epoch % 10 == 0:
                    self.logger.info(f"Epoch {epoch}/{epochs} - Loss: {self.compute_loss(X, y, weights, bias)}")

            return {
                "success": True,
                "weights": weights,
                "bias": bias
            }

        except Exception as e:
            error = traceback.format_exc()
            self.logger.error(f"[Scaling] Error in mini_batch_gradient_descent: {str(e)}")
            self.logger.error(f"[Scaling] {error}")

            return {
                "success": False,
                "message": str(e),
                "error": error
            }
            
    
    def compute_loss(self, X, y, weights, bias):
        y_pred = np.dot(X, weights) + bias
        loss = np.mean((y - y_pred) ** 2)
        
        return loss