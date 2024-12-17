import os
import sys
import time
import joblib
import logging
import traceback
import pandas as pd


from pathlib import Path
from typing import Optional
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

from ..Helper.dataloader import load_data


class Test:
    def __init__(self, logger: Optional[logging.Logger] = None, model = None, model_path: Optional[str] = None):
        # Setup logging
        if logger is None:
            logging.basicConfig(level=logging.INFO)
            self.logger = logging.getLogger(__name__)
        else:
            self.logger = logger

        self.data = None
        self.model_path = model_path

        # If model is provided, use it; otherwise try to load from path
        if model:
            self.clf = model
        elif model_path:
            try:
                self.clf = joblib.load(model_path)
                self.logger.info(f"[Testing] Loaded model from {model_path}")
            except Exception as e:
                self.logger.error(f"[Testing] Error loading model: {str(e)}")
                raise
        else:
            raise ValueError("Either a model or a model path must be provided")

    def load_data(self, file: str) -> dict:
        response = load_data("Testing", file, self.logger)
        
        if response["success"]:
            self.data = response["data"]
            
        return response
    
    def test_model(self):
        self.logger.info("[test_model] Starting model testing.")
        try:
            if self.data is None:
                raise ValueError("No data loaded. Use load_data() first.")

            # Remove rows with NaN values in the target column
            self.data = self.data.dropna(subset=['target'])

            X = self.data.drop(columns=['target'])
            y = self.data['target']

            # Handle NaN in features
            X = X.dropna()
            y = y[X.index]  # Align y with X after dropping NaNs

            # Identify categorical and numeric columns
            categorical_columns = X.select_dtypes(include=['object', 'category']).columns
            numeric_columns = X.select_dtypes(include=['float64', 'int64']).columns

            # Create preprocessor
            preprocessor = ColumnTransformer(
                transformers=[
                    ('num', Pipeline([
                        ('imputer', SimpleImputer(strategy='median')),
                    ]), numeric_columns),
                    ('cat', Pipeline([
                        ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
                        ('onehot', OneHotEncoder(handle_unknown='ignore'))
                    ]), categorical_columns)
                ])

            # Split data for training and testing
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

            # Create a pipeline that includes preprocessing and the model
            pipeline = Pipeline([
                ('preprocessor', preprocessor),
                ('classifier', self.clf)
            ])

            # Train the model
            self.logger.info("[test_model] Model training completed.")
            pipeline.fit(X_train, y_train)

            # Measure prediction time
            start_pred = time.time()
            y_pred = pipeline.predict(X_test)
            end_pred = time.time()

            # Evaluate accuracy
            accuracy = accuracy_score(y_test, y_pred)

            # self.logger.info(f"[test_model] Model accuracy: {accuracy:.4%}.")
            # self.logger.info(f"[Testing] Prediction Time: {end_pred - start_pred:.4f} seconds")
            # self.logger.info(f"[Testing] Model Accuracy: {accuracy:.4%}")

            return {
                "success": True,
                "message": "Model tested successfully",
                "data": {
                    "accuracy": accuracy,
                    "prediction_time": end_pred - start_pred,
                    "pipeline": pipeline
                },
                "error": None
            }
        except Exception as e:
            error = traceback.format_exc()
            self.logger.error(f"[Testing] Error in test_model: {str(e)}")
            self.logger.error(f"[Testing] {error}")

            return {
                "success": False,
                "message": str(e),
                "data": None,
                "error": error
            }