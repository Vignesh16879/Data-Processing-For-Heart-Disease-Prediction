import os
import logging
import traceback
import pandas as pd

from typing import Optional, List, Union
from sklearn.preprocessing import MinMaxScaler, StandardScaler

from ..Helper.dataloader import load_data
from ..Helper.datasaver import save_file


class FeatureScaler:
    def __init__(self, columns: Optional[List[str]] = None, logger: Optional[logging.Logger] = None):
        if logger is None:
            logging.basicConfig(level=logging.INFO)
            self.logger = logging.getLogger(__name__)
        else:
            self.logger = logger
        
        self.columns = columns
        self.scaler = None
        self.data = None  


    def load_data(self, file: str) -> dict:
        response = load_data("CLEANER", file)
        
        if response["success"]:
            self.data = response["data"]
            
        return response
        
            
    def save_file(self, output_file: str, index: bool = False, **kwargs) -> dict:
        response = save_file("CLEANER", self.data, output_file)
        
        return response
            

    def fit_transform(self, method: str = 'normalize') -> pd.DataFrame:
        try:
            if self.data is not None:
                self.data = self.data

            if self.data is None:
                raise ValueError("No data loaded. Please load data first using load_data().")

            if self.columns is None:
                # Automatically select numerical columns if none are specified
                self.columns = self.data.select_dtypes(include=['float64', 'int']).columns.tolist()

            if method == 'normalize':
                self.scaler = MinMaxScaler()
            elif method == 'standardize':
                self.scaler = StandardScaler()
            else:
                raise ValueError("Unsupported scaling method. Choose 'normalize' or 'standardize'.")

            self.logger.info(f"[FeatureScaler] Scaling columns: {self.columns} using {method} method.")
            
            # Perform scaling and return transformed DataFrame
            self.data[self.columns] = self.scaler.fit_transform(self.data[self.columns])
            
            self.logger.info("[FeatureScaler] Scaling completed.")
            
            return {
                "success": True, 
                "message": None, 
                "data": self.data,
                "error" : None
            }
        except Exception as e:
            error = traceback.print_exc()
            self.logger.error(f"[FeatureScaler] Error processing data: {str(e)}")
            self.logger.error(f"[FeatureScaler] {error}")
            
            return {
                "success": False, 
                "message": str(e), 
                "data": None,
                "error" : error
            }
            

    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        try:
            if self.scaler is None:
                raise ValueError("Scaler has not been fitted. Call fit_transform() first.")
            
            self.logger.info("[FeatureScaler] Transforming new data with fitted scaler.")
            
            data_transformed = data.copy()
            data_transformed[self.columns] = self.scaler.transform(data[self.columns])
            
            return {
                "success": True, 
                "message": None, 
                "data": data_transformed,
                "error" : None
            }
        except Exception as e:
            error = traceback.print_exc()
            self.logger.error(f"[FeatureScaler] Error processing data: {str(e)}")
            self.logger.error(f"[FeatureScaler] {error}")
            
            return {
                "success": False, 
                "message": str(e), 
                "data": None,
                "error" : error
            }