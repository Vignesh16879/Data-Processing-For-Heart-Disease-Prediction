import os
import logging
import traceback
import pandas as pd

from typing import Optional, List, Union
from sklearn.preprocessing import LabelEncoder

from ..Helper.dataloader import load_data
from ..Helper.datasaver import save_file


class FeatureEngineering:
    def __init__(self, columns: Optional[List[str]] = None, logger: Optional[logging.Logger] = None):
        if logger is None:
            logging.basicConfig(level=logging.INFO)
            self.logger = logging.getLogger(__name__)
        else:
            self.logger = logger
        
        self.columns = columns
        self.data = None
        

    def load_data(self, file: str) -> dict:
        response = load_data("CLEANER", file)
        
        if response["success"]:
            self.data = response["data"]
            
        return response
        
            
    def save_file(self, output_file: str, index: bool = False, **kwargs) -> dict:
        response = save_file("CLEANER", self.data, output_file)
        
        return response
            

    def create_new_features(self) -> pd.DataFrame:
        try:
            if self.data is None:
                raise ValueError("No data loaded. Please load data first using load_data().")

            self.logger.info("[FeatureEngineering] Creating new features...")

            if 'age' in self.data.columns:
                self.data['age_squared'] = self.data['age'] ** 2
                self.logger.info("[FeatureEngineering] Created 'age_squared' feature.")
            
            if 'trestbps' in self.data.columns and 'chol' in self.data.columns:
                self.data['bp_chol_ratio'] = self.data['trestbps'] / self.data['chol']
                self.logger.info("[FeatureEngineering] Created 'bp_chol_ratio' feature.")

            return {
                "success": True, 
                "message": None,
                "data": self.data,
                "error": None
            }
        except Exception as e:
            error = traceback.print_exc()
            self.logger.error(f"[FeatureEngineering] Error creating new feature: {str(e)}")
            self.logger.error(f"[FeatureEngineering] {error}")
            
            return {
                "success": False, 
                "message": str(e), 
                "data": None,
                "error" : error
            }


    def encode_categorical_features(self, columns: List[str] = None, encoding_method: str = 'one_hot') -> pd.DataFrame:
        try:
            if self.data is None:
                raise ValueError("No data loaded. Please load data first using load_data().")

            if columns is None:
                columns = self.data.select_dtypes(include=['object']).columns.tolist()
            
            self.logger.info(f"[FeatureEngineering] Encoding categorical columns: {columns} using {encoding_method} method.")

            if encoding_method == 'one_hot':
                self.data = pd.get_dummies(self.data, columns=columns)
            elif encoding_method == 'label':
                for col in columns:
                    le = LabelEncoder()
                    self.data[col] = le.fit_transform(self.data[col])
            else:
                raise ValueError("Unsupported encoding method. Choose 'one_hot' or 'label'.")

            return {
                "success": True, 
                "message": None,
                "data": self.data,
                "error": None
            }
        except Exception as e:
            error = traceback.print_exc()
            self.logger.error(f"[FeatureEngineering] Error encoding categorical features: {str(e)}")
            self.logger.error(f"[FeatureEngineering] {error}")
            
            return {
                "success": False, 
                "message": str(e), 
                "data": None,
                "error" : error
            }
