import os
import logging
import traceback
import numpy as np
import pandas as pd

from scipy.stats import zscore
from typing import Optional, List, Union
from sklearn.ensemble import IsolationForest

from ..Helper.dataloader import load_data
from ..Helper.datasaver import save_file


class OutlierDetector:
    def __init__(self, logger: Optional[logging.Logger] = None, columns: Optional[List[str]] = None):
        if logger is None:
            logging.basicConfig(level=logging.INFO)
            self.logger = logging.getLogger(__name__)
        else:
            self.logger = logger
        
        self.columns = columns
        self.data = None
        self.outliers = None

    
    def load_data(self, file: str) -> dict:
        response = load_data("CLEANER", file)
        
        if response["success"]:
            self.data = response["data"]
            
        return response
        
            
    def save_file(self, output_file: str, index: bool = False, **kwargs) -> dict:
        response = save_file("CLEANER", self.data, output_file)
        
        return response
        

    def detect_outliers(self, method: str = 'iqr', threshold: float = 1.5) -> Union[pd.DataFrame, None]:
        try:
            if self.data is None:
                raise ValueError("[OutlierDetector] No data loaded. Please load data first using load_data().")
            
            if self.columns is None:
                self.columns = self.data.select_dtypes(include=['float64', 'int']).columns.tolist()

            self.logger.info(f"[OutlierDetector] Detecting outliers in columns: {self.columns} using {method} method.")

            if method == 'iqr':
                outliers = []
                
                for col in self.columns:
                    Q1 = self.data[col].quantile(0.25)
                    Q3 = self.data[col].quantile(0.75)
                    IQR = Q3 - Q1
                    lower_bound = Q1 - threshold * IQR
                    upper_bound = Q3 + threshold * IQR
                    outliers.extend(self.data[(self.data[col] < lower_bound) | (self.data[col] > upper_bound)].index)
                
                outlier_indices = list(set(outliers))

            elif method == 'zscore':
                z_scores = np.abs(zscore(self.data[self.columns]))
                outlier_indices = np.where(z_scores > threshold)[0]

            elif method == 'isolation_forest':
                isolation_forest = IsolationForest(contamination=0.05, random_state=42)
                preds = isolation_forest.fit_predict(self.data[self.columns])
                outlier_indices = np.where(preds == -1)[0]

            else:
                raise ValueError("Unsupported method. Choose 'iqr', 'zscore', or 'isolation_forest'.")

            self.outliers = self.data.iloc[outlier_indices]
            self.logger.info(f"[OutlierDetector] Outlier detection completed. Found {len(outlier_indices)} outliers.")
            
            return {
                "success": True, 
                "message": None,
                "data": self.outliers,
                "error": None
            }
        except Exception as e:
            error = traceback.print_exc()
            self.logger.error(f"[OutlierDetector] Error detecting outlier(s): {str(e)}")
            self.logger.error(f"[OutlierDetector] {error}")
            
            return {
                "success": False, 
                "message": str(e), 
                "data": None,
                "error" : error
            }
            

    def remove_outliers(self) -> Union[pd.DataFrame, None]:
        try:
            if self.outliers is None:
                self.logger.warning("[OutlierDetector] No outliers detected. Running detect outliers...")
                response = self.detect_outliers()
                
                if not response["success"]:
                    return response

            self.logger.info("[OutlierDetector] Removing detected outliers from the dataset.")
            self.data = self.data.drop(self.outliers.index).reset_index(drop=True)
            self.logger.info(f"[OutlierDetector] Outliers removed. New shape of data: {self.data.shape}")
            
            return {
                "success": True, 
                "message": None,
                "data": self.data,
                "error": None
            }
        except Exception as e:
            error = traceback.print_exc()
            self.logger.error(f"[OutlierDetector] Error removing outliers: {str(e)}")
            self.logger.error(f"[OutlierDetector] {error}")
            
            return {
                "success": False, 
                "message": str(e), 
                "data": None,
                "error" : error
            }