import os
import logging
import traceback
import numpy as np
import pandas as pd

from typing import Optional, Union
from imblearn.combine import SMOTEENN
from sklearn.impute import SimpleImputer
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler

from ..Helper.dataloader import load_data
from ..Helper.datasaver import save_file


class DatasetBalancer:
    def __init__(self, logger: Optional[logging.Logger] = None, target_column: str = "target"):
        if logger is None:
            logging.basicConfig(level=logging.INFO)
            self.logger = logging.getLogger(__name__)
        else:
            self.logger = logger
        
        self.target_column = target_column
        self.data = None


    def load_data(self, file: str) -> dict:
        response = load_data("DatasetBalancer", file)
        
        if response["success"]:
            self.data = response["data"]
            
        return response
        
            
    def save_file(self, output_file: str, index: bool = False, **kwargs) -> dict:
        response = save_file("DatasetBalancer", self.data, output_file)
        
        return response
            

    def balance_data(self, method: str = 'smote', handle_missing: str = 'mean') -> Union[pd.DataFrame, None]:
        try:
            if self.data is None:
                raise ValueError("[DatasetBalancer] No data loaded. Please load data first using load_data().")
            
            X = self.data.drop(columns=[self.target_column])
            y = self.data[self.target_column]
            
            X.replace([np.inf, -np.inf], np.nan, inplace=True)

            if handle_missing:
                self.logger.info(f"[DatasetBalancer] Handling missing values using {handle_missing} strategy.")
                imputer = SimpleImputer(strategy=handle_missing)
                X = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)

            if method == 'smote':
                sampler = SMOTE()
            elif method == 'undersample':
                sampler = RandomUnderSampler()
            elif method == 'combine':
                sampler = SMOTEENN()
            else:
                raise ValueError("Unsupported balancing method. Choose 'smote', 'undersample', or 'combine'.")

            self.logger.info(f"[DatasetBalancer] Balancing data using {method} method.")
            
            # Perform balancing
            try:
                X_resampled, y_resampled = sampler.fit_resample(X, y)
                balanced_data = pd.concat([pd.DataFrame(X_resampled, columns=X.columns), pd.DataFrame(y_resampled, columns=[self.target_column])], axis=1)
                self.data = balanced_data
                self.logger.info(f"[DatasetBalancer] Data balancing completed. New shape: {self.data.shape}")
                
                return {
                    "success": True, 
                    "message": None,
                    "data": self.data,
                    "error": None
                }
            except Exception as e:
                error = traceback.print_exc()
                self.logger.error(f"[DatasetBalancer] Error during data balancing: {str(e)}")
                self.logger.error(f"[DatasetBalancer] {error}")
            
                return {
                    "success": False, 
                    "message": str(e), 
                    "data": None,
                    "error" : error
                }
        except Exception as e:
            error = traceback.print_exc()
            self.logger.error(f"[DatasetBalancer] Error saving file: {str(e)}")
            self.logger.error(f"[DatasetBalancer] {error}")
            
            return {
                "success": False, 
                "message": str(e), 
                "data": None,
                "error" : error
            }