import os
import logging
import traceback
import numpy as np
import pandas as pd

from typing import Optional

from ..Helper.dataloader import load_data
from ..Helper.datasaver import save_file


class CLEANER:
    def __init__(self, logger: Optional[logging.Logger] = None):
        if logger is None:
            logging.basicConfig(level=logging.INFO)
            self.logger = logging.getLogger(__name__)
        else:
            self.logger = logger
        
        self.data = None
        self.columns = [
            'age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 
            'restecg', 'thalach', 'exang', 'oldpeak', 
            'slope', 'ca', 'thal', 'target'
        ]
        
        # Define expected value ranges and types for each column
        self.value_ranges = {
            'age': (20, 100),
            'sex': (0, 1),
            'cp': (0, 3),
            'trestbps': (80, 200),
            'chol': (100, 600),
            'fbs': (0, 1),
            'restecg': (0, 2),
            'thalach': (60, 220),
            'exang': (0, 1),
            'oldpeak': (0, 6.5),
            'slope': (0, 2),
            'ca': (0, 4),
            'thal': (0, 3),
            'target': (0, 1)
        }
    
    
    def load_data(self, file: str) -> dict:
        response = load_data("CLEANER", file)
        
        if response["success"]:
            self.data = response["data"]
            
        return response
        
            
    def save_file(self, output_file: str, index: bool = False, **kwargs) -> dict:
        response = save_file("CLEANER", self.data, output_file)
        
        return response
    
    
    def clean_data(self) -> pd.DataFrame:
        try:
            if self.data is None:
                raise ValueError("[CLEANER] No data loaded. Please load data first using load_data()")
            
            self.logger.info("Starting data cleaning process...")
            print("[CLEANER] Starting data cleaning process...")
            df = self.data.copy()
            df.replace(['?', '', 'NA', 'nan'], np.nan, inplace=True)
            
            # Convert columns to appropriate types
            for col in df.columns:
                try:
                    df[col] = pd.to_numeric(df[col], errors='coerce')  # Coerce non-numeric to NaN
                except ValueError:
                    self.logger.warning(f"[CLEANER] Could not convert {col} to numeric")
            
            # Handle missing values
            initial_missing = df.isnull().sum()
            if initial_missing.any():
                self.logger.info(f"[CLEANER] Found missing values:\n{initial_missing[initial_missing > 0]}")
                print(f"[CLEANER] Found missing values:\n{initial_missing[initial_missing > 0]}")
                
                # For categorical columns, fill with mode
                categorical_cols = ['sex', 'cp', 'fbs', 'restecg', 'exang', 'slope', 'ca', 'thal', 'target']
                for col in categorical_cols:
                    if df[col].isnull().any():
                        df[col].fillna(df[col].mode()[0], inplace=True)
                
                # For numerical columns, fill with median
                numerical_cols = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak']
                for col in numerical_cols:
                    if df[col].isnull().any():
                        df[col].fillna(df[col].median(), inplace=True)
            
            # Handle outliers using IQR method for numerical columns
            for col in numerical_cols:
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                
                # Cap outliers at boundaries
                df[col] = df[col].clip(lower_bound, upper_bound)
            
            # Validate values are within expected ranges
            for col, (min_val, max_val) in self.value_ranges.items():
                invalid_mask = (df[col] < min_val) | (df[col] > max_val)
                if invalid_mask.any():
                    self.logger.warning(f"[CLEANER] Found {invalid_mask.sum()} invalid values in {col}")
                    print(f"[CLEANER] Found {invalid_mask.sum()} invalid values in {col}")
                    # Replace invalid values with median for numerical or mode for categorical
                    if col in categorical_cols:
                        df.loc[invalid_mask, col] = df[col].mode()[0]
                    else:
                        df.loc[invalid_mask, col] = df[col].median()
            
            # Round values to appropriate decimals
            df['oldpeak'] = df['oldpeak'].round(1)
            df = df.round(0).astype(int, errors='ignore')  # Avoid errors on non-numeric columns
            
            # Log cleaning results
            self.logger.info(f"[CLEANER] Data cleaning completed. Final shape: {df.shape}")
            print(f"[CLEANER] Data cleaning completed. Final shape: {df.shape}")
            
            message = {
                "success" : True,
                "message" :None,
                "data" : self.data,
                "error" : None
            }
        except Exception as e:
            error_message = f"[CLEANER] Error loading data: {str(e)}"
            error = traceback.print_exc()
            self.logger.error(error_message)
            
            message = {
                "success" : False,
                "message" : error_message,
                "data" : None,
                "error" : error
            }
            
        return message
    
    
    def get_summary_statistics(self) -> dict:
        if self.data is None:
            raise ValueError("[CLEANER] No data available. Please load and clean data first.")
            
        stats = {
            'shape': self.data.shape,
            'missing_values': self.data.isnull().sum().to_dict(),
            'descriptive_stats': self.data.describe().to_dict(),
            'value_counts': {col: self.data[col].value_counts().to_dict() 
                           for col in self.data.columns}
        }
        
        return stats