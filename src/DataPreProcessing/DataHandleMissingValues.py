import os
import logging
import traceback
import pandas as pd
import numpy as np

from sklearn.impute import SimpleImputer
from typing import Optional, Dict, Union, List

from ..Helper.dataloader import load_data
from ..Helper.datasaver import save_file


class HandleMissingValues:
    def __init__(self, logger: Optional[logging.Logger] = None):
        if logger is None:
            logging.basicConfig(level=logging.INFO)
            self.logger = logging.getLogger(__name__)
        else:
            self.logger = logger
            
        self.data = None
        self.missing_stats = None
        self.imputer = None
        
        
    def load_data(self, file: str) -> dict:
        response = load_data("CLEANER", file)
        
        if response["success"]:
            self.data = response["data"]
            
        return response
        
            
    def save_file(self, output_file: str, index: bool = False, **kwargs) -> dict:
        response = save_file("CLEANER", self.data, output_file)
        
        return response
    
    
    def _generate_missing_stats(self) -> None:
        if self.data is None:
            return
        
        self.missing_stats = {
            'total_missing': self.data.isnull().sum().sum(),
            'missing_by_column': self.data.isnull().sum().to_dict(),
            'missing_percentage': (self.data.isnull().sum() / len(self.data) * 100).to_dict(),
            'columns_with_missing': list(self.data.columns[self.data.isnull().any()]),
        }
        
        
    def get_missing_stats(self) -> Dict:
        if self.missing_stats is None:
            self._generate_missing_stats()
        return self.missing_stats
    
    
    def handle_missing(
        self, 
        strategy: Dict[str, str] = None, 
        fill_values: Dict[str, Union[str, int, float]] = None,
        threshold: float = None
    ) -> Dict[str, Union[bool, str, pd.DataFrame]]:
        try:
            if self.data is None:
                raise ValueError("[HandleMissingValues] No data loaded. Please load data first.")
            
            # Make a copy of the data
            df = self.data.copy()
            
            # Handle threshold-based dropping if specified
            if threshold is not None:
                if not 0 <= threshold <= 1:
                    raise ValueError("[HandleMissingValues] Threshold must be between 0 and 1")
                
                # Drop columns with missing values above threshold
                cols_to_drop = df.columns[df.isnull().mean() > threshold]
                df.drop(columns=cols_to_drop, inplace=True)
                if cols_to_drop.any():
                    self.logger.info(f"[HandleMissingValues] Dropped columns {list(cols_to_drop)} due to missing values above threshold")
                
                # Drop rows with missing values above threshold
                df.dropna(thresh=int((1-threshold) * df.shape[1]), inplace=True)
            
            # Handle specific fill values
            if fill_values:
                for col, value in fill_values.items():
                    if col in df.columns:
                        df[col].fillna(value, inplace=True)
                        self.logger.info(f"[HandleMissingValues] Filled missing values in {col} with {value}")
            
            # Handle imputation strategies
            if strategy:
                for col, strat in strategy.items():
                    if col not in df.columns:
                        continue
                        
                    if strat in ['mean', 'median', 'most_frequent', 'constant']:
                        imputer = SimpleImputer(strategy=strat)
                        df[col] = imputer.fit_transform(df[[col]]).ravel()  # Flatten to 1D
                        self.logger.info(f"[HandleMissingValues] Applied {strat} imputation to {col}")
                    else:
                        self.logger.warning(f"[HandleMissingValues] Unknown strategy {strat} for column {col}")
            
            self.data = df
            self._generate_missing_stats()
            
            return {
                "success": True,
                "message": "Missing values handled successfully",
                "data": self.data
            }
            
        except Exception as e:
            self.logger.error(f"[HandleMissingValues] Error handling missing values: {str(e)}")
            self.logger.error(traceback.format_exc())
            return {
                "success": False,
                "message": str(e),
                "data": None
            }