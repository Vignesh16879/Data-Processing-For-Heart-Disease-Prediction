import os
import logging
import traceback
import numpy as np
import pandas as pd

from typing import Optional, Dict, Union, List
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

from ..Helper.dataloader import load_data
from ..Helper.datasaver import save_file


class EncodingCategorical:
    def __init__(self, logger: Optional[logging.Logger] = None):
        if logger is None:
            logging.basicConfig(level=logging.INFO)
            self.logger = logging.getLogger(__name__)
        else:
            self.logger = logger
            
        self.data = None
        self.label_encoders = {}
        self.onehot_encoders = {}
        self.categorical_stats = None
        
        
    def load_data(self, file: str) -> dict:
        response = load_data("CLEANER", file)
        
        if response["success"]:
            self.data = response["data"]
            
        return response
        
            
    def save_file(self, output_file: str, index: bool = False, **kwargs) -> dict:
        response = save_file("CLEANER", self.data, output_file)
        
        return response
    
    
    def _generate_categorical_stats(self) -> None:
        """Generate statistics about categorical variables in the dataset."""
        if self.data is None:
            return
        
        categorical_columns = self.data.select_dtypes(include=['object', 'category']).columns
        
        self.categorical_stats = {
            'total_categorical': len(categorical_columns),
            'categorical_columns': list(categorical_columns),
            'unique_values': {col: self.data[col].nunique() for col in categorical_columns},
            'value_counts': {col: self.data[col].value_counts().to_dict() for col in categorical_columns}
        }
    
    def get_categorical_stats(self) -> Dict:
        """Return statistics about categorical variables."""
        if self.categorical_stats is None:
            self._generate_categorical_stats()
        return self.categorical_stats
    
    def encode_categorical(self, 
        label_encode: List[str] = None,
        onehot_encode: List[str] = None,
        drop_first: bool = True,
        max_categories: int = None
    ) -> Dict[str, Union[bool, str, pd.DataFrame]]:
        try:
            if self.data is None:
                raise ValueError("[ENCODER] No data loaded. Please load data first.")
            
            # Make a copy of the data
            df = self.data.copy()
            
            # Label Encoding
            if label_encode:
                for col in label_encode:
                    if col in df.columns:
                        le = LabelEncoder()
                        df[col] = le.fit_transform(df[col].astype(str))
                        self.label_encoders[col] = le
                        self.logger.info(f"[ENCODER] Applied label encoding to {col}")
                        
                        # Store mapping
                        mapping = dict(zip(le.classes_, le.transform(le.classes_)))
                        self.logger.info(f"[ENCODER] Mapping for {col}: {mapping}")
            
            # One-Hot Encoding
            if onehot_encode:
                for col in onehot_encode:
                    if col in df.columns:
                        # Check number of unique values
                        if max_categories and df[col].nunique() > max_categories:
                            self.logger.warning(f"[ENCODER] Column {col} has more than {max_categories} categories. Skipping.")
                            continue
                        
                        # Create dummy variables
                        dummies = pd.get_dummies(df[col], prefix=col, drop_first=drop_first)
                        
                        # Drop original column and add dummy columns
                        df = pd.concat([df.drop(col, axis=1), dummies], axis=1)
                        self.logger.info(f"[ENCODER] Applied one-hot encoding to {col}")
            
            self.data = df
            self._generate_categorical_stats()
            
            return {
                "success": True,
                "message": "Categorical variables encoded successfully",
                "data": self.data
            }
            
        except Exception as e:
            self.logger.error(f"[ENCODER] Error encoding categorical variables: {str(e)}")
            self.logger.error(traceback.format_exc())
            return {
                "success": False,
                "message": str(e),
                "data": None
            }