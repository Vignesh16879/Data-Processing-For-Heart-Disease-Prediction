import os
import logging
import traceback
import pandas as pd

from typing import Optional, List
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import SelectKBest, f_classif

from ..Helper.dataloader import load_data
from ..Helper.datasaver import save_file


class FeatureSelector:
    def __init__(self, logger: Optional[logging.Logger] = None):
        if logger is None:
            logging.basicConfig(level=logging.INFO)
            self.logger = logging.getLogger(__name__)
        else:
            self.logger = logger
        
        self.data = None
        self.selected_features = None


    def load_data(self, file: str) -> dict:
        response = load_data("CLEANER", file)
        
        if response["success"]:
            self.data = response["data"]
            
        return response
        
            
    def save_file(self, output_file: str, index: bool = False, **kwargs) -> dict:
        response = save_file("CLEANER", self.data, output_file)
        
        return response
        

    def select_k_best(self, target: str, k: int) -> List[str]:
        try:
            self.logger.info(f"[FeatureSelector] Selecting top {k} features using SelectKBest.")
            
            X = self.data.drop(columns=[target])
            y = self.data[target]
            
            selector = SelectKBest(score_func=f_classif, k=k)
            selector.fit(X, y)

            selected_features = X.columns[selector.get_support()].tolist()
            self.logger.info(f"[FeatureSelector] Selected features: {selected_features}")
            
            self.selected_features = selected_features
            
            return {
                "success": True, 
                "message": None,
                "data": selected_features,
                "error": None
            }
        except Exception as e:
            error = traceback.print_exc()
            self.logger.error(f"[FeatureSelector] Error selecting {k} features: {str(e)}")
            self.logger.error(f"[FeatureSelector] {error}")
            
            return {
                "success": False, 
                "message": str(e), 
                "data": None,
                "error" : error
            }


    def recursive_feature_elimination(self, target: str, n_features_to_select: int) -> List[str]:
        try:
            self.logger.info(f"[FeatureSelector] Performing RFE to select {n_features_to_select} features.")
            
            X = self.data.drop(columns=[target])
            y = self.data[target]
            
            model = LogisticRegression(solver='liblinear')
            rfe = RFE(model, n_features_to_select=n_features_to_select)
            fit = rfe.fit(X, y)
            
            selected_features = X.columns[fit.support_].tolist()
            self.logger.info(f"[FeatureSelector] Selected features: {selected_features}")
            
            self.selected_features = selected_features
            
            return {
                "success": True, 
                "message": None,
                "data": selected_features,
                "error": None
            }
        except Exception as e:
            error = traceback.print_exc()
            self.logger.error(f"[FeatureSelector] Error recursive feature elimination: {str(e)}")
            self.logger.error(f"[FeatureSelector] {error}")
            
            return {
                "success": False, 
                "message": str(e), 
                "data": None,
                "error" : error
            }