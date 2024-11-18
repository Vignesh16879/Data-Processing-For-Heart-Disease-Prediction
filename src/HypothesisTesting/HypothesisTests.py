import os
import logging
import traceback
import pandas as pd
import scipy.stats as stats

from typing import Optional

from ..Helper.dataloader import load_data


class HypothesisTests:
    def __init__(self, logger: Optional[logging.Logger] = None, alpha = 0.05):
        if logger is None:
            logging.basicConfig(level=logging.INFO)
            self.logger = logging.getLogger(__name__)
        else:
            self.logger = logger
            
        self.data = None
        self.alpha = alpha


    def load_data(self, file: str) -> dict:
        response = load_data("HypothesisTests", file)
        
        if response["success"]:
            self.data = response["data"]
            
        return response
    

    def _get_result(self, test_name, p_value, null_hypothesis, alt_hypothesis):
        self.logger.info(f"[HypothesisTests] {test_name}")
        self.logger.info(f"[HypothesisTests] Null Hypothesis: {null_hypothesis}")
        self.logger.info(f"[HypothesisTests] Alternative Hypothesis: {alt_hypothesis}")
        self.logger.info(f"[HypothesisTests] p-value = {p_value:.4f}")

        if p_value < self.alpha:
            self.logger.info(f"[HypothesisTests] Result: Reject the null hypothesis (p < {self.alpha})")
        else:
            self.logger.info(f"[HypothesisTests] Result: Accept the null hypothesis (p >= {self.alpha})")
    

    def test_age(self):
        try:
            null_hypothesis = "There is no significant difference in the mean age between individuals with heart disease and those without."
            alt_hypothesis = "There is a significant difference in the mean age between individuals with heart disease and those without."
            
            heart_disease = self.data[self.data['target'] == 1]['age']
            no_heart_disease = self.data[self.data['target'] == 0]['age']
            t_stat, p_value = stats.ttest_ind(heart_disease, no_heart_disease)
            self._get_result("Age and Heart Disease Test", p_value, null_hypothesis, alt_hypothesis)
            
            return {
                "success": True, 
                "message": None, 
                "data": {
                    't_stat': t_stat,
                    'p_value': p_value
                },
                "error" : None
            }
        except Exception as e:
            error = traceback.format_exc()
            self.logger.error(f"[HypothesisTests] Error in test_age: {str(e)}")
            self.logger.error(f"[HypothesisTests] {error}")
            
            return {
                "success": False, 
                "message": str(e), 
                "data": None,
                "error" : error
            }


    def test_cholesterol(self):
        try:
            null_hypothesis = "There is no significant difference in the mean cholesterol levels between individuals with heart disease and those without."
            alt_hypothesis = "There is a significant difference in the mean cholesterol levels between individuals with heart disease and those without."
            
            heart_disease = self.data[self.data['target'] == 1]['chol']
            no_heart_disease = self.data[self.data['target'] == 0]['chol']
            t_stat, p_value = stats.ttest_ind(heart_disease, no_heart_disease)
            self._get_result("Cholesterol and Heart Disease Test", p_value, null_hypothesis, alt_hypothesis)
            
            return {
                "success": True, 
                "message": None, 
                "data": {
                    't_stat': t_stat,
                    'p_value': p_value
                },
                "error" : None
            }
        except Exception as e:
            error = traceback.format_exc()
            self.logger.error(f"[HypothesisTests] Error in test_cholesterol: {str(e)}")
            self.logger.error(f"[HypothesisTests] {error}")
            
            return {
                "success": False, 
                "message": str(e), 
                "data": None,
                "error" : error
            }


    def test_blood_pressure(self):
        try:
            null_hypothesis = "There is no linear relationship between blood pressure and the presence of heart disease."
            alt_hypothesis = "There is a linear relationship between blood pressure and the presence of heart disease."
            
            corr, p_value = stats.pearsonr(self.data['trestbps'], self.data['target'])
            self._get_result("Blood Pressure Correlation Test", p_value, null_hypothesis, alt_hypothesis)
            
            return {
                "success": True, 
                "message": None, 
                "data": {
                    'corr': corr,
                    'p_value': p_value
                },
                "error" : None
            }
        except Exception as e:
            error = traceback.format_exc()
            self.logger.error(f"[HypothesisTests] Error in test_blood_pressure: {str(e)}")
            self.logger.error(f"[HypothesisTests] {error}")
            
            return {
                "success": False, 
                "message": str(e), 
                "data": None,
                "error" : error
            }


    def test_chest_pain(self):
        try:
            null_hypothesis = "There is no association between chest pain type and the presence of heart disease."
            alt_hypothesis = "There is an association between chest pain type and the presence of heart disease."
            
            contingency_table = pd.crosstab(self.data['cp'], self.data['target'])
            chi2, p_value, dof, expected = stats.chi2_contingency(contingency_table)
            self._get_result("Chest Pain Type and Heart Disease Test", p_value, null_hypothesis, alt_hypothesis)
            
            return {
                "success": True, 
                "message": None, 
                "data": {
                    'chi2': chi2,
                    'p_value': p_value,
                    'dof': dof,
                    'expected': expected
                },
                "error" : None
            }
        except Exception as e:
            error = traceback.format_exc()
            self.logger.error(f"[HypothesisTests] Error in test_chest_pain: {str(e)}")
            self.logger.error(f"[HypothesisTests] {error}")
            
            return {
                "success": False, 
                "message": str(e), 
                "data": None,
                "error" : error
            }


    def test_slope(self):
        try:
            null_hypothesis = "There is no association between the slope of the ST segment and the presence of heart disease."
            alt_hypothesis = "There is an association between the slope of the ST segment and the presence of heart disease."
            
            contingency_table = pd.crosstab(self.data['slope_Flat'], self.data['target'])
            chi2, p_value, dof, expected = stats.chi2_contingency(contingency_table)
            self._get_result("Slope of ST Segment and Heart Disease Test", p_value, null_hypothesis, alt_hypothesis)
            
            return {
                "success": True, 
                "message": None, 
                "data": {
                    'chi2': chi2,
                    'p_value': p_value,
                    'dof': dof,
                    'expected': expected
                },
                "error" : None
            }
        except Exception as e:
            error = traceback.format_exc()
            self.logger.error(f"[HypothesisTests] Error in test_slope: {str(e)}")
            self.logger.error(f"[HypothesisTests] {error}")
            
            return {
                "success": False, 
                "message": str(e), 
                "data": None,
                "error" : error
            }


    def run_all_tests(self):
        try:
            self.logger.info("[HypothesisTests] Running Hypothesis Tests...")
            self.test_age()
            self.test_cholesterol()
            self.test_blood_pressure()
            self.test_chest_pain()
            self.test_slope()
            
            return {
                "success": True, 
                "message": None, 
                "data": None,
                "error" : None
            }
        except Exception as e:
            error = traceback.format_exc()
            self.logger.error(f"[HypothesisTests] Error in run_all_tests: {str(e)}")
            self.logger.error(f"[HypothesisTests] {error}")
            
            return {
                "success": False, 
                "message": str(e), 
                "data": None,
                "error" : error
            }