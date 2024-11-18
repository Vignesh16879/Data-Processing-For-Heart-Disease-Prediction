import logging
import traceback
import pandas as pd
import scipy.stats as stats

from .HypothesisTests import HypothesisTests
from typing import Optional


class ValidateHypothesisTesting:
    def __init__(self, logger: Optional[logging.Logger] = None, hypothesis_tests: Optional[HypothesisTests] = None):
        if logger is None:
            logging.basicConfig(level=logging.INFO)
            self.logger = logging.getLogger(__name__)
        else:
            self.logger = logger
            
        if hypothesis_tests is None:
            raise Exception("Must pass proper `HypothesisTests` loaded class variable.")

        self.hypothesis_tests = hypothesis_tests
        self.data = hypothesis_tests.data
        self.alpha = hypothesis_tests.alpha


    def _get_result(self, test_name, p_value):
        self.logger.info(f"[ValidateHypothesisTesting] {test_name}")
        self.logger.info(f"[ValidateHypothesisTesting] p-value = {p_value:.4f}")

        if p_value < self.alpha:
            self.logger.info(f"[ValidateHypothesisTesting] Result: Reject the null hypothesis (p < {self.alpha})")
        else:
            self.logger.info(f"[ValidateHypothesisTesting] Result: Fail to reject the null hypothesis (p ≥ {self.alpha})")


    def validate_age_test(self):
        try:
            self.logger.info("[ValidateHypothesisTesting] Validating Age Hypothesis Test...")
            heart_disease = self.data[self.data['target'] == 1]['age']
            no_heart_disease = self.data[self.data['target'] == 0]['age']

            # Shapiro-Wilk Test for Normality
            stat_hd, p_hd = stats.shapiro(heart_disease)
            stat_no_hd, p_no_hd = stats.shapiro(no_heart_disease)

            self.logger.info("[Validation] Shapiro-Wilk Test for Normality:")
            self.logger.info(f"[Heart Disease] W = {stat_hd:.4f}, p = {p_hd:.4f}")
            self.logger.info(f"[No Heart Disease] W = {stat_no_hd:.4f}, p = {p_no_hd:.4f}")

            if p_hd < self.alpha or p_no_hd < self.alpha:
                self.logger.info("[Validation] Age data is not normally distributed. Using Mann-Whitney U Test.")
                u_stat, p_value = stats.mannwhitneyu(heart_disease, no_heart_disease)
                self._get_result("Mann-Whitney U Test for Age", p_value)
            else:
                self.logger.info("[Validation] Age data is normally distributed. The original t-test is valid.")
            
            return {
                "success": True, 
                "message": None, 
                "data": None,
                "error" : None
            }
        except Exception as e:
            error = traceback.format_exc()
            self.logger.error(f"[ValidateHypothesisTesting] Error in validate_age_test: {str(e)}")
            self.logger.error(f"[ValidateHypothesisTesting] {error}")
            
            return {
                "success": False, 
                "message": str(e), 
                "data": None,
                "error" : error
            }
            

    def validate_chi_squared_assumptions(self, contingency_table):
        try:
            expected = stats.chi2_contingency(contingency_table)[3]
            min_expected = expected.min()
            self.logger.info(f"[ValidateHypothesisTesting] Minimum expected frequency in contingency table: {min_expected:.4f}")
            
            if min_expected < 5:
                self.logger.info("[ValidateHypothesisTesting] Expected frequencies too small. Consider using Fisher's Exact Test.")
            else:
                self.logger.info("[ValidateHypothesisTesting] Assumptions of Chi-squared test are met.")
            
            return {
                "success": True, 
                "message": None, 
                "data": None,
                "error" : None
            }
        except Exception as e:
            error = traceback.format_exc()
            self.logger.error(f"[ValidateHypothesisTesting] Error in validate_chi_squared_assumptions: {str(e)}")
            self.logger.error(f"[ValidateHypothesisTesting] {error}")
            
            return {
                "success": False, 
                "message": str(e), 
                "data": None,
                "error" : error
            }
            

    def validate_chest_pain_test(self):
        try:
            self.logger.info("[ValidateHypothesisTesting] Validating Chest Pain Hypothesis Test...")
            contingency_table = pd.crosstab(self.data['cp'], self.data['target'])
            self.validate_chi_squared_assumptions(contingency_table)
            
            return {
                "success": True, 
                "message": None, 
                "data": None,
                "error" : None
            }
        except Exception as e:
            error = traceback.format_exc()
            self.logger.error(f"[ValidateHypothesisTesting] Error in validate_chest_pain_test: {str(e)}")
            self.logger.error(f"[ValidateHypothesisTesting] {error}")
            
            return {
                "success": False, 
                "message": str(e), 
                "data": None,
                "error" : error
            }
            

    def run_all_tests(self):
        try:
            self.logger.info("[ValidateHypothesisTesting] Validating all tests...")
            self.validate_age_test()
            self.validate_chest_pain_test()
            # Add more validation methods here as needed.
            
            return {
                "success": True, 
                "message": None, 
                "data": None,
                "error" : None
            }
        except Exception as e:
            error = traceback.format_exc()
            self.logger.error(f"[ValidateHypothesisTesting] Error in validate_all_tests: {str(e)}")
            self.logger.error(f"[ValidateHypothesisTesting] {error}")
            
            return {
                "success": False, 
                "message": str(e), 
                "data": None,
                "error" : error
            }