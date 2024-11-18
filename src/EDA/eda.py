import os
import traceback
import logging
import pandas as pd
import numpy as np
import seaborn as sns
import scipy.stats as stats
import matplotlib.pyplot as plt

from typing import Optional

from ..Helper.dataloader import load_data


class EDA:
    def __init__(self, data: Optional[pd.DataFrame] = None, logger: Optional[logging.Logger] = None):
        if logger is None:
            logging.basicConfig(level=logging.INFO)
            self.logger = logging.getLogger(__name__)
        else:
            self.logger = logger
            
        self.data = data
        self.graph_dir = "../Graphs/"
        
    
    def set_graphs_dir(self, graph_dir: str) -> dict:
        self.graph_dir = graph_dir
        
        # Create the graph directory if it doesn't exist
        if not os.path.exists(self.graph_dir):
            os.makedirs(self.graph_dir)


    def load_data(self, file: str) -> dict:
        response = load_data("HypothesisTests", file)
        
        if response["success"]:
            self.data = response["data"]
            
        return response
    

    def basic_info(self):
        try:
            self.logger.info("[EDA] Basic Information:")
            self.logger.info(f"[EDA] Data Info:\n{self.data.info()}")
            self.logger.info(f"[EDA] First few rows of the data:\n{self.data.head()}")
            self.logger.info(f"[EDA] Missing values:\n{self.data.isnull().sum()}")
            self.logger.info(f"[EDA] Summary statistics for numerical columns:\n{self.data.describe()}")
            # Check if there are categorical columns
            categorical_cols = self.data.select_dtypes(include=[object]).columns
            
            if len(categorical_cols) > 0:
                self.logger.info(f"[EDA] Summary statistics for categorical columns:\n{self.data[categorical_cols].describe()}")
            else:
                self.logger.info("[EDA] No categorical columns found.")
            
            return {
                "success": True,
                "message": None,
                "data": None,
                "error": None
            }
        except Exception as e:
            error = traceback.format_exc()
            self.logger.error(f"[EDA] Error in basic_info: {str(e)}")
            self.logger.error(f"[EDA] {error}")
            
            return {
                "success": False, 
                "message": str(e), 
                "data": None,
                "error" : error
            }


    def save_plot(self, plot_name: str):
        file_path = os.path.join(self.graph_dir, f"{plot_name}.png")
        plt.savefig(file_path)
        plt.close()
        self.logger.info(f"[EDA] Plot saved to {file_path}")
     
        
    def plot_histograms(self):
        try:
            numerical_cols = self.data.select_dtypes(include=[np.number]).columns
            self.data[numerical_cols].hist(figsize=(12, 10), bins=20)
            plt.tight_layout()
            self.save_plot("histograms")
            
            return {
                "success": True,
                "message": None,
                "data": None,
                "error": None
            }
        except Exception as e:
            error = traceback.format_exc()
            self.logger.error(f"[EDA] Error in plot_histograms: {str(e)}")
            self.logger.error(f"[EDA] {error}")
            
            return {
                "success": False, 
                "message": str(e), 
                "data": None,
                "error" : error
            }


    def plot_boxplots(self):
        try:
            numerical_cols = self.data.select_dtypes(include=[np.number]).columns
            for col in numerical_cols:
                plt.figure(figsize=(8, 6))
                sns.boxplot(x=self.data[col])
                plt.title(f"Boxplot for {col}")
                self.save_plot(f"boxplot_{col}")
            
            return {
                "success": True,
                "message": None,
                "data": None,
                "error": None
            }
        except Exception as e:
            error = traceback.format_exc()
            self.logger.error(f"[EDA] Error in plot_boxplots: {str(e)}")
            self.logger.error(f"[EDA] {error}")
            
            return {
                "success": False, 
                "message": str(e), 
                "data": None,
                "error" : error
            }


    def plot_categorical(self):
        try:
            categorical_cols = self.data.select_dtypes(include=[object]).columns
            for col in categorical_cols:
                plt.figure(figsize=(8, 6))
                sns.countplot(data=self.data, x=col)
                plt.title(f"Bar plot for {col}")
                self.save_plot(f"barplot_{col}")
            
            return {
                "success": True,
                "message": None,
                "data": None,
                "error": None
            }
        except Exception as e:
            error = traceback.format_exc()
            self.logger.error(f"[EDA] Error in plot_categorical: {str(e)}")
            self.logger.error(f"[EDA] {error}")
            
            return {
                "success": False, 
                "message": str(e), 
                "data": None,
                "error" : error
            }


    def plot_correlation_heatmap(self):
        try:
            numerical_cols = self.data.select_dtypes(include=[np.number]).columns
            corr_matrix = self.data[numerical_cols].corr()
            plt.figure(figsize=(12, 10))
            sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
            plt.title('Correlation Matrix')
            self.save_plot("correlation_heatmap")
            
            return {
                "success": True,
                "message": None,
                "data": None,
                "error": None
            }
        except Exception as e:
            error = traceback.format_exc()
            self.logger.error(f"[EDA] Error in plot_correlation_heatmap: {str(e)}")
            self.logger.error(f"[EDA] {error}")
            
            return {
                "success": False, 
                "message": str(e), 
                "data": None,
                "error" : error
            }


    def plot_pairplots(self):
        try:
            numerical_cols = self.data.select_dtypes(include=[np.number]).columns
            sns.pairplot(self.data[numerical_cols])
            self.save_plot("pairplot")
            
            return {
                "success": True,
                "message": None,
                "data": None,
                "error": None
            }
        except Exception as e:
            error = traceback.format_exc()
            self.logger.error(f"[EDA] Error in plot_pairplots: {str(e)}")
            self.logger.error(f"[EDA] {error}")
            
            return {
                "success": False, 
                "message": str(e), 
                "data": None,
                "error" : error
            }


    def plot_scatter(self):
        try:
            numerical_cols = self.data.select_dtypes(include=[np.number]).columns
            for col1 in numerical_cols:
                for col2 in numerical_cols:
                    if col1 != col2:
                        plt.figure(figsize=(8, 6))
                        sns.scatterplot(data=self.data, x=col1, y=col2)
                        plt.title(f"Scatter plot: {col1} vs {col2}")
                        self.save_plot(f"scatter_{col1}_vs_{col2}")
            
            return {
                "success": True,
                "message": None,
                "data": None,
                "error": None
            }
        except Exception as e:
            error = traceback.format_exc()
            self.logger.error(f"[EDA] Error in plot_scatter: {str(e)}")
            self.logger.error(f"[EDA] {error}")
            
            return {
                "success": False, 
                "message": str(e), 
                "data": None,
                "error" : error
            }


    def grouped_analysis(self, target_col: str):
        try:
            numerical_cols = self.data.select_dtypes(include=[np.number]).columns
            for col in numerical_cols:
                plt.figure(figsize=(8, 6))
                sns.boxplot(data=self.data, x=target_col, y=col)
                plt.title(f"Boxplot: {col} by {target_col}")
                self.save_plot(f"boxplot_{col}_by_{target_col}")

            categorical_cols = self.data.select_dtypes(include=[object]).columns
            for col in categorical_cols:
                plt.figure(figsize=(8, 6))
                sns.countplot(data=self.data, x=col, hue=target_col)
                plt.title(f"Count plot: {col} by {target_col}")
                self.save_plot(f"countplot_{col}_by_{target_col}")
            
            self.logger.info(f"[EDA] Grouped analysis by target column '{target_col}' completed.")
            
            return {
                "success": True,
                "message": None,
                "data": None,
                "error": None
            }
        except Exception as e:
            error = traceback.format_exc()
            self.logger.error(f"[EDA] Error in grouped_analysis: {str(e)}")
            self.logger.error(f"[EDA] {error}")
            
            return {
                "success": False, 
                "message": str(e), 
                "data": None,
                "error" : error
            }


    def check_missing_values(self):
        try:
            plt.figure(figsize=(10, 6))
            sns.heatmap(self.data.isnull(), cbar=False, cmap='viridis')
            plt.title('Missing Values Heatmap')
            self.save_plot("missing_values_heatmap")
            
            return {
                "success": True,
                "message": None,
                "data": None,
                "error": None
            }
        except Exception as e:
            error = traceback.format_exc()
            self.logger.error(f"[EDA] Error in check_missing_values: {str(e)}")
            self.logger.error(f"[EDA] {error}")
            
            return {
                "success": False, 
                "message": str(e), 
                "data": None,
                "error" : error
            }

    
    def outlier_detection_iqr(self):
        try:
            numerical_cols = self.data.select_dtypes(include=[np.number]).columns
            Q1 = self.data[numerical_cols].quantile(0.25)
            Q3 = self.data[numerical_cols].quantile(0.75)
            IQR = Q3 - Q1
            outliers = ((self.data[numerical_cols] < (Q1 - 1.5 * IQR)) | 
                        (self.data[numerical_cols] > (Q3 + 1.5 * IQR))).any(axis=1)
            self.logger.info(f"[EDA] Outliers detected (IQR method): {np.sum(outliers)}")
            
            return {
                "success": True,
                "message": None,
                "data": None,
                "error": None
            }
        except Exception as e:
            error = traceback.format_exc()
            self.logger.error(f"[EDA] Error in outlier_detection_iqr: {str(e)}")
            self.logger.error(f"[EDA] {error}")
            
            return {
                "success": False, 
                "message": str(e), 
                "data": None,
                "error" : error
            }
            

    def outlier_detection_zscore(self, threshold: float = 3):
        try:
            numerical_cols = self.data.select_dtypes(include=[np.number]).columns
            z_scores = np.abs(stats.zscore(self.data[numerical_cols]))
            outliers = (z_scores > threshold).all(axis=1)
            self.logger.info(f"[EDA] Outliers detected (Z-score > {threshold}): {np.sum(outliers)}")
            
            return {
                "success": True,
                "message": None,
                "data": None,
                "error": None
            }
        except Exception as e:
            error = traceback.format_exc()
            self.logger.error(f"[EDA] Error in outlier_detection_zscore: {str(e)}")
            self.logger.error(f"[EDA] {error}")
            
            return {
                "success": False, 
                "message": str(e), 
                "data": None,
                "error" : error
            }


    def run_all(self):
        try:
            self.basic_info()
            self.plot_histograms()
            self.plot_boxplots()
            self.plot_categorical()
            self.plot_correlation_heatmap()
            self.plot_pairplots()
            self.plot_scatter()
            self.grouped_analysis(target_col='target')
            self.check_missing_values()
            self.outlier_detection_zscore()
            self.outlier_detection_iqr()
            self.logger.info("[EDA] All EDA steps completed.")
        except Exception as e:
            error = traceback.format_exc()
            self.logger.error(f"[EDA] {error}")
            
            return {
                "success": False, 
                "message": str(e), 
                "data": None,
                "error" : error
            }