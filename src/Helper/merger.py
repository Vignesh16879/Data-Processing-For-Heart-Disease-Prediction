import os
import logging
import traceback
import pandas as pd

from typing import Optional


class MERGER:
    def __init__(self, logger: Optional[logging.Logger] = None):
        if logger is None:
            logging.basicConfig(level=logging.INFO)
            self.logger = logging.getLogger(__name__)
        else:
            self.logger = logger
            
        self.merged_data = pd.DataFrame()
    
    
    def load_csv_file(self, file):
        try:
            data = pd.read_csv(file)
            
            if "ChestPainType" in data.columns:
                data.rename(columns={
                    "Age": "age",
                    "Sex": "sex",
                    "ChestPainType": "cp",
                    "RestingBP": "trestbps",
                    "Cholesterol": "chol",
                    "FastingBS": "fbs",
                    "RestingECG": "restecg",
                    "MaxHR": "thalach",
                    "ExerciseAngina": "exang",
                    "Oldpeak": "oldpeak",
                    "ST_Slope": "slope",
                    "HeartDisease": "target"
                }, inplace=True)
                
                data["ca"] = None
                data["thal"] = None
                
            self.merged_data = pd.concat([self.merged_data, data], ignore_index=True)
            self.logger.info(f"[MERGER] Successfully loaded file: {file}")
            
            message = {
                "success": True,
                "message" : None,
                "data" : None
            }
        except Exception as e:
            error = traceback.print_exc()
            self.logger.error(f"[MERGER] Error loading file {file}: {e}")
            self.logger.error(error)
            
            message = {
                "success": False,
                "message" : None,
                "data" : None
            }
        
        return message


    def save_merged_data(self, output_file):
        try:
            self.merged_data.to_csv(output_file, index=False)
            self.logger.info(f"[MERGER] Merged data successfully saved to {output_file}")
            
            message = {
                "success" : True
            }
        except Exception as e:
            error = traceback.print_exc()
            self.logger.error(f"[MERGER] Error saving file {output_file}: {e}")
            self.logger.error(error)
            
            message = {
                "success" : False,
                "message" : None,
                "data" : None
            }
        
        return message
    
    
    def save_merged_data(self, output_file: str, index: bool = False, **kwargs) -> None:
        if self.merged_data is None:
            self.logger.error("[MERGER] No data available to save. Please load and clean data first.")
            
            message = {
                "success" : False,
                "message" : None,
                "data" : None
            }
            
            return message
            
        try:
            output_dir = os.path.dirname(output_file)
            
            if output_dir and not os.path.exists(output_dir):
                os.makedirs(output_dir)
            
            file_ext = os.path.splitext(output_file)[1].lower()
            
            if file_ext == '.csv':
                self.merged_data.to_csv(output_file, index=index, **kwargs)
            elif file_ext == '.xlsx':
                self.merged_data.to_excel(output_file, index=index, **kwargs)
            elif file_ext == '.parquet':
                self.merged_data.to_parquet(output_file, index=index, **kwargs)
            elif file_ext == '.pickle':
                self.merged_data.to_pickle(output_file, **kwargs)
            else:
                raise ValueError(f"[MERGER] Unsupported file format: {file_ext}. "
                               "Supported formats are: .csv, .xlsx, .parquet, .pickle")
            
            self.logger.info(f"Data successfully saved to {output_file}")
            
            message = {
                "success" : True,
                "message" : None,
                "data" : None
            }
        except Exception as e:
            error = traceback.print_exc()
            self.logger.error(f"[MERGER] Error saving file: {str(e)}")
            self.logger.error("[MERGER] ", error)
            
            message = {
                "success" : False,
                "message" : None,
                "data" : None
            }
        
        return message