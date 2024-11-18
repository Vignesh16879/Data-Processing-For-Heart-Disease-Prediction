import os
import logging
import traceback
import pandas as pd

from typing import Optional


def load_data(func: str, file: str, logger: Optional[logging.Logger] = None) -> dict:
    try:
        if logger is None:
            logging.basicConfig(level=logging.INFO)
            logger = logging.getLogger(__name__)
            
        if file.endswith('.csv'):
            data = pd.read_csv(file)
        elif file.endswith('.xlsx'):
            data = pd.read_excel(file)
        else:
            raise ValueError("Unsupported file format. Use .csv or .xlsx")

        logger.info(f"[{func}] Data loaded successfully from {file}")
        
        return {
            "success": True, 
            "message": None, 
            "data": data,
            "error" : None
        }
    except Exception as e:
        error = traceback.print_exc()
        logger.error(f"[{func}] Error loading data: {str(e)}")
        logger.error(f"[{func}] {error}")
        
        return {
            "success": False, 
            "message": str(e), 
            "data": None,
            "error" : error
        }