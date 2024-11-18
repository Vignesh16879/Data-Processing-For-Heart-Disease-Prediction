import os
import logging
import traceback
import pandas as pd

from typing import Optional, Union

        
def save_file(func: str, data: Union[pd.DataFrame, None], output_file: str, logger: Optional[logging.Logger] = None, index: bool = False, **kwargs) -> dict:
    try:
        if logger is None:
            logging.basicConfig(level=logging.INFO)
            logger = logging.getLogger(__name__)
            
        if data is None:
            raise ValueError("No data available to save. Please load and balance data first.")
        
        output_dir = os.path.dirname(output_file)
        
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        if output_file.endswith('.csv'):
            data.to_csv(output_file, index=index, **kwargs)
        elif output_file.endswith('.xlsx'):
            data.to_excel(output_file, index=index, **kwargs)
        else:
            raise ValueError("Unsupported file format. Use .csv or .xlsx")

        logger.info(f"[{func}] Data successfully saved to {output_file}")
        
        return {
            "success": True, 
            "message": None,
            "data": None,
            "error": None
        }
    except Exception as e:
        error = traceback.print_exc()
        logger.error(f"[{func}] Error saving file: {str(e)}")
        logger.error(f"[{func}] {error}")
        
        return {
            "success": False, 
            "message": str(e), 
            "data": None,
            "error" : error
        }