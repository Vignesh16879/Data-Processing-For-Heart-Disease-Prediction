import os
import sys
import logging
import traceback
import pandas as pd

from typing import Optional
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

from .test import Test
from .train import Train


def CheckModel(logger, file, model = None, model_path = None):
    try:
        if not os.path.exists(file):
            raise FileNotFoundError(f"Test data file {file} does not exist.")
        
        # Validate model or model path
        if model is None and model_path is None:
            raise ValueError("Either a model or a model path is required.")
        
        if not hasattr(model, 'estimators_'):
            logger.warning("[TestModel] Model is untrained. Train the model first.")
            
            # Load sample data
            sample_data = pd.read_csv(file, header=0)
            logger.info("[TestModel] Sample data loaded successfully.")

            # Split data into features and target
            X = sample_data.drop(columns=['target']).head(10)   # Train on only 10 instances
            Y = sample_data['target'].head(10)                  # Corresponding target values for 10 instances
            
            # Fill NaN values in Y with the mean
            Y.fillna(Y.mean(), inplace=True)
            Y = pd.cut(Y, bins=3, labels=[0, 1, 2])

            # Train the model on the sample data
            model.fit(X, Y)
            logger.info("[TestModel] Model trained on sample data successfully.")

        # Check if the model is one of the expected types without invoking its methods
        valid_model_types = (RandomForestClassifier, GradientBoostingClassifier)
    
        if model and type(model) not in valid_model_types and not hasattr(model, 'estimators_'):
            raise ValueError(f"Model is not a valid type. Expected one of {valid_model_types}, got {type(model)}")
        
        if model_path and not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file {model_path} does not exist.")

        return {
            "success" : True
        }
    except AttributeError as e:
        error = traceback.format_exc()
        logger.error(f"[TestModel] Model might be untrained or improperly loaded. Error: {e}")
        logger.error(f"{error}")
        
        return {
            "success": False,
            "message": str(e),
            "data": None,
            "error": error
        }
    except Exception as e:
        error = traceback.format_exc()
        logger.error(f"[TestModel] Error in testing process: {str(e)}")
        logger.error(f"{error}")
        
        return {
            "success": False,
            "message": str(e),
            "data": None,
            "error": error
        }
        

def TrainModel(logger: Optional[logging.Logger] = None, file: str = "./", model=None, model_path=None, ):
    logger.info("[TrainModel] Starting model testing process.")
    
    if logger is None:
        logging.basicConfig(level=logging.INFO)
        logger = logging.getLogger(__name__)
            
    try:
        response = CheckModel(
            logger = logger, 
            file = file, 
            model = model, 
            model_path = model_path
        )
        
        if not response["success"]:
            return response
        
        logger.info("[TestModel] Model and path validation completed.")
        
        # Training
        training = Train(
            logger = logger,
            model = model
        )
        training.load_data(file = file)
        response = training.train_model()
        
        return response
    except Exception as e:
        error = traceback.format_exc()
        logger.error(f"[TrainModel] Error in training process: {str(e)}")
        logger.error(f"{error}")
        
        return {
            "success": False,
            "message": str(e),
            "data": None,
            "error": error
        }


def TestModel(logger: Optional[logging.Logger] = None, file: str = "./", model=None, model_path=None):
    logger.info("[TestModel] Starting model testing process.")

    if logger is None:
        logging.basicConfig(level=logging.INFO)
        logger = logging.getLogger(__name__)

    try:
        response = CheckModel(
            logger = logger, 
            file = file, 
            model = model, 
            model_path = model_path
        )
        
        if not response["success"]:
            return response
        
        logger.info("[TestModel] Model and path validation completed.")

        # Create a new instance of the test class
        test_instance = Test(logger=logger, model=model, model_path=model_path)

        logger.info("[TestModel] Data loading initiated.")
        response = test_instance.load_data(str(file))

        if not response["success"]:
            sys.exit(f"Error loading data: {response['message']}")

        logger.info("[TestModel] Data loading completed.")

        # Test model
        test_response = test_instance.test_model()

        if not test_response["success"]:
            sys.exit(f"Error testing model: {test_response['message']}")

        logger.info("[TestModel] Model testing completed successfully.")
        logger.info("Testing completed successfully")
        logger.info(f"Accuracy: {test_response['data']['accuracy']:.4%}")
        logger.info(f"Prediction Time: {test_response['data']['prediction_time']:.4f} seconds")

        return {
            "success": True,
            "message": None,
            "data": {
                "Accuracy": test_response['data']['accuracy'],
                "Prediction Time": test_response['data']['prediction_time']
            },
            "error": None
        }
    except Exception as e:
        error = traceback.format_exc()
        logger.error(f"[TestModel] Error in testing process: {str(e)}")
        logger.error(f"{error}")
        
        return {
            "success": False,
            "message": str(e),
            "data": None,
            "error": error
        }