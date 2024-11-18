import sys
import logging

from typing import Optional


def AnErrorOccured(message, response, logger: Optional[logging.Logger] = None):
    logger.error(f"{message} Message from kernel: \n{response['message']}")
    logger.error(response['error'])
    logger.info(f"Can't continue. \nExiting System...")
    sys.exit(0)


def ApplyPreProccessingTask(task, task_class, sub_func, sub_func_args, input_file, output_file, logger: Optional[logging.Logger] = None):
    if logger is None:
        logging.basicConfig(level=logging.INFO)
        logger = logging.getLogger(__name__)
    
    info = f"Applying {task} to data."
    logger.info(info)
    
    info = f"Creating {task}..."
    logger.info(info)
    handle = task_class(logger=logger)
    
    info = f"Loading data..."
    logger.info(info)
    
    response = handle.load_data(input_file)
    
    if response["success"]:
        info = f"Data loaded successfully."
        logger.info(info)
        
        info = f"Starting prcossing data..."
        logger.info(info)
        
        if sub_func_args:
            kwargs = sub_func_args 
            response = getattr(handle, sub_func)(**kwargs)
        else:
            response = getattr(handle, sub_func)()
        
        if response["success"]:
            info = f"Data processed successfully."
            logger.info(info)
            
            info = f"Saving file to {output_file}."
            logger.info(info)
            
            response = handle.save_file(output_file)
            
            if response["success"]:
                info = f"Data saved to file-{output_file}."
                logger.info(info)
            else:
                AnErrorOccured(
                    message = f"An error occured while saving the data to file-{output_file}.", 
                    response = response, 
                    logger = logger
                )
        else:
            AnErrorOccured(
                message = f"An error occured while processing the data.", 
                response = response, 
                logger = logger
            )
    else:
        AnErrorOccured(
            message = f"An error occured while loading data.", 
            response = response, 
            logger = logger
        )