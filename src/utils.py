import os
import yaml
import logging
from datetime import datetime
from typing import Dict

def load_config(config_path: str = "config.yaml") -> dict:
    """Loads the YAML configuration file into a dictionary.
    
    Args:
        config_path (str): Relative or absolute path to the YAML file.
        
    Returns:
        dict: Parsed configuration parameters.
    """
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config

def setup_experiment_dir(base_dir: str) -> Dict[str, str]:
    """Scaffolds a timestamped output directory for the current execution.
    
    Args:
        base_dir (str): Core directory to house all subsequent runs.
        
    Returns:
        Dict[str, str]: Dictionary mapping folder identifiers to their specific paths.
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_dir = os.path.join(base_dir, f"run_{timestamp}")
    
    dirs = {
        'base': exp_dir,
    }
    
    for d in dirs.values():
        os.makedirs(d, exist_ok=True)
        
    return dirs

def get_logger(name: str, log_dir: str) -> logging.Logger:
    """Initializes an MLOps-compliant tracking logger.
    
    Redirects stdout logging to both the console and a persistent execution log file,
    ensuring thread safety by verifying handler existence.
    
    Args:
        name (str): The namespace of the logger.
        log_dir (str): Target directory to save the output text log.
        
    Returns:
        logging.Logger: Instantiated logger ready for standard tracking integration.
    """
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    
    if not logger.handlers:
        file_handler = logging.FileHandler(os.path.join(log_dir, "pipeline.log"))
        file_format = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
        file_handler.setFormatter(file_format)
        
        console_handler = logging.StreamHandler()
        console_format = logging.Formatter('%(message)s')
        console_handler.setFormatter(console_format)
        
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)
        
    return logger