import json
import os

def load_config(module_path):
    """
    Load configuration from a JSON file.
    
    Args:
        module_path: The path to the module that needs the config (usually __file__)
        
    Returns:
        dict: The configuration dictionary
    """
    config_path = os.path.join(os.path.dirname(os.path.abspath(module_path)), "config.json")
    
    with open(config_path, "r") as f:
        config = json.load(f)
    
    return config 