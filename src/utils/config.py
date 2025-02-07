import json
import os
import sys

def load_config(config_path='config.json'):
    """
    Load configuration from a JSON file.
    
    Args:
        config_path: Path to the config file (default: config.json)
        
    Returns:
        dict: The configuration dictionary
    """
    # Get project root directory using sys.path[0]
    root_dir = os.path.dirname(sys.path[0])

    if not os.path.isabs(config_path):
        config_path = os.path.join(root_dir, config_path)
    
    print(f"Loading config from: {config_path}")
    
    with open(config_path, "r") as f:
        config = json.load(f)
    
    return config