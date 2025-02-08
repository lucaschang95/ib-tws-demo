import logging
import os

def setup_logger(clear_log=True):
    script_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    log_file = os.path.join(script_dir, "logs", "app.log")

    # Create logs directory if it doesn't exist
    os.makedirs(os.path.dirname(log_file), exist_ok=True)

    if clear_log:
        with open(log_file, "w") as f:
            f.write("")

    # Configure logging
    logging.basicConfig(
        filename=log_file,
    )
    
    logging.info("setup_logger done...")

    return logging.getLogger(__name__) 