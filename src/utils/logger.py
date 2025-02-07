import logging
import os

def setup_logger(clear_log=True):
    script_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    log_file = os.path.join(script_dir, "src", "lstm", "app.log")

    if clear_log:
        with open(log_file, "w") as f:
          f.write("")

    # Configure logging
    logging.basicConfig(
        filename=log_file,
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )
    
    logging.info("setup_logger done...")

    return logging.getLogger(__name__) 