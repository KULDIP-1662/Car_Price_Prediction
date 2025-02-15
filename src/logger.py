import os
import sys
import logging

def get_logger(name=__name__, log_level=logging.INFO, log_format=None, log_filename="app.log"):

    logger = logging.getLogger(name)
    if logger.handlers: # Check if logger already has handlers
        return logger # Logger already configured, return it

    log_dir = "logs"
    os.makedirs(log_dir, exist_ok=True)
    log_filepath = os.path.join(log_dir, log_filename)

    if not log_format:
        log_format = "%(asctime)s: %(levelname)s: %(module)s: %(message)s"

    formatter = logging.Formatter(log_format)

    file_handler = logging.FileHandler(log_filepath)
    file_handler.setFormatter(formatter)

    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)

    logger.setLevel(log_level)
    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)
    return logger