"""Small logging helper so every module logs consistently."""
import logging
import sys


def get_logger(name: str, verbose: bool = False) -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:  # avoid duplicate handlers on repeated calls
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(
            logging.Formatter("[%(asctime)s] %(levelname)s %(name)s: %(message)s",
                               datefmt="%H:%M:%S")
        )
        logger.addHandler(handler)
    logger.setLevel(logging.DEBUG if verbose else logging.INFO)
    return logger
