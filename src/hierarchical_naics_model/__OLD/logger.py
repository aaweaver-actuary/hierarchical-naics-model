import logging


def get_logger(name: str, level: str = "info") -> logging.Logger:
    """Get a named logger for the hierarchical_naics_model package."""
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    handler = logging.StreamHandler()
    handler.setFormatter(formatter)
    logger = logging.getLogger(name)
    if not logger.hasHandlers():
        logger.addHandler(handler)
    log_level = getattr(logging, level.upper(), logging.INFO)
    logger.setLevel(log_level)
    return logger
