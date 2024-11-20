import logging
from typing import Optional

from pytorch_lightning.utilities import rank_zero_only


LOG_LEVELS = (
    "debug",
    "info",
    "warning",
    "error",
    "exception",
    "fatal",
    "critical",
)


def get_logger(name: Optional[str] = None) -> logging.Logger:
    """Create logger with multi-GPU-friendly python command line logger.

    Ensure all logging levels get marked with the rank zero decorator, otherwise logs would get
    multiplied for each GPU process in multi-GPU setup.
    """
    logger = logging.getLogger(name)

    for level in LOG_LEVELS:
        setattr(logger, level, rank_zero_only(getattr(logger, level)))

    return logger
