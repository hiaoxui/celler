import logging
from dataclasses import dataclass


def get_logger():
    logger_ = logging.getLogger('cell')
    fmt = logging.Formatter('{asctime} {message}', style='{')
    stm_hdl = logging.StreamHandler()
    stm_hdl.setFormatter(fmt)
    logger_.addHandler(stm_hdl)
    return logger_


@dataclass()
class Config:
    gaussian_sigma: float
    threshold_adjustment: float
    min_size: int
    max_hole: int
    max_size: int


logger = get_logger()
