import logging
from typing import *
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
    # for blob finding
    gaussian_sigma: float = 1.0
    min_size: int = 20 ** 2
    max_size: Optional[int] = None
    max_hole: Optional[int] = 40**2
    # for tracking
    search_range: float = 500.
    # others
    threshold_adjustment: float = -1.5

    lower_intensity: float = 0.5
    upper_intensity: float = 1.5
    dilation: int = 3
    debug: bool = False


def user_cmd(prompt: str, choices: str):
    choice = None
    while choice not in set(choices):
        choice = input(prompt).lower()
        # choice = get_chr().lower()
    return choice


logger = get_logger()
cfg = Config()
