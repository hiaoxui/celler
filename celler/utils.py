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
    # for blob finding
    gaussian_sigma: float
    min_size: int
    max_size: int
    # for tracking
    search_range: float
    # others
    threshold_adjustment: float = -1.5
    intensity_variation: float = 0.5
    debug: bool = False


def user_cmd(prompt: str, choices: str):
    choice = None
    while choice not in set(choices):
        choice = input(prompt).lower()
        # choice = get_chr().lower()
    return choice


logger = get_logger()
