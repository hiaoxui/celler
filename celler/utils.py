import logging
import sys
from dataclasses import dataclass
import termios
import tty


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
    frames_per_step: int = 10
    intensity_variation: float = 0.5


logger = get_logger()

logging.getLogger('trackpy.linking.linking.link_iter').setLevel('WARNING')


def get_chr():
    fd = sys.stdin.fileno()
    old_settings = termios.tcgetattr(fd)
    try:
        tty.setraw(sys.stdin.fileno())
        ch = sys.stdin.read(1)
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
    return ch


def user_cmd(prompt: str, choices: str):
    choice = None
    while choice not in set(choices):
        choice = input(prompt).lower()
        # choice = get_chr().lower()
    return choice

