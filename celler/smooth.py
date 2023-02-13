import multiprocessing as mp
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import Queue
from skimage import filters

import numpy as np

from .utils import logger

smooth_queue = Queue()
n_process = 1
worker = ProcessPoolExecutor(n_process, mp.get_context('spawn'))


def _smooth_one_img(pixels: np.ndarray, sigma: float, log_folder: str, frame_idx: int):
    smooth_queue.put((sigma, frame_idx, smooth_one_img(pixels, sigma, log_folder, frame_idx)))


def smooth_one_img(pixels: np.ndarray, sigma: float, log_folder: str, frame_idx: int):
    logger.debug(f"Smoothing frame {frame_idx}...")
    cache_path = os.path.join(log_folder, 'cache', f'smooth_{sigma}', f'{frame_idx}.npy')
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    if not os.path.exists(cache_path):
        smoothed = filters.gaussian(pixels, sigma, preserve_range=True)
        np.save(cache_path, smoothed)
    else:
        smoothed = np.load(cache_path)
    return smoothed


def smooth_img(all_frames: np.ndarray, sigma: float, log_folder: str):
    logger.warning("Submitting jobs")
    for i_frame, pixels in enumerate(all_frames):
        worker.submit(_smooth_one_img, pixels.copy(), sigma, log_folder, i_frame)
