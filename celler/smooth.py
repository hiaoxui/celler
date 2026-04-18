import logging
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import Queue
from pathlib import Path

import numpy as np
from skimage import filters

logger = logging.getLogger('cell')

smooth_queue = Queue()
n_process = 1
worker = ProcessPoolExecutor(n_process, mp.get_context('spawn'))


def _smooth_one_img(pixels: np.ndarray, sigma: float, log_folder: str | Path, frame_idx: int):
    smooth_queue.put((sigma, frame_idx, smooth_one_img(pixels, sigma, log_folder, frame_idx)))


def smooth_one_img(pixels: np.ndarray, sigma: float, log_folder: str | Path, frame_idx: int):
    logger.debug(f"Smoothing frame {frame_idx}...")
    cache_path = Path(log_folder) / 'cache' / f'smooth_{sigma}' / f'{frame_idx}.npy'
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    if not cache_path.exists():
        smoothed = filters.gaussian(pixels, sigma, preserve_range=True).astype(np.uint16)
        np.save(cache_path, smoothed)
    else:
        smoothed = np.load(cache_path)
        if smoothed.dtype != np.uint16:
            smoothed = smoothed.astype(np.unit16)
    return smoothed


def smooth_img(all_frames: np.ndarray, sigma: float, log_folder: str | Path):
    logger.warning("Submitting jobs")
    for i_frame, pixels in enumerate(all_frames):
        worker.submit(_smooth_one_img, pixels.copy(), sigma, log_folder, i_frame)
