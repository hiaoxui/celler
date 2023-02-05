from typing import *

import trackpy
import pandas as pd

from .blob import Blob, Region
from .utils import Config


class Predictor:
    def __init__(self, config):
        self.config: Config = config

    def predict(self, past_frames: List[Region], current_blobs: Blob) -> Optional[Region]:
        raise NotImplementedError


class SimpleTrackPYPredictor(Predictor):
    def predict(self, past_frames: List[Region], current_blobs: Blob) -> Optional[Region]:
        rows = list()
        for i, region in enumerate(past_frames):
            rows.append({'x': region.centroid[0], 'y': region.centroid[1], 'label': -1, 'frame': i})
        for label, region in current_blobs.regions.items():
            rows.append({'x': region.centroid[0], 'y': region.centroid[1], 'label': label, 'frame': len(past_frames)})
        tracked = trackpy.link(pd.DataFrame(rows), self.config.search_range, memory=10)
        particle_num = int(tracked[tracked.label == -1].particle[0])
        fetched = tracked[(tracked.particle == particle_num) & (tracked.label != -1)]
        if len(fetched) == 0:
            return None
        assert len(fetched) == 1
        cell_label = int(fetched.label)
        return current_blobs[cell_label]
