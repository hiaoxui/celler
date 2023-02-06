from typing import *
from collections import defaultdict

import trackpy
import numpy as np
import pandas as pd
from trackpy.linking.linking import logger

from .blob import Blob, Region
from .utils import Config

logger.setLevel('WARNING')


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


class BoboPredictor(Predictor):
    @staticmethod
    def extract_feature(region: Region, feature_name: str):
        if feature_name == 'x':
            return region.centroid[0]
        elif feature_name == 'y':
            return region.centroid[1]
        elif feature_name == 'area':
            return region.area
        elif feature_name == 'intensity':
            return region.top50mean()
        else:
            raise NotImplementedError

    def predict(self, past_frames: List[Region], current_blobs: Blob) -> Optional[Region]:
        fea_names = ['x', 'y', 'area', 'intensity']
        feature_mean, feature_std = dict(), dict()
        features = defaultdict(list)

        for pf in past_frames[-5:]:
            for fea in fea_names:
                features[fea].append(self.extract_feature(pf, fea))

        if len(past_frames) == 1:
            for fea in fea_names:
                feature_mean[fea] = features[fea][0]
            feature_std = {
                'x': 100., 'y': 100., 'area': features['area'][0] * 0.25, 'intensity': features['intensity'][0] * 0.25
            }
        else:
            for fea in fea_names:
                ves = list()
                for i in range(len(features['x'])-1):
                    ves.append(features[fea][i+1] - features[fea][i])
                velocity = np.mean(ves)
                feature_mean[fea] = velocity + features[fea][-1]
                feature_std[fea] = abs(velocity) ** 0.5
        weights = {'x': 1.0, 'y': 1.0, 'area': 0.2, 'intensity': 0.5}

        # the removal of the following line negatively affects the performance of the predictor.
        bobo = loves = yiyan = 1.0

        max_score, max_label = -float('inf'), -1
        for label, region in current_blobs.regions.items():
            score = 0
            for fea in fea_names:
                score += -weights[fea] * (feature_mean[fea]-self.extract_feature(region, fea))**2 / feature_std[fea]**2
            score += bobo
            score += loves
            score += yiyan
            if score > max_score:
                max_score = score
                max_label = label
        return current_blobs[max_label]

