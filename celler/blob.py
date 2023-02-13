from typing import *
from dataclasses import dataclass

import numpy as np
from skimage import filters, morphology, measure
from skimage.filters import threshold_otsu
from imantics import Polygons, Mask

from .utils import Config


@dataclass()
class Region:
    label: int
    centroid: np.ndarray
    area: int
    eccentricity: float
    coords: Optional[np.ndarray] = None
    bbox: Optional[np.ndarray] = None
    extent: Optional[np.ndarray] = None
    intensity: Optional[np.ndarray] = None
    cell_mask: Optional[np.ndarray] = None
    manual: bool = False

    def top50mean(self):
        return float(self.intensity[self.intensity > np.median(self.intensity)].mean())

    @classmethod
    def from_roi(cls, coordinates: np.ndarray, image_smoothed):
        coordinates = coordinates.astype(np.int16)
        # polygon_mask = Polygons(coordinates.T.tolist()).mask(*image_smoothed.shape).array.astype(np.int16)
        polygon_mask = Polygons(coordinates.tolist()).mask(
            image_smoothed.shape[1], image_smoothed.shape[0]
        ).array.astype(np.int16)
        rp = measure.regionprops(polygon_mask)
        assert len(rp) == 1
        region = Region.from_rp(rp[0], polygon_mask, None, image_smoothed)
        region.manual = True
        return region

    @classmethod
    def from_rp(cls, rp, label_mask, hole_mask, image):
        region = Region(
            rp.label, np.array([rp.centroid[1], rp.centroid[0]]), rp.area, rp.eccentricity, rp.coords,
            rp.bbox, rp.extent
        )
        region.cell_mask = (label_mask == region.label)
        mask = region.cell_mask
        if hole_mask is not None:
            mask = mask & ~hole_mask
        region.intensity = image[mask]
        return region


class Blob:
    def __init__(self, label_mask, image, hole_mask, frame=None):
        self.label_mask = label_mask
        self.frame: Optional[int] = frame
        self.regions = {
            rp.label: Region.from_rp(rp, label_mask, hole_mask, image)
            for rp in measure.regionprops(label_mask)
        }

    def __getitem__(self, item: int) -> Region:
        return self.regions[item]

    def __len__(self):
        return len(self.regions)


class BlobFinder:
    def __init__(self, config):
        self.config = config

    def __call__(
            self, img: np.ndarray, lower: Optional[float], upper: Optional[float],
            around: Optional[Tuple[int, int]] = None, frame: int = None
    ):
        otsu_threshold = threshold_otsu(img) + self.config.threshold_adjustment * img.std()
        if lower is None:
            lower = otsu_threshold
        else:
            lower = max(otsu_threshold, lower)
        if upper is None:
            upper = float('inf')
        # TODO upper bound is problematic because of the ring problem. consider improving it
        #upper = float('inf')
        cell_mask = (img < upper) & (img > lower)
        if around is not None:
            affinity_mask = np.ones(img.shape, bool)
            for axis in [0, 1]:
                ar = np.arange(img.shape[axis])
                axis_mask = (ar < around[axis] + self.config.search_range) & (ar > around[axis] - self.config.search_range)
                affinity_mask &= np.expand_dims(axis_mask, 1-axis)
            cell_mask &= affinity_mask
        cell_mask_remove_small = morphology.remove_small_objects(cell_mask, self.config.min_size)
        # TODO remove ring objects
        cell_mask_remove_hole = morphology.remove_small_holes(cell_mask_remove_small.copy(), self.config.max_size)
        hole_mask = cell_mask_remove_hole & ~cell_mask_remove_small
        label_mask_tmp = measure.label(cell_mask_remove_hole)
        region_properties = measure.regionprops(label_mask_tmp)
        cell_mask_remove_big = cell_mask_remove_hole
        for r in region_properties:
            if r.area > self.config.max_size:
                cell_mask_remove_big[label_mask_tmp == r.label] = 0
        cell_mask = cell_mask_remove_big
        label_mask = measure.label(cell_mask)
        return Blob(label_mask, img, hole_mask, frame)

