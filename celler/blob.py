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
    def __init__(self, label_mask, image, hole_mask):
        self.label_mask = label_mask
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

    def __call__(self, img: np.ndarray, lower: Optional[float], upper: Optional[float]):
        smooth = filters.gaussian(img, self.config.gaussian_sigma, preserve_range=True)
        smoothed_std = smooth.std()
        otsu_threshold = threshold_otsu(smooth)
        if lower is not None and upper is not None:
            lower = otsu_threshold + lower * smoothed_std
            upper = otsu_threshold + upper * smoothed_std
        else:
            lower = otsu_threshold + self.config.threshold_adjustment * np.std(img)
            upper = float('inf')
        cell_mask = (smooth < upper) & (smooth > lower)
        cell_mask_remove_small = morphology.remove_small_objects(cell_mask, self.config.min_size)
        cell_mask_remove_hole = morphology.remove_small_holes(cell_mask_remove_small.copy(), self.config.max_hole)
        hole_mask = cell_mask_remove_hole & ~cell_mask_remove_small
        label_mask_tmp = measure.label(cell_mask_remove_hole)
        region_properties = measure.regionprops(label_mask_tmp)
        cell_mask_remove_big = cell_mask_remove_hole
        for r in region_properties:
            if r.area > self.config.max_size:
                cell_mask_remove_big[label_mask_tmp == r.label] = 0
        cell_mask = cell_mask_remove_big
        label_mask = measure.label(cell_mask)
        return Blob(label_mask, smooth, hole_mask)
