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
    offsets: List[int]
    canvas: Tuple[int, int]
    crop_cell_mask: Optional[np.ndarray] = None
    intensity: Optional[np.ndarray] = None
    manual: bool = False

    def top_mean(self):
        return float(self.intensity[self.intensity > np.quantile(self.intensity, 0.6)].mean())

    def erosion(self, radius, max_hole=None, small_size=None):
        if radius == 0:
            return
        pad_widths = []
        for axis in range(2):
            left = min(self.offsets[axis], radius)
            right = min(self.canvas[axis] - self.offsets[axis] - self.crop_cell_mask.shape[axis], radius)
            pad_widths.append([left, right])
        self.offsets = [self.offsets[axis] - pad_widths[axis][0] for axis in range(2)]
        reverse_crop = ~np.pad(self.crop_cell_mask, pad_widths)
        for _ in range(radius):
            reverse_crop = morphology.erosion(reverse_crop)
        crop = ~reverse_crop
        if max_hole is not None:
            crop = morphology.remove_small_holes(crop, max_hole)
        if small_size is not None:
            crop = morphology.remove_small_objects(crop, small_size)
        self.crop_cell_mask = crop

    @staticmethod
    def crop_image(cell_mask):
        def get_limits(axis):
            ar = np.arange(cell_mask.shape[axis])[cell_mask.any(1 - axis)]
            return max(0, ar.min()-1), min(ar.max()+1, cell_mask.shape[axis]-1)

        lim0, lim1 = get_limits(0), get_limits(1)
        offsets = [lim0[0], lim1[0]]
        crop = cell_mask[lim0[0]:lim0[1] + 1, lim1[0]:lim1[1] + 1]
        return crop, offsets

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
        cell_mask = (label_mask == rp.label)
        crop, offsets = Region.crop_image(cell_mask)
        region = Region(
            rp.label, np.array([rp.centroid[1], rp.centroid[0]]), rp.area, rp.eccentricity,
            offsets, tuple(cell_mask.shape), crop
        )
        if hole_mask is not None:
            cell_mask = cell_mask & ~hole_mask
        region.intensity = image[cell_mask]
        return region

    @property
    def cell_mask(self):
        cm = np.zeros(self.canvas, dtype=bool)
        slices = [slice(self.offsets[axis], self.offsets[axis] + self.crop_cell_mask.shape[axis]) for axis in [0, 1]]
        cm[slices[0], slices[1]] = self.crop_cell_mask
        return cm


class Blob:
    def __init__(self, label_mask, image, hole_mask, frame=None, erosion: int = 0, cfg: Config = None):
        self.label_mask = label_mask
        self.frame: Optional[int] = frame
        self.regions: Dict[int, Region] = {
            rp.label: Region.from_rp(rp, label_mask, hole_mask, image)
            for rp in measure.regionprops(label_mask)
        }
        to_remove = list()
        for r in self.regions.values():
            if cfg is not None:
                r.erosion(erosion, cfg.max_hole, cfg.min_size)
            else:
                r.erosion(erosion)
            if not r.crop_cell_mask.any():
                to_remove.append(r.label)
            else:
                self.label_mask[r.cell_mask] = r.label
        for tr in to_remove:
            self.regions.pop(tr)

    def __getitem__(self, item: int) -> Region:
        return self.regions[item]

    def __len__(self):
        return len(self.regions)

    def __iter__(self):
        yield from self.regions.values()


class BlobFinder:
    def __init__(self, config):
        self.config = config

    def gen_mask(self, img, lower, upper, around, addition_mask=None):
        otsu_threshold = threshold_otsu(img) + self.config.threshold_adjustment * img.std()
        if lower is None:
            lower = otsu_threshold
        else:
            lower = max(otsu_threshold, lower)
        if upper is None:
            upper = float('inf')
        cell_mask = (img < upper) & (img > lower)
        if addition_mask is not None:
            cell_mask &= addition_mask
        if around is not None:
            affinity_mask = np.ones(img.shape, bool)
            for axis in [0, 1]:
                ar = np.arange(img.shape[axis])
                axis_mask = (ar < around[axis] + self.config.search_range) & (
                            ar > around[axis] - self.config.search_range)
                affinity_mask &= np.expand_dims(axis_mask, 1 - axis)
            cell_mask &= affinity_mask
        cell_mask_remove_small = morphology.remove_small_objects(cell_mask, self.config.min_size)
        cell_mask_remove_hole = morphology.remove_small_holes(cell_mask_remove_small.copy(), self.config.max_hole)
        hole_mask = cell_mask_remove_hole & ~cell_mask_remove_small

        if self.config.max_size is not None:
            label_mask_tmp = measure.label(cell_mask_remove_hole)
            region_properties = measure.regionprops(label_mask_tmp)
            cell_mask_remove_big = cell_mask_remove_hole
            for r in region_properties:
                if r.area > self.config.max_size:
                    cell_mask_remove_big[label_mask_tmp == r.label] = 0
            cell_mask = cell_mask_remove_big
        else:
            cell_mask = cell_mask_remove_hole
        return cell_mask, hole_mask

    def __call__(
            self, img: np.ndarray, lower: Optional[float], upper: Optional[float],
            around: Optional[Tuple[int, int]] = None, frame: int = None,
            erosion: int = 0
    ) -> Blob:
        upper_add = None
        if upper is not None:
            upper_regions = self(img, upper, None, around, erosion=10)
            upper_add = ~sum([ur.cell_mask for ur in list(upper_regions)], np.zeros(img.shape, bool))
        cell_mask, hole_mask = self.gen_mask(img, lower, upper, around, upper_add)
        label_mask = measure.label(cell_mask)
        regions = Blob(label_mask, img, hole_mask, frame, erosion, self.config)
        return regions
