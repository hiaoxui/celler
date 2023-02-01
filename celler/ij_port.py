from typing import *
import time
import pickle
import os

import imagej
import numpy as np
import scyjava as sj
from skimage import filters, morphology, measure
from skimage.filters import threshold_otsu
import trackpy
from matplotlib import pyplot as plt
from imantics import Polygons, Mask
import pandas as pd

from .utils import logger, Config
from .region import Region


class IJPort:
    def init_ij(self):
        if self.ij is not None:
            return
        logger.info('Initializing ImageJ')
        sj.config.add_option('-Xmx8g')
        self.ij = ij = imagej.init('sc.fiji:fiji', mode='interactive')
        ij.ui().showUI()
        logger.info(f'ImageJ version {ij.getVersion()}')

        logger.info(f'Loading {self.image_file_path}')
        self.dataset = dataset = ij.io().open(self.image_file_path)
        self.imp = imp = ij.py.to_imageplus(dataset)
        sp = sj.jimport('ij.plugin.ChannelSplitter')
        channels = sp.split(imp)
        if len(channels) > 1:
            logger.warning(f'Found {len(channels)} channels. Showing the first one.')
            self.imp = imp = channels[0]
        imp.getProcessor().resetMinAndMax()
        self.pixels = ij.py.from_java(imp).to_numpy()
        if not os.path.exists(self._np_data_path):
            np.save(self._np_data_path, self.pixels)
        ij.ui().show(self.imp)
        self.roi_manager = ij.RoiManager.getRoiManager()
        ij.RoiManager()
        logger.info(f'Done with loading.')

    def __init__(self, image_file_path: str, config: Config):
        self.config = config
        self.log_folder = image_file_path + '_logs'
        self._np_data_path = os.path.join(self.log_folder, 'cache', 'pixels.npy')
        self.image_file_path = image_file_path
        os.makedirs(os.path.join(self.log_folder, 'cache'), exist_ok=True)
        self.roi_manager = self.dataset = self.imp = self.ij = None
        self.pixels: Optional[np.ndarray] = None
        self.cell_name = None
        self.rois = list()
        self.blobs = dict()
        # All cells are final from this step on
        self.auto_rois = set()
        self.trusted_cell_info = dict()

    def retrieve_rois(self):
        self.rois = [self.roi_manager.getRoi(i) for i in range(self.roi_manager.getCount())]
        return self.rois

    def find_blobs(self, frame: int) -> Tuple[np.ndarray, np.ndarray, List[Region]]:
        if frame in self.blobs:
            return self.blobs[frame]
        cache_path = os.path.join(self.log_folder, 'cache', f'blob_{frame:03}.pkl')
        if os.path.exists(cache_path):
            smooth, label_mask_clean, regions = pickle.load(open(cache_path, 'rb'))
        else:
            img = self.pixels[frame]
            smooth = filters.gaussian(img, self.config.gaussian_sigma, preserve_range=True)
            smoothed_std = smooth.std()
            otsu_threshold = threshold_otsu(smooth) + self.config.threshold_adjustment * smoothed_std
            cell_mask = smooth > otsu_threshold
            cell_mask_clean = morphology.remove_small_objects(cell_mask, self.config.min_size)
            cell_mask_clean = morphology.remove_small_holes(cell_mask_clean, self.config.max_hole)
            label_mask = measure.label(cell_mask_clean)
            region_properties = measure.regionprops(label_mask)
            for r in region_properties:
                if r.area > self.config.max_size:
                    cell_mask_clean[label_mask == r.label] = 0
            label_mask_clean = measure.label(cell_mask_clean)
            region_properties = measure.regionprops(label_mask_clean)
            regions = [Region(rp) for rp in region_properties]
            pickle.dump((smooth, label_mask_clean, regions), open(cache_path, 'wb'))
        self.blobs[frame] = (smooth, label_mask_clean, regions)
        return smooth, label_mask_clean, regions

    def _plot(self, smooth, label_mask_clean, regions, output_name):
        fig, ax = plt.subplots(1, 1, figsize=(20, 20))
        ax.imshow(smooth)
        ax.contour(label_mask_clean, colors='red')
        for idx, rp in enumerate(regions):
            ax.text(*rp.centroid, str(idx))
        fig.tight_layout()
        fig.savefig(os.path.join(self.log_folder, f'{output_name}.pdf'))

    def plot(self):
        if os.path.exists(self._np_data_path):
            self.pixels = np.load(self._np_data_path)
        else:
            self.init_ij()
        smooth, label_mask_clean, regions = self.find_blobs(0)
        self._plot(smooth, label_mask_clean, regions, 'cells')

    def track(self, cell_idx_in_frame0: int, end_frame: Optional[int] = None):
        rows = list()
        end_frame = end_frame if end_frame is not None else self.pixels.shape[0]
        for fi in range(0, end_frame):
            if fi in self.trusted_cell_info:
                tci = self.trusted_cell_info[fi]
                rows.append({'x': tci['x'], 'y': tci['y'], 'frame': fi, 'cell': -1})
            else:
                smooth, label_mask_clean, regions = self.find_blobs(fi)
                for ci, rp in enumerate(regions):
                    rows.append({'x': rp.centroid[0], 'y': rp.centroid[1], 'frame': fi, 'cell': ci})
        tracked = trackpy.link(pd.DataFrame(rows), self.pixels.shape[1] * self.config.search_range, memory=self.config.track_memory)
        if 0 not in self.trusted_cell_info:
            particle_idx = tracked[(tracked.frame == 0) & (tracked.cell == cell_idx_in_frame0)].particle.array[0]
        else:
            frame0 = tracked[tracked.frame == 0]
            assert len(frame0) == 1
            particle_idx = frame0.particle.array[0]
        fetched = tracked[tracked.particle == particle_idx]
        for cell_in_frame in fetched.itertuples():
            cur_frame_idx, cur_cell_idx = cell_in_frame.frame, cell_in_frame.cell
            if cur_frame_idx in self.trusted_cell_info:
                continue
            smooth, label_mask_clean, regions = self.find_blobs(cur_frame_idx)
            cell_mask = regions[cur_cell_idx].label == label_mask_clean
            self.add_roi(cur_frame_idx, cell_mask)
        self.roi_manager.runCommand('Sort')

    def segment(self):
        self.init_ij()
        logger.warning("Select your cell.")
        while len(self.retrieve_rois()) == 0:
            time.sleep(1)
        logger.warning('Found the cell.')
        cell_idx = self.find_closest(self.rois[-1], 0)
        self.cell_name = f'cell_{cell_idx:02}'
        if os.path.exists(self.cell_folder):
            logger.warning('Cell might already exist!')

        logger.warning('Just improved your selection. Start to track.')
        self.track(cell_idx)
        while True:
            choice = ''
            while choice.lower() not in ['s', 'r', 'd']:
                choice = input('(S)ave, (R)edo, or (D)iscard.')
            if choice.lower() == 's':
                os.makedirs(os.path.join(self.log_folder, self.cell_name), exist_ok=True)
                self.roi_manager.setSelectedIndexes(list(range(len(self.retrieve_rois()))))
                self.roi_manager.save(os.path.join(self.cell_folder, 'RoiSet.zip'))
                break
            elif choice.lower() == 'r':
                self.delete_auto()
                self.read_user_input()
                self.track(cell_idx)
            else:
                break

    @staticmethod
    def read_roi(roi):
        center = np.array([roi.getBounds().getCenterX(), roi.getBounds().getCenterY()])
        if roi.getTypeAsString() != 'Polygon':
            return None, center
        n = roi.getNCoordinates()
        xs, ys = np.array(roi.getXCoordinates()[:n], dtype=float), np.array(roi.getYCoordinates()[:n], dtype=float)
        xs += roi.getXBase()
        ys += roi.getYBase()
        coordinates = np.array([xs, ys]).T
        return coordinates, center

    def find_closest(self, input_roi, frame: int):
        _, input_center = self.read_roi(input_roi)

        distances = list()
        smooth, label_mask_clean, regions = self.find_blobs(frame)
        for rp in regions:
            distances.append(np.sqrt(np.sum((rp.centroid - input_center)**2)))
        cell_idx = np.argmin(distances)
        self.delete_roi(0)
        return int(cell_idx)

    def add_roi(self, frame_idx, cell_mask):
        self.imp.setT(frame_idx+1)
        polygon_class = sj.jimport('ij.gui.PolygonRoi')
        polygon = Mask(cell_mask).polygons().points[0].astype(float)
        roi = polygon_class(polygon[:, 0].tolist(), polygon[:, 1].tolist(), polygon.shape[0], 2)
        # overlay_class = sj.jimport('ij.gui.Overlay')
        # ov = overlay_class()
        # ov.add(roi)
        self.roi_manager.addRoi(roi)
        self.auto_rois.add(roi.getName())

    def delete_roi(self, index: int):
        self.retrieve_rois()
        self.roi_manager.select(index)
        self.roi_manager.runCommand('Delete')
        self.retrieve_rois()

    def delete_roi_in_frame(self, frame: int):
        while True:
            for idx, roi in enumerate(self.retrieve_rois()):
                if roi.getTPosition() == frame:
                    self.delete_roi(idx)
                    continue
            break

    @property
    def cell_folder(self):
        return os.path.join(self.log_folder, self.cell_name)

    def read_user_input(self):
        for roi in self.retrieve_rois():
            name = roi.getName()
            frame_idx = int(name[:4]) - 1
            if name in self.auto_rois or frame_idx in self.trusted_cell_info:
                continue
            # This is a new input
            _, center = self.read_roi(roi)
            self.trusted_cell_info[frame_idx] = {'x': center[0], 'y': center[1]}

    def delete_auto(self):
        while True:
            deleted = False
            for idx, roi in enumerate(self.retrieve_rois()):
                if roi.getName() in self.auto_rois:
                    self.delete_roi(idx)
                    deleted = True
                    break
            if not deleted:
                break
