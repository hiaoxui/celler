from typing import *
import time
import pickle
import os

import imagej
import numpy as np
import scyjava as sj
from skimage import filters, morphology, measure
from skimage.filters import threshold_otsu
from imantics import Polygons, Mask
import trackpy
from matplotlib import pyplot as plt
import pandas as pd

from .utils import logger, Config, user_cmd
from .blob import Region, Blob, BlobFinder
from .predict import SimpleTrackPYPredictor


class IJPort:
    def init_ij(self):
        if self.ij is not None:
            return
        logger.info('Initializing ImageJ')
        sj.config.add_option('-Xmx8g')
        self.ij = ij = imagej.init('2.9.0', mode='interactive')
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
        sj.jimport('ij.plugin.filter.Analyzer').setMeasurements(12582911)
        logger.info(f'Done with loading.')

    def __init__(self, image_file_path: str, config: Config):
        self.config = config
        self.find_blobs = BlobFinder(config)
        # paths
        self.log_folder = image_file_path + '_logs'
        self._np_data_path = os.path.join(self.log_folder, 'cache', 'pixels.npy')
        self.image_file_path = image_file_path
        os.makedirs(os.path.join(self.log_folder, 'cache'), exist_ok=True)
        self.predictor = SimpleTrackPYPredictor(config)
        # shared objects
        self.roi_manager = self.dataset = self.imp = self.ij = None
        self.pixels: Optional[np.ndarray] = None
        self.cell_name = None

        # for current step

    def retrieve_rois(self):
        return [self.roi_manager.getRoi(i) for i in range(self.roi_manager.getCount())]

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
        smooth, label_mask_clean, regions = self.find_blobs(self.pixels[0], None, None)
        self._plot(smooth, label_mask_clean, regions, 'cells')

    @property
    def total_frames(self) -> int:
        return self.pixels.shape[0]

    def check_user_input(
            self, user_inputs: Dict[int, Tuple[str, Region]], auto_rois: Set[str]
    ) -> List[Tuple[int, Region]]:
        new_frames = list()
        for roi in self.retrieve_rois():
            if roi.getName() in auto_rois:
                continue
            frame = int(roi.getName()[:4]) - 1
            if frame in user_inputs and roi.getName() == user_inputs[frame][0]:
                # It has been recorded
                continue
            coo, center = self.read_roi(roi)
            region = Region.from_roi(
                coo, filters.gaussian(self.pixels[frame], self.config.gaussian_sigma, preserve_range=True)
            )
            user_inputs[frame] = (roi.getName(), region)
            new_frames.append((frame, region))
        return new_frames

    def segment_one_cell(self):
        logger.warning("Select your cell.")
        while len(self.retrieve_rois()) == 0:
            time.sleep(1)
        # logger.warning('Found the cell.')
        user_selected_region = self.find_closest(self.retrieve_rois()[-1])
        self.cell_name = f'cell_{user_selected_region.label:02}'
        if os.path.exists(self.cell_folder):
            logger.warning('Cell might already exist!')
        past_regions: List[Region] = [user_selected_region]
        logger.warning('Just improved your selection. Start to track.')

        auto_rois: Set[str] = set()
        user_inputs: Dict[int, Tuple[str, Region]] = dict()
        auto_rois.add(self.add_roi(0, past_regions[0].cell_mask))
        # TRACKING STARTS
        while True:
            n_step = min(self.total_frames - len(past_regions), self.config.frames_per_step)
            for i_frame in range(len(past_regions), len(past_regions) + n_step):
                # TODO adaptive upper and lower bound
                if i_frame in user_inputs:
                    past_regions.append(user_inputs[i_frame][1])
                    # do not redo the frames with user inputs
                    continue
                regions = self.find_blobs(self.pixels[i_frame], None, None)
                region_next_step = self.predictor.predict(past_regions, regions)
                if region_next_step is None:
                    raise NotImplementedError
                past_regions.append(region_next_step)
                auto_rois.add(self.add_roi(i_frame, region_next_step.cell_mask))
            choice = user_cmd('(C)ontinue, (S)ave, or (D)iscard.', 'scd')
            if choice == 'd':
                return
            elif choice == 's':
                break
            # to continue, check if user has some inputs
            new_inputs = self.check_user_input(user_inputs, auto_rois)
            if len(new_inputs) == 0:
                # if user has no input, set the pointer to the next step
                if len(past_regions) == self.total_frames:
                    break
                continue
            # user has new inputs. read some and set the pointer to the first user input
            # set the pointer to the step after the first user input
            past_regions = past_regions[:min(ni[0] for ni in new_inputs)]
            # delete the frames after the pointer but keep user's inputs
            self.delete_auto(auto_rois, len(past_regions))
            self.roi_manager.runCommand('Sort')
        self.save()

    def save(self):
        self.roi_manager.runCommand('Sort')
        os.makedirs(os.path.join(self.log_folder, self.cell_name), exist_ok=True)
        self.roi_manager.setSelectedIndexes(list(range(len(self.retrieve_rois()))))
        self.roi_manager.save(os.path.join(self.cell_folder, 'RoiSet.zip'))
        self.roi_manager.runCommand('Measure')
        self.ij.IJ.saveAs('measurements', os.path.join(self.cell_folder, 'measurements.csv'))
        self.ij.py.run_macro('run("Clear Results");', {})

    def segment(self):
        self.init_ij()
        while True:
            # Clean past traces
            self.delete_roi(list(range(len(self.retrieve_rois()))))
            self.retrieve_rois()
            self.cell_name = None
            # do a new one
            self.segment_one_cell()
            logger.warning(f'Done with {self.cell_name}.')
            time.sleep(0.1)
            choice = user_cmd('(C)ontinue or (E)xit.', 'ce')
            if choice.lower() == 'e':
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

    def find_closest(self, input_roi) -> Region:
        _, input_center = self.read_roi(input_roi)
        self.delete_roi(0)
        blobs = self.find_blobs(self.pixels[0], None, None)
        min_distance, min_label = float('inf'), -1
        for label in blobs.regions:
            distance = np.sqrt(np.sum((blobs[label].centroid - input_center) ** 2))
            if distance < min_distance:
                min_distance = distance
                min_label = label
        region = blobs[min_label]
        return region

    def add_roi(self, frame_idx, cell_mask) -> str:
        self.imp.setT(frame_idx + 1)
        polygon_class = sj.jimport('ij.gui.PolygonRoi')
        polygon = Mask(cell_mask).polygons().points[0].astype(float)
        roi = polygon_class(polygon[:, 0].tolist(), polygon[:, 1].tolist(), polygon.shape[0], 2)
        # overlay_class = sj.jimport('ij.gui.Overlay')
        # ov = overlay_class()
        # ov.add(roi)
        self.roi_manager.addRoi(roi)
        return roi.getName()

    def delete_roi(self, index: Union[int, List[int]]):
        if isinstance(index, int):
            self.delete_roi([index])
            return
        if len(index) > 0:
            self.roi_manager.deselect()
            time.sleep(0.2)
            self.roi_manager.setSelectedIndexes(index)
            time.sleep(0.2)
            self.roi_manager.runCommand('Delete')
            time.sleep(0.2)
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

    def delete_auto(self, auto_rois: Set[str], after=None):
        to_delete = list()
        for idx, roi in enumerate(self.retrieve_rois()):
            frame = int(roi.getName()[:4]) - 1
            if after is not None and frame < after:
                continue
            if roi.getName() in auto_rois:
                to_delete.append(idx)
        self.delete_roi(to_delete)
