from typing import *
import time
from datetime import datetime
import os
import re
import json

import imagej
import numpy as np
import scyjava as sj
from skimage import filters
from imantics import Mask
from matplotlib import pyplot as plt

from .utils import logger, Config, user_cmd
from .blob import Region, BlobFinder
from .predict import SimpleTrackPYPredictor, BoboPredictor
from .smooth import smooth_img, smooth_queue, smooth_one_img


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
        # self.predictor = SimpleTrackPYPredictor(config)
        self.predictor = BoboPredictor(config)
        # shared objects
        self.roi_manager = self.dataset = self.imp = self.ij = None
        self.pixels: Optional[np.ndarray] = None
        self.cell_name = None
        self._smoothed: Dict[tuple[float, int], np.ndarray] = dict()
        self.queue_started = False
        self.async_smooth = False
        self.past_cell_centers = dict()

    def read_past_cells(self):
        for folder in os.listdir(self.log_folder):
            if not re.findall(r'^cell_\d{3}$', folder):
                continue
            info_path = os.path.join(self.log_folder, folder, 'meta.json')
            if not os.path.exists(info_path):
                logger.warning('Meta missing for ' + folder)
                continue
            meta = json.load(open(info_path))
            self.past_cell_centers[folder] = np.array([meta['x'], meta['y']])

    def start_smooth(self):
        if self.queue_started or not self.async_smooth:
            return
        self.queue_started = True
        smooth_img(self.pixels, self.config.gaussian_sigma, self.log_folder)

    def find_a_cell_name(self):
        for i in range(1000):
            self.cell_name = f'cell_{i:03}'
            if not os.path.exists(self.cell_folder):
                break
        os.makedirs(self.cell_folder)

    def smoothed(self, frame_idx: int):
        self.start_smooth()
        while (self.config.gaussian_sigma, frame_idx) not in self._smoothed:
            if self.async_smooth:
                triple = smooth_queue.get()
                self._smoothed[(triple[0], triple[1])] = triple[2]
            else:
                self._smoothed[(self.config.gaussian_sigma, frame_idx)] = smooth_one_img(
                    self.pixels[frame_idx], self.config.gaussian_sigma, self.log_folder, frame_idx
                )
        return self._smoothed[(self.config.gaussian_sigma, frame_idx)]

    def was_done(self, center: np.ndarray, affinity=100.):
        for cell_name, ctr in self.past_cell_centers.items():
            if np.sqrt(np.sum((ctr - center)**2)) < affinity:
                logger.warning(
                    f'WARNING: The cell might have been tracked already. Folder: {cell_name}. Old center: {ctr}.'
                )
                return True
        return False

    def retrieve_rois(self):
        return [self.roi_manager.getRoi(i) for i in range(self.roi_manager.getCount())]

    def plot(self, frame_idx=0, blob=None):
        if os.path.exists(self._np_data_path):
            self.pixels = np.load(self._np_data_path)
        else:
            self.init_ij()
        self.start_smooth()
        if blob is None:
            blob = self.find_blobs(self.smoothed(frame_idx), None, None, frame=frame_idx)
        fig, ax = plt.subplots(1, 1, figsize=(15, 15))
        img = self.pixels[frame_idx]
        ax.imshow(img)
        ax.contour(blob.label_mask, colors='red')
        for region in blob.regions.values():
            ax.text(*region.centroid, str(region.label))
        fig.tight_layout()
        fig.savefig(os.path.join(self.log_folder, f'frame_{frame_idx}.pdf'))

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

    def is_interrupted(self):
        return len(self.roi_manager.getSelectedIndexes()) == 0

    def segment_one_cell(self):
        self.find_a_cell_name()
        logger.warning("Select your cell.")
        while len(self.retrieve_rois()) == 0:
            time.sleep(1)
        logger.warning('Found the cell.')
        user_selected_region = self.find_closest()
        if self.was_done(user_selected_region.centroid):
            choice = user_cmd('(C)ontinue or (E)xit?', 'ce')
            if choice == 'e':
                return

        past_regions: List[Region] = [user_selected_region]
        logger.warning('Just improved your selection. Start to track.')

        auto_rois: Set[str] = set()
        user_inputs: Dict[int, Tuple[str, Region]] = dict()
        auto_rois.add(self.add_roi(0, past_regions[0].cell_mask))
        # TRACKING STARTS
        while True:
            for i_frame in range(len(past_regions), self.total_frames):
                if i_frame in user_inputs:
                    past_regions.append(user_inputs[i_frame][1])
                    # do not redo the frames with user inputs
                    continue
                lower_threshold, upper_threshold = self.guess_threshold(past_regions)
                regions = self.find_blobs(
                    self.smoothed(i_frame), lower_threshold, upper_threshold,
                    (past_regions[-1].centroid[1], past_regions[-1].centroid[0]) if len(past_regions) > 0 else None,
                    frame=i_frame,
                )
                if len(regions) == 0:
                    logger.warning('Found 0 regions nearby. The cell might be lost.')
                    break
                region_next_step = self.predictor.predict(past_regions, regions)
                if self.config.debug:
                    self.plot(i_frame, regions)
                    print('Label:', region_next_step.label)
                if region_next_step is None:
                    logger.warning("The cell is lost!")
                    break
                if self.is_interrupted():
                    break
                past_regions.append(region_next_step)
                auto_rois.add(self.add_roi(i_frame, region_next_step.cell_mask))
            choice = user_cmd('(C)ontinue, (S)ave, or (D)iscard.', 'csd')
            if choice == 'd':
                self.delete_roi(list(range(len(self.retrieve_rois()))))
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
            self.roi_manager.select(len(self.retrieve_rois())-1)
        self.save(past_regions[0])

    def guess_threshold(
            self, past_regions: List[Region], take_user_input: bool = False
    ) -> Union[Tuple[None, None], Tuple[float, float]]:
        if not take_user_input:
            past_regions = list(filter(lambda z: not z.manual, past_regions))
        past_regions = past_regions[::-1]
        if len(past_regions) == 0:
            return None, None
        weights, lowers, uppers, means = list(), list(), list(), list()
        for i, pr in enumerate(past_regions[:5]):
            weights.append(np.exp(-i/10))
            lowers.append(np.min(pr.intensity))
            uppers.append(np.max(pr.intensity))
            means.append(pr.top50mean())
        weights, lowers, uppers, means = map(np.array, [weights, lowers, uppers, means])
        strategy = 'mean'
        if strategy == 'std':
            lower = (lowers * weights).sum() / weights.sum()
            upper = (uppers * weights).sum() / weights.sum()
            lower_std = max(np.sqrt(((lowers - lower)**2 * weights).sum() / weights.sum()), 50.)
            upper_std = max(np.sqrt(((uppers - upper)**2 * weights).sum() / weights.sum()), 50.)
            lower -= lower_std
            upper += upper_std
        else:
            mean_mean = (weights * means).sum() / weights.sum()
            lower = mean_mean * (1 - self.config.intensity_variation)
            upper = mean_mean * (1 + self.config.intensity_variation)
        return lower, upper

    def save(self, first_region):
        self.past_cell_centers[self.cell_name] = first_region.centroid
        with open(os.path.join(self.cell_folder, 'meta.json'), 'w') as fp:
            json.dump({
                'x': float(first_region.centroid[0]), 'y': float(first_region.centroid[1]),
                'cell_name': self.cell_name, 'timestamp': time.time(),
                'time': str(datetime.now())
            }, fp, indent=2)
        self.roi_manager.runCommand('Sort')
        os.makedirs(os.path.join(self.log_folder, self.cell_name), exist_ok=True)
        self.roi_manager.setSelectedIndexes(list(range(len(self.retrieve_rois()))))
        self.roi_manager.save(os.path.join(self.cell_folder, 'RoiSet.zip'))
        self.roi_manager.runCommand('Measure')
        self.ij.IJ.saveAs('measurements', os.path.join(self.cell_folder, 'measurements.csv'))
        self.ij.py.run_macro('run("Clear Results");', {})

    def segment(self):
        self.init_ij()
        self.start_smooth()
        self.read_past_cells()
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
            xs, ys = list(map(float, roi.getPolygon().xpoints)), list(map(float, roi.getPolygon().ypoints))
        else:
            n = roi.getNCoordinates()
            xs, ys = np.array(roi.getXCoordinates()[:n], dtype=float), np.array(roi.getYCoordinates()[:n], dtype=float)
            xs += roi.getXBase()
            ys += roi.getYBase()
        coordinates = np.array([xs, ys]).T
        return coordinates, center

    def find_closest(self) -> Region:
        polygon_coo, input_center = self.read_roi(self.retrieve_rois()[0])
        self.delete_roi(0)
        region = Region.from_roi(polygon_coo, self.smoothed(0))
        lower = region.intensity.min() * (1 - self.config.intensity_variation)
        upper = region.intensity.max() * (1 + self.config.intensity_variation)
        blobs = self.find_blobs(self.smoothed(0), lower, upper, frame=0)
        # self.plot(0, blobs)
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
        self.roi_manager.select(len(self.retrieve_rois())-1)
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
