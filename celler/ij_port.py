from typing import *
import time
from datetime import datetime
import os
import re
import json

import imagej
import numpy as np
import scyjava as sj
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
        # self.predictor = BoboPredictor(config)
        self.predictor = SimpleTrackPYPredictor(config)
        # shared objects
        self.roi_manager = self.dataset = self.imp = self.ij = None
        self.pixels: Optional[np.ndarray] = None
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
            blob = self.find_blobs(self.smoothed(frame_idx), None, None, frame=frame_idx, erosion=self.config.erosion)
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
            self, user_inputs: Dict[int, Tuple[str, Region]], auto_rois: Set[str], past_regions: List[Region]
    ) -> bool:
        to_add, to_delete = list(), list()
        delete_after = 99999
        for roi_idx, roi in enumerate(self.retrieve_rois()):
            if roi.getName() in auto_rois:
                continue
            frame = int(roi.getName()[:4]) - 1
            if frame in user_inputs and roi.getName() == user_inputs[frame][0]:
                # It has been recorded
                continue
            # New input
            while len(past_regions) > frame:
                past_regions.pop(-1)
            delete_after = min(delete_after, frame)
            is_refined, region = self.parse_user_input(roi, frame)
            if is_refined:
                past_regions.append(region)
                to_delete.append(roi_idx)
                to_add.append([frame, region.cell_mask])
            else:
                user_inputs[frame] = (roi.getName(), region)

        if delete_after == 99999:
            return False

        self.delete_roi(to_delete)
        new_auto = list()
        for frame, cm in to_add:
            new_auto.append(self.add_roi(frame, cm))
        self.delete_auto(auto_rois, delete_after)
        auto_rois |= set(new_auto)
        return True

    def is_interrupted(self):
        return len(self.roi_manager.getSelectedIndexes()) == 0

    def segment_one_cell(self):
        logger.warning("Select your cell.")
        while len(self.retrieve_rois()) == 0:
            time.sleep(1)
        logger.warning('Found the cell.')
        is_refined, user_selected_region = self.parse_user_input(self.retrieve_rois()[0], 0)
        if is_refined:
            self.delete_roi(0)

        if self.was_done(user_selected_region.centroid):
            choice = user_cmd('(C)ontinue or (E)xit?', 'ce')
            if choice == 'e':
                return

        past_regions: List[Region] = [user_selected_region]
        logger.warning('Just improved your selection. Start to track.')

        auto_rois: Set[str] = set()
        user_inputs: Dict[int, Tuple[str, Region]] = dict()
        auto_rois.add(self.add_roi(0, user_selected_region.cell_mask))
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
            input_exist = self.check_user_input(user_inputs, auto_rois, past_regions)
            if not input_exist:
                # if user has no input, set the pointer to the next step
                if len(past_regions) == self.total_frames:
                    break
                continue
            # delete the frames after the pointer but keep user's inputs
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
            means.append(pr.top_mean())
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
            lower, upper = mean_mean * np.array([self.config.lower_intensity, self.config.upper_intensity])
        return lower, upper

    def save(self, first_region):
        cell_name = cell_folder = ''
        for i in range(1000):
            cell_name = f'cell_{i:03}'
            cell_folder = os.path.join(self.log_folder, cell_name)
            if not os.path.exists(cell_folder):
                break
        os.makedirs(os.path.join(self.log_folder, cell_name), exist_ok=True)

        self.past_cell_centers[cell_name] = first_region.centroid
        with open(os.path.join(cell_folder, 'meta.json'), 'w') as fp:
            json.dump({
                'x': float(first_region.centroid[0]), 'y': float(first_region.centroid[1]),
                'cell_name': cell_name, 'timestamp': time.time(),
                'time': str(datetime.now())
            }, fp, indent=2)
        self.roi_manager.runCommand('Sort')
        self.roi_manager.setSelectedIndexes(list(range(len(self.retrieve_rois()))))
        self.roi_manager.save(os.path.join(cell_folder, 'RoiSet.zip'))
        self.roi_manager.runCommand('Measure')
        self.ij.IJ.saveAs('measurements', os.path.join(cell_folder, 'measurements.csv'))
        self.ij.py.run_macro('run("Clear Results");', {})

    def segment(self):
        self.init_ij()
        self.start_smooth()
        self.read_past_cells()
        while True:
            # Clean past traces
            self.delete_roi(list(range(len(self.retrieve_rois()))))
            # do a new one
            self.segment_one_cell()
            logger.warning(f'Done with current cell.')
            time.sleep(0.1)
            choice = user_cmd('(C)ontinue or (E)xit.', 'ce')
            if choice.lower() == 'e':
                break

    @staticmethod
    def read_roi(roi) -> Tuple[np.ndarray, np.ndarray, str]:
        center = np.array([roi.getBounds().getCenterX(), roi.getBounds().getCenterY()])
        if roi.getTypeAsString() != 'Polygon':
            xs, ys = list(map(float, roi.getPolygon().xpoints)), list(map(float, roi.getPolygon().ypoints))
        else:
            n = roi.getNCoordinates()
            xs, ys = np.array(roi.getXCoordinates()[:n], dtype=float), np.array(roi.getYCoordinates()[:n], dtype=float)
            xs += roi.getXBase()
            ys += roi.getYBase()
        coordinates = np.array([xs, ys]).T
        return coordinates, center, roi.getTypeAsString()

    def parse_user_input(self, roi, frame_idx) -> Tuple[bool, Region]:
        """
        :param roi: ROI of user input
        :param frame_idx: Frame index
        :return: Return 2 items. The first is a bool flag, set True if the input is refined. The second is the region.
        """
        polygon_coo, input_center, roi_type = self.read_roi(roi)
        region = Region.from_roi(polygon_coo, self.smoothed(frame_idx))
        if roi_type != 'Polygon':
            # self.delete_roi(roi_index)
            lower, upper = region.top_mean() * self.config.lower_intensity, region.top_mean() * self.config.upper_intensity
            blobs = self.find_blobs(self.smoothed(frame_idx), lower, upper, around=region.centroid, frame=frame_idx)
            assert len(blobs) > 0, "Cannot find any regions around user input."
            # self.plot(0, blobs)
            closest_region = sorted(
                list(blobs.regions.values()), key=lambda bb: np.sum((bb.centroid - input_center) ** 2)
            )[0]
            return True, closest_region
        else:
            return False, region

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

    def roi_by_frame(self, frame: int):
        for idx, roi in enumerate(self.retrieve_rois()):
            if int(roi.getName()[:4]) - 1 == frame:
                yield idx

    def delete_auto(self, auto_rois: Set[str], after=None):
        to_delete = list()
        for idx, roi in enumerate(self.retrieve_rois()):
            roi_name = roi.getName()
            frame = int(roi_name[:4]) - 1
            if after is not None and frame < after:
                continue
            if roi_name in auto_rois:
                to_delete.append(idx)
                auto_rois.remove(roi_name)
        self.delete_roi(to_delete)
