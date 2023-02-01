from typing import *
import imagej
import os
import numpy as np
import jpype
import scyjava as sj
from skimage import io, filters, morphology, measure, draw, exposure
from skimage.filters import threshold_otsu
import imageio
from scipy.stats import ttest_ind
import seaborn as sns
from scipy import stats
import scipy
from scipy.stats import median_abs_deviation
from scipy.optimize import curve_fit
from sklearn.neighbors import KernelDensity
from matplotlib import pyplot as plt
from imantics import Polygons, Mask

from .utils import logger, Config


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
        self._np_data_path = os.path.join(self.log_folder, 'pixels.npy')
        self.image_file_path = image_file_path
        os.makedirs(self.log_folder, exist_ok=True)
        self.roi_manager = self.dataset = self.imp = self.ij = None
        self.pixels: Optional[np.ndarray] = None
        self.rois = list()

    def retrieve_rois(self):
        self.rois = [self.roi_manager.getRoi(i) for i in range(self.roi_manager.getCount())]

    def find_blobs(self, frame: int):
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
        return smooth, label_mask_clean, region_properties

    @staticmethod
    def mask2polygon(mask):
        return Mask(mask).polygons().points

    def _plot(self, smooth, label_mask_clean, region_properties, output_name):
        fig, ax = plt.subplots(1, 1, figsize=(20, 20))
        ax.imshow(smooth)
        ax.contour(label_mask_clean, colors='red')
        for idx, rp in enumerate(region_properties):
            ax.text(rp.centroid[1], rp.centroid[0], str(idx))
        fig.tight_layout()
        fig.savefig(os.path.join(self.log_folder, f'{output_name}.pdf'))

    def plot(self):
        if os.path.exists(self._np_data_path):
            self.pixels = np.load(self._np_data_path)
        else:
            self.init_ij()
        smooth, label_mask_clean, region_properties = self.find_blobs(0)
        self._plot(smooth, label_mask_clean, region_properties, 'cells')

    def segment(self):
        self.init_ij()
        logger.warning("Select your cell.")
        input('Press Enter to proceed.')
        self.retrieve_rois()
        self.find_closest(self.rois[-1], 0)
        input('Please check the revision.')

    @staticmethod
    def read_roi(roi):
        n = roi.getNCoordinates()
        xs, ys = np.array(roi.getXCoordinates()[:n], dtype=float), np.array(roi.getYCoordinates()[:n], dtype=float)
        xs += roi.getXBase()
        ys += roi.getYBase()
        coordinates = np.array([xs, ys]).T
        center = np.array([roi.getBounds().getCenterX(), roi.getBounds().getCenterY()])
        return coordinates, center

    def find_closest(self, input_roi, frame: int):
        input_coordinates, input_center = self.read_roi(input_roi)

        distances = list()
        smooth, label_mask_clean, region_properties = self.find_blobs(frame)
        for rp in region_properties:
            distances.append(np.sqrt(np.sum((np.array([rp.centroid[1], rp.centroid[0]]) - input_center)**2)))
        cell_idx = np.argmin(distances)

        polygon = self.mask2polygon(label_mask_clean == region_properties[cell_idx].label)[0].astype(float)
        polygon_class = sj.jimport('ij.gui.PolygonRoi')
        roi = polygon_class(polygon[:, 0].tolist(), polygon[:, 1].tolist(), polygon.shape[0], 2)
        # overlay_class = sj.jimport('ij.gui.Overlay')
        # ov = overlay_class()
        # ov.add(roi)
        self.roi_manager.addRoi(roi)
        self.delete_roi(0)

    def delete_roi(self, index: int):
        self.retrieve_rois()
        self.roi_manager.select(index)
        self.roi_manager.runCommand('Delete')
        self.retrieve_rois()
