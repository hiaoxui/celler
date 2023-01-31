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
        ij.ui().show(self.imp)
        self.roi_manager = ij.RoiManager.getRoiManager()
        ij.RoiManager()
        logger.info(f'Done with loading.')

    def __init__(self, image_file_path: str, config: Config):
        self.config = config
        self.log_folder = image_file_path + '_logs'
        self.image_file_path = image_file_path
        os.makedirs(self.log_folder, exist_ok=True)
        self.roi_manager = self.pixels = self.dataset = self.imp = self.ij = None
        np_data_path = os.path.join(self.log_folder, 'pixels.npy')
        if os.path.exists(np_data_path):
            self.pixels = np.load(np_data_path)
        else:
            self.init_ij()
            np.save(np_data_path, self.pixels)

    def select_last_roi(self):
        self.roi_manager.deselect()
        count = self.roi_manager.getCount()
        self.roi_manager.select(count-1)
        return self.roi_manager.getRoi(count-1)

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
        label_mask_clean = label_mask.copy()
        for r in region_properties:
            if r.area > self.config.max_size:
                label_mask_clean[label_mask_clean == r.label] = 0
        region_properties = measure.regionprops(label_mask_clean)
        return smooth, label_mask_clean, region_properties

    def plot(self):
        smooth, label_mask_clean, region_properties = self.find_blobs(0)
        fig, ax = plt.subplots(1, 1, figsize=(20, 20))
        ax.imshow(smooth)
        ax.contour(label_mask_clean, colors='red')
        fig.tight_layout()
        fig.savefig(os.path.join(self.log_folder, 'cells.pdf'))

    def segment(self):
        self.init_ij()
        logger.warning("Select your cell.")
        # input('Press Enter to proceed.')
        roi = self.select_last_roi()
        x = 1
