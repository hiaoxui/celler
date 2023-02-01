import numpy as np


class Region:
    def __init__(self, region_property):
        self.label: int = region_property.label
        self.centroid: np.ndarray = np.array([region_property.centroid[1], region_property.centroid[0]])
        self.centroid_local: np.ndarray = np.array(
            [region_property.centroid_local[1], region_property.centroid_local[0]]
        )
        self.coords: np.ndarray = region_property.coords
        self.bbox = region_property.bbox
        self.area = region_property.area
        self.axis_major_length, self.axis_major_length = \
            region_property.axis_major_length, region_property.axis_minor_length
        self.eccentricity, self.extent = region_property.eccentricity, region_property.extent
