from typing import *
import zipfile
from struct import unpack
from dataclasses import dataclass, field
import numpy as np


# header 1
VERSION_OFFSET = 4
TYPE = 6
TOP = 8
LEFT = 10
BOTTOM = 12
RIGHT = 14
N_COORDINATES = 16
X1 = 18
Y1 = 22
X2 = 26
Y2 = 30
XD = 18
YD = 22
WIDTHD = 26
HEIGHTD = 30
SIZE = 18
STROKE_WIDTH = 34
SHAPE_ROI_SIZE = 36
STROKE_COLOR = 40
FILL_COLOR = 44
SUBTYPE = 48
OPTIONS = 50
ARROW_STYLE = 52
FLOAT_PARAM = 52
POINT_TYPE= 52
ARROW_HEAD_SIZE = 53
ROUNDED_RECT_ARC_SIZE = 54
POSITION = 56
HEADER2_OFFSET = 60
COORDINATES = 64


# header 2
C_POSITION = 4
Z_POSITION = 8
T_POSITION = 12
NAME_OFFSET = 16
NAME_LENGTH = 20
OVERLAY_LABEL_COLOR = 24
OVERLAY_FONT_SIZE = 28
GROUP = 30
IMAGE_OPACITY = 31
IMAGE_SIZE = 32
FLOAT_STROKE_WIDTH = 36
ROI_PROPS_OFFSET = 40
ROI_PROPS_LENGTH = 44
COUNTERS_OFFSET = 48

# options
SPLINE_FIT = 1
DOUBLE_HEADED = 2
OUTLINE = 4
OVERLAY_LABELS = 8
OVERLAY_NAMES = 16
OVERLAY_BACKGROUNDS = 32
OVERLAY_BOLD = 64
SUB_PIXEL_RESOLUTION = 128
DRAW_OFFSET = 256
ZERO_TRANSPARENT = 512
SHOW_LABELS = 1024
SCALE_LABELS = 2048
PROMPT_BEFORE_DELETING = 4096
SCALE_STROKE_WIDTH = 8192


class ROIReader:
    def __init__(self, bites: bytes):
        self.bites = bites
        self.unused = set(range(4, len(bites)))

    def _numbers(self, position, width, count, symbol):
        parsed = unpack('>' + symbol*count, self.bites[position:width*count+position])
        self.unused -= set(range(position, position+count*width))
        if count == 1:
            return parsed[0]
        return parsed

    def _short(self, position, count=1):
        return self._numbers(position, 2, count, 'h')

    def _long(self, position, count=1):
        return self._numbers(position, 4, count, 'i')

    def _single(self, position, count=1):
        return self._numbers(position, 4, count, 'f')

    def parse(self):
        assert self.bites[:4] == b'Iout'
        frame = ROIFrame()
        frame.file_version, frame.roi_type = self._short(VERSION_OFFSET, 2)
        frame.top, frame.left, frame.bottom, frame.right = self._short(TOP, 4)
        n = frame.n_coordinates = self._short(N_COORDINATES)
        frame.options = self._short(OPTIONS)
        frame.points = np.zeros([n, 2], dtype=np.int64)
        frame.point_floats = np.zeros([n, 2], dtype=np.float32)
        frame.points[:, 0] = self._short(COORDINATES, n)
        frame.points[:, 1] = self._short(COORDINATES + n*2, n)
        frame.points[:, 0] += frame.left
        frame.points[:, 1] += frame.top
        frame.point_floats[:, 0] = self._single(COORDINATES + n * 4)
        frame.point_floats[:, 1] = self._single(COORDINATES + n * 8)
        frame.name = self.read_name()
        self.read_stroke(frame)
        self.read_props()

        return frame

    def read_name(self):
        name_offset = self._long(self.header2_offset+NAME_OFFSET)
        name_length = self._long(self.header2_offset+NAME_LENGTH)
        name_chars = self._short(name_offset, name_length)
        return ''.join((map(chr, name_chars)))

    def read_stroke(self, frame):
        frame.stroke_width = self._short(STROKE_WIDTH)
        frame.stroke_width_float = self._single(self.header2_offset+FLOAT_STROKE_WIDTH)
        frame.stroke_color = self._long(STROKE_COLOR)
        frame.fill_color = self._long(FILL_COLOR)

    def read_props(self):
        offset = self._long(self.header2_offset + ROI_PROPS_OFFSET)
        length = self._long(self.header2_offset + ROI_PROPS_LENGTH)
        assert offset == length == 0

    @property
    def header2_offset(self):
        return self._long(HEADER2_OFFSET)


@dataclass()
class ROIFrame:
    file_version: int = None
    roi_type: int = 0
    n_coordinates: int = None
    options: int = 128
    name: str = None

    stroke_width: int = 0
    stroke_width_float: float = 0
    stroke_color: int = 0
    fill_color: int = 0

    top: int = None
    left: int = None
    bottom: int = None
    right: int = None

    points: np.ndarray = None
    point_floats: np.ndarray = None


class ROIFile:
    def __init__(self, filepath: str):
        zipped = zipfile.ZipFile(filepath)
        files = dict()
        for fn in zipped.filelist:
            files[fn.filename] = ROIReader(zipped.read(fn.filename)).parse()


def test():
    rf = ROIFile('/home/hiaoxui/images/Ax2 mry-C2B-iai_AcquisitionBlock3.czi/RoiSet.zip')


if __name__ == '__main__':
    test()
