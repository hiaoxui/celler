from typing import *
import zipfile
from struct import unpack, pack
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
POINT_TYPE = 52
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


@dataclass()
class ROIFrame:
    file_version: int = 228
    roi_type: int = 0
    roi_subtype: int = 0
    options: int = 0
    position: int = None
    name: str = None

    stroke_width: int = 0
    stroke_width_float: float = 0.0
    stroke_color: int = -256
    fill_color: int = 0

    top: int = None
    left: int = None
    bottom: int = None
    right: int = None

    points: np.ndarray = None
    point_floats: np.ndarray = None

    @property
    def subpixel(self):
        return (self.options & SUB_PIXEL_RESOLUTION) != 0


class ROIReader:
    def __init__(self, bites: bytes):
        self.bites = bites
        self.frame = ROIFrame()
        self.unused = set(range(COORDINATES, len(bites)))

    def _numbers(self, position, width, count, symbol):
        parsed = unpack('>' + symbol*count, self.bites[position:width*count+position])
        self.unused -= set(range(position, position+count*width))
        if count == 1:
            return parsed[0]
        return parsed

    def _short(self, position, count=1):
        return self._numbers(position, 2, count, 'h')

    def _byte(self, position, count=1):
        return self._numbers(position, 1, count, 'b')

    def _long(self, position, count=1):
        return self._numbers(position, 4, count, 'i')

    def _single(self, position, count=1):
        return self._numbers(position, 4, count, 'f')

    def parse(self):
        assert self.bites[:4] == b'Iout'
        frame = self.frame
        frame.file_version, frame.roi_type = self._short(VERSION_OFFSET, 2)
        frame.top, frame.left, frame.bottom, frame.right = self._short(TOP, 4)
        frame.roi_subtype = self._short(SUBTYPE)
        n = self._short(N_COORDINATES)
        frame.options = self._short(OPTIONS)
        frame.position = self._long(POSITION)
        frame.points = np.zeros([n, 2], dtype=np.int64)
        frame.points[:, 0] = self._short(COORDINATES, n)
        frame.points[:, 1] = self._short(COORDINATES + n*2, n)
        frame.points[:, 0] += frame.left
        frame.points[:, 1] += frame.top
        if frame.subpixel:
            frame.point_floats = np.zeros([n, 2], dtype=np.float32)
            frame.point_floats[:, 0] = self._single(COORDINATES + n * 4, n)
            frame.point_floats[:, 1] = self._single(COORDINATES + n * 8, n)
        frame.name = self.read_name()

        channel = self._long(self.header2_offset + C_POSITION)
        slice_int = self._long(self.header2_offset + Z_POSITION)
        frame_int = self._long(self.header2_offset + T_POSITION)
        overlay_label_color = self._long(self.header2_offset + OVERLAY_LABEL_COLOR)
        overlay_font_size = self._short(self.header2_offset + OVERLAY_FONT_SIZE)
        image_opacity = self._byte(self.header2_offset + IMAGE_OPACITY)
        image_size = self._long(self.header2_offset + IMAGE_SIZE)
        group = self._byte(self.header2_offset + GROUP)
        assert channel == slice_int == frame_int == overlay_label_color \
               == overlay_font_size == image_size == image_opacity == group == 0

        self.read_stroke()
        self.read_props()
        self.read_counter()
        assert group == 0

        return frame

    def read_name(self):
        name_offset = self._long(self.header2_offset+NAME_OFFSET)
        name_length = self._long(self.header2_offset+NAME_LENGTH)
        name_chars = self._short(name_offset, name_length)
        return ''.join((map(chr, name_chars)))

    def read_stroke(self):
        self.frame.stroke_width = self._short(STROKE_WIDTH)
        self.frame.stroke_width_float = self._single(self.header2_offset+FLOAT_STROKE_WIDTH)
        self.frame.stroke_color = self._long(STROKE_COLOR)
        self.frame.fill_color = self._long(FILL_COLOR)
        # assert self._short(STROKE_WIDTH) == self._long(STROKE_COLOR) == self._long(FILL_COLOR) == 0
        # assert self._single(self.header2_offset+FLOAT_STROKE_WIDTH) == 0.

    def read_props(self):
        offset = self._long(self.header2_offset + ROI_PROPS_OFFSET)
        length = self._long(self.header2_offset + ROI_PROPS_LENGTH)
        assert offset == length == 0

    def read_counter(self):
        offset = self._long(self.header2_offset + COUNTERS_OFFSET)
        assert offset == 0

    @property
    def header2_offset(self):
        return self._long(HEADER2_OFFSET)

    def iter_unused(self):
        for s in self.unused:
            if s % 2 == 1:
                continue
            print(f'idx: {s:3}', unpack('>h', self.bites[s:s+2])[0])


class ROIWriter:
    def __init__(self, frame: ROIFrame):
        self.frame = frame
        length = 64  # header 1
        length += self.frame.points.shape[0] * (4 if not frame.subpixel else 12)
        self.header2_offset = length
        length += 52 + 12  # header 2 + empty pads
        length += len(frame.name) * 2
        self.bites = [0] * length

    def _number(self, position, numbers: Union[List[int], List[float], int, float], symbol, width):
        if not isinstance(numbers, list):
            return self._number(position, [numbers], symbol, width)
        self.bites[position:position+len(numbers)*width] = list(pack(f'>' + symbol * len(numbers), *numbers))

    def _short(self, position, numbers):
        self._number(position, numbers, 'h', 2)

    def _long(self, position, numbers):
        self._number(position, numbers, 'i', 4)

    def _single(self, position, numbers):
        self._number(position, numbers, 'f', 4)

    def _header1(self):
        self.bites[:4] = list(b'Iout')
        self._short(VERSION_OFFSET, self.frame.file_version)
        self._short(TYPE, self.frame.roi_type)
        self._short(TOP, self.frame.top)
        self._short(LEFT, self.frame.left)
        self._short(BOTTOM, self.frame.bottom)
        self._short(RIGHT, self.frame.right)
        self._short(N_COORDINATES, self.frame.points.shape[0])
        self._short(OPTIONS, self.frame.options)
        self._long(POSITION, self.frame.position)
        self._short(SUBTYPE, self.frame.roi_subtype)
        self._long(HEADER2_OFFSET, self.header2_offset)
        self._long(STROKE_COLOR, self.frame.stroke_color)

    def _points(self):
        n = self.frame.points.shape[0]
        xs = (self.frame.points[:, 0] - self.frame.left).tolist()
        ys = (self.frame.points[:, 1] - self.frame.top).tolist()
        self._short(COORDINATES, xs)
        self._short(COORDINATES+2*n, ys)
        if self.frame.subpixel:
            xf, yf = self.frame.point_floats.T.tolist()
            self._single(COORDINATES+4*n, xf)
            self._single(COORDINATES+8*n, yf)

    def _header2(self):
        self._long(self.header2_offset+NAME_LENGTH, len(self.frame.name))
        self._long(self.header2_offset+NAME_OFFSET, self.header2_offset+52+12)
        self._short(self.header2_offset + 52 + 12, list(map(ord, self.frame.name)))

    def generate(self):
        self._header1()
        self._points()
        self._header2()
        return bytes(self.bites)

    def sanity_check(self, bites):
        generated = self.generate()
        b, g = np.array(list(bites)), np.array(list(generated))
        assert (b == g).all()


class ROIFile:
    def __init__(self, filepath: str):
        zipped = zipfile.ZipFile(filepath)
        files = dict()
        for fn in zipped.filelist:
            bites = zipped.read(fn.filename)
            frame = ROIReader(bites).parse()
            files[fn.filename] = frame
            ROIWriter(frame).sanity_check(bites)
