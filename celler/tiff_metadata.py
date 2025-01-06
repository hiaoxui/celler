import re
import json
import os

from PIL import Image
from PIL.TiffTags import TAGS


def get_segments(nf: int):
    while True:
        print(f'There are {nf} frames. What are the segments?')
        # print('Suppose there are 30 frames, example input: 12 7 11')
        segments = input('Number of frames for each segment: ').strip().split()
        segments = list(map(int, segments))
        if sum(segments) == nf:
            return segments
        print('Wrong inputs. Try again.')


def read_tiff_metadata(tiff_path: str, metadata_path: str):
    if not os.path.exists(metadata_path):
        with Image.open(tiff_path) as img:
            dikt = {TAGS[key] : img.tag[key] for key in img.tag_v2}
            desc = dikt['ImageDescription'][0]
            assert 'unit=micron' in desc
            # frame interval
            fv = float(re.findall(r'finterval=(\d+\.\d+)', desc)[0])
            xr, yr = dikt['XResolution'][0], dikt['YResolution'][0]
            assert xr == yr
            # pixel interval
            pi = xr[1]/xr[0]
            # number of frames
            nf = int(re.findall(r'frames=(\d+)', desc)[0])
        segments = get_segments(nf)
        meta = {
            'pixel_size': pi,
            'time_deltas': [fv] * nf,
            'pixel_size_unit': 'µm',
            'segments': segments,
        }
        with open(metadata_path, 'w') as fp:
            json.dump(meta, fp)
        return meta
    else:
        with open(metadata_path, 'r') as fp:
            return json.load(fp)
