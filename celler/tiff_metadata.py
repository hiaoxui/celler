import json
import re
from pathlib import Path

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


def read_tiff_metadata(tiff_path: str | Path, metadata_path: str | Path):
    tiff_path = Path(tiff_path)
    metadata_path = Path(metadata_path)
    if not metadata_path.exists():
        with Image.open(tiff_path) as img:
            dikt = {TAGS[key]: img.tag[key] for key in img.tag_v2}
            desc = dikt['ImageDescription'][0]
            # assert 'unit=micron' in desc
            # frame interval
            # if not exist, default to 1.0
            try:
                fv = float(re.findall(r'finterval=(\d+\.\d+)', desc)[0])
            except IndexError:
                fv = 1.0
            xr, yr = dikt['XResolution'][0], dikt['YResolution'][0]
            assert xr == yr
            # pixel interval
            pi = xr[1] / xr[0]
            # number of frames
            # if not found in description, load tif to get the number of frames
            try:
                nf = int(re.findall(r'frames=(\d+)', desc)[0])
            except IndexError:
                nf = 0
                while True:
                    try:
                        img.seek(nf)
                        nf += 1
                    except EOFError:
                        break
        segments = get_segments(nf)
        meta = {
            'pixel_size': pi,
            'time_deltas': [fv] * nf,
            'pixel_size_unit': '\u00b5m',
            'segments': segments,
        }
        with metadata_path.open('w') as fp:
            json.dump(meta, fp)
        return meta

    with metadata_path.open('r') as fp:
        return json.load(fp)
