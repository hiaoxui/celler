from argparse import ArgumentParser
import re
import json
import os

import bioformats as bf
import javabridge as jvb


def get_czi(czi_root):
    czis = list()
    for fn in os.listdir(czi_root):
        folder = os.path.join(czi_root, fn)
        if not os.path.isdir(folder):
            continue
        for fn_ in os.listdir(folder):
            if fn_.endswith('.czi'):
                czis.append(os.path.join(folder, fn_))
    return sorted(czis)


def read_param(czi_path):
    md = bf.get_omexml_metadata(czi_path)
    o = bf.OMEXML(md)
    pixel_size = o.image(0).Pixels.get_PhysicalSizeY()
    pixel_unit = o.image(0).Pixels.get_PhysicalSizeYUnit()
    # time_delta = o.image(1).Pixels.Plane(0).get_DeltaT()
    n_frames = o.image(0).Pixels.get_SizeT()
    n_channel = o.image(0).Pixels.plane_count // n_frames
    time_delta = o.image(0).Pixels.plane(n_channel).get_DeltaT() - o.image(0).Pixels.plane(0).get_DeltaT()
    time_deltas = [time_delta] * n_frames
    return {'pixel_size': pixel_size, 'pixel_unit': pixel_unit, 'time_deltas': time_deltas, 'n_frames': n_frames}


def main():
    parser = ArgumentParser()
    parser.add_argument('--czi', type=str, required=True)
    parser.add_argument('--log', type=str, required=True)
    args = parser.parse_args()

    czis = get_czi(args.czi)

    jvb.start_vm(class_path=bf.JARS, max_heap_size='8G')
    params = list()
    for czi_path in czis:
        params.append(read_param(czi_path))
    assert len({param['pixel_size'] for param in params}) == 1
    jvb.kill_vm()
    meta = {
        'pixel_size': params[0]['pixel_size'], 'pixel_size_unit': params[0]['pixel_unit'],
        'segments': [param['n_frames'] for param in params],
        'time_deltas': sum([param['time_deltas'] for param in params], []),
        'czi_path': args.czi,
    }
    with open(os.path.join(args.log, 'meta.json'), 'w') as fp:
        json.dump(meta, fp, indent=2)
    for folder_ in os.listdir(args.log):
        folder = os.path.join(args.log, folder_)
        if os.path.isdir(folder) and re.findall(r'^cell_\d+$', folder_):
            meta_path = os.path.join(folder, 'meta.json')
            cell_meta = json.load(open(meta_path))
            cell_meta.update(meta)
            with open(meta_path, 'w') as fp:
                json.dump(cell_meta, fp, indent=2)


if __name__ == '__main__':
    main()
