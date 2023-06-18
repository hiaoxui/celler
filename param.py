from argparse import ArgumentParser
import re
import json
import os
import pandas as pd

try:
    import bioformats as bf
    import javabridge as jvb
except ImportError:
    pass


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


def read_from_czi(args):
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


def read_from_cli(args):
    assert args.pixel_size is not None and args.time_interval is not None
    for root, _, fns in os.walk(args.log):
        if 'meta.json' not in fns:
            continue
        csv_file = [fn for fn in fns if fn.endswith('.csv')][0]
        df = pd.read_csv(os.path.join(root, csv_file))
        n_frame = len(df)
        meta = json.load(open(os.path.join(root, 'meta.json')))
        meta['pixel_size'] = args.pixel_size
        meta['time_deltas'] = [args.time_interval] * n_frame
        meta['pixel_size_unit'] = 'µm'
        meta['segments'] = [n_frame]
        meta['czi_path'] = f'cli://{args.pixel_size}:{args.time_interval}'
        with open(os.path.join(root, 'meta.json'), 'w') as fp:
            json.dump(meta, fp, indent=2)


def main():
    parser = ArgumentParser()
    parser.add_argument('--log', type=str, required=True)
    # either input a CZI file, if metadata should be read from CZI
    parser.add_argument('--czi', type=str)
    # or manually pass them via cli
    parser.add_argument('--pixel-size', type=float)
    parser.add_argument('--time-interval', type=float)
    args = parser.parse_args()
    if args.czi is not None:
        read_from_czi(args)
    else:
        read_from_cli(args)


if __name__ == '__main__':
    main()
