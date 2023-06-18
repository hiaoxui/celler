import json
from argparse import ArgumentParser
import os
import shutil
import pandas as pd


def run():
    p = ArgumentParser()
    p.add_argument('-i')
    p.add_argument('-o')
    p.add_argument('--pixel-size', type=float)
    p.add_argument('--time-interval', type=float)
    args = p.parse_args()

    cell_order = 0
    for f in os.listdir(args.i):
        if not f.endswith('.csv'):
            continue
        df = pd.read_csv(os.path.join(args.i, f))
        tgt_folder = os.path.join(args.o, f'cell_{cell_order:03}')
        os.makedirs(tgt_folder, exist_ok=True)
        shutil.copy(os.path.join(args.i, f), os.path.join(tgt_folder, f'cell_{cell_order:03}_measurements.csv'))
        shutil.copy(os.path.join(args.i, f[:-4]+'.zip'), os.path.join(tgt_folder, 'RoiSet.zip'))
        meta = {
            'x': df.iloc[0].X, 'y': df.iloc[0].Y, 'cell_name': f'cell_{cell_order:03}',
            'timestamp': os.path.getmtime(os.path.join(args.i, f)),
            'pixel_size': args.pixel_size, 'time_deltas': [args.time_interval] * len(df),
            'pixel_size_unit': 'µm', 'czi_path': f'cli://{args.pixel_size}:{args.time_interval}'
        }
        with open(os.path.join(tgt_folder, 'meta.json'), 'w') as fp:
            json.dump(meta, fp, indent=2)
        cell_order += 1


if __name__ == '__main__':
    run()
