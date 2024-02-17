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
        df = pd.read_csv(os.path.join(args.i, f), encoding='iso-8859-1')
        tgt_folder = os.path.join(args.o, f'cell_{cell_order:03}')
        os.makedirs(tgt_folder, exist_ok=True)
        src_zip_path = os.path.join(args.i, f[:-4]+'.zip')
        tgt_csv_path = os.path.join(tgt_folder, f'cell_{cell_order:03}_measurements.csv')
        if os.path.exists(src_zip_path):
            shutil.copy(src_zip_path, os.path.join(tgt_folder, 'RoiSet.zip'))
        # hard-coded for chemotaxis; sometimes it is C3 and C4 for X and Y
        df = df.rename(columns={'C3': 'X', 'C4': 'Y'})
        df.to_csv(tgt_csv_path)
        meta = {
            'x': df.iloc[0].X, 'y': df.iloc[0].X, 'cell_name': f'cell_{cell_order:03}',
            'timestamp': os.path.getmtime(os.path.join(args.i, f)),
            'pixel_size': args.pixel_size, 'time_deltas': [args.time_interval] * len(df),
            'pixel_size_unit': 'µm', 'czi_path': f'cli://{args.pixel_size}:{args.time_interval}'
        }
        with open(os.path.join(tgt_folder, 'meta.json'), 'w') as fp:
            json.dump(meta, fp, indent=2)
        cell_order += 1


if __name__ == '__main__':
    run()
