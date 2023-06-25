"""
If the cell measurements csv files are NOT saved into separate folders
(i.e. cells are not separated by folders), will sort into folders.
The `pixel_size` and `time-interval` should be passed through CLI.
"""
from argparse import ArgumentParser
import os
import shutil
from param import read_from_cli


def main():
    parser = ArgumentParser()
    parser.add_argument('-i', type=str, help='input folder')
    parser.add_argument('-o', type=str, help='output folder')
    parser.add_argument('--pixel-size', type=float, required=True)
    parser.add_argument('--time-interval', type=float, required=True)
    args = parser.parse_args()
    if os.path.exists(args.o):
        choice = input(f'Path {args.o} exists. Continue? (y/n)')
        if choice != 'y':
            return

    ci = 0
    for f in os.listdir(args.i):
        if not f.endswith('.csv'):
            continue
        cell_name = f'cell_{ci:03}'
        os.makedirs(os.path.join(args.o, cell_name), exist_ok=True)
        shutil.copy(os.path.join(args.i, f), os.path.join(args.o, cell_name, f'{cell_name}_measurements.csv'))
        shutil.copy(os.path.join(args.i, f[:-4] + '.zip'), os.path.join(args.o, cell_name, 'RoiSet.zip'))
        ci += 1
    args.log = args.o
    read_from_cli(args)


if __name__ == '__main__':
    main()
