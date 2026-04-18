"""
If the cell measurements csv files are NOT saved into separate folders
(i.e. cells are not separated by folders), will sort into folders.
The `pixel_size` and `time-interval` should be passed through CLI.
"""
import shutil
from argparse import ArgumentParser
from pathlib import Path

from param import read_from_cli


def main():
    parser = ArgumentParser()
    parser.add_argument('-i', type=str, help='input folder')
    parser.add_argument('-o', type=str, help='output folder')
    parser.add_argument('--pixel-size', type=float, required=True)
    parser.add_argument('--time-interval', type=float, required=True)
    args = parser.parse_args()

    input_dir = Path(args.i)
    output_dir = Path(args.o)
    if output_dir.exists():
        choice = input(f'Path {args.o} exists. Continue? (y/n)')
        if choice != 'y':
            return

    ci = 0
    for csv_path in input_dir.iterdir():
        if csv_path.suffix != '.csv':
            continue
        cell_name = f'cell_{ci:03}'
        cell_dir = output_dir / cell_name
        cell_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy(csv_path, cell_dir / f'{cell_name}_measurements.csv')
        shutil.copy(csv_path.with_suffix('.zip'), cell_dir / 'RoiSet.zip')
        ci += 1
    args.log = str(output_dir)
    read_from_cli(args)


if __name__ == '__main__':
    main()
