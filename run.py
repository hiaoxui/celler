from argparse import ArgumentParser

from celler.ij_port import IJPort
from celler.utils import logger, Config


def main():
    parser = ArgumentParser()
    parser.add_argument('action', metavar='ACTION', type=str, choices=['plot', 'segment'])
    parser.add_argument('-i', metavar='IMAGE', type=str, required=True, help='image path (tif)')
    # Parameters
    parser.add_argument('--gaussian-sigma', type=float, default=2.0)
    parser.add_argument('--threshold-adjustment', type=float, default=-1.5)
    parser.add_argument('--min-size', type=int, default=20**2)
    parser.add_argument('--max-hole', type=float, default=40**2)
    parser.add_argument('--max-size', type=float, default=1000**2)
    parser.add_argument('--track-memory', type=float, default=2)
    parser.add_argument('--search-range', type=float, default=5e-2)

    args = parser.parse_args()
    logger.setLevel('INFO')
    config = Config(
        args.gaussian_sigma, args.threshold_adjustment, args.min_size, args.max_hole, args.max_size,
        args.track_memory, args.search_range
    )

    port = IJPort(args.i, config)
    if args.action == 'plot':
        port.plot()
    else:
        port.segment()


if __name__ == '__main__':
    main()
