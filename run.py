from argparse import ArgumentParser

from celler.ij_port import IJPort
from celler.utils import logger, Config


def main():
    parser = ArgumentParser()
    parser.add_argument('action', metavar='ACTION', type=str, choices=['plot', 'segment'])
    parser.add_argument('-i', metavar='IMAGE', type=str, required=True, help='image path (tif)')
    # Parameters
    # parser.add_argument('--gaussian-sigma', type=float, default=5.0)
    # parser.add_argument('--min-size', type=int, default=20**2)
    # parser.add_argument('--max-size', type=float, default=1000**2)
    # parser.add_argument('--search-range', type=float, default=500.)
    parser.add_argument('--debug', action='store_true')
    args = parser.parse_args()
    logger.setLevel('INFO')
    config = Config(debug=args.debug)
    logger.setLevel('DEBUG' if args.debug else 'WARNING')

    port = IJPort(args.i, config)
    if args.action == 'plot':
        port.plot()
    else:
        port.segment()


if __name__ == '__main__':
    main()
