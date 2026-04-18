from argparse import ArgumentParser
import logging
from pathlib import Path

from celler.ij_port import IJPort
from celler.utils import cfg

logger = logging.getLogger('cell')


def main():
    parser = ArgumentParser()
    parser.add_argument('action', metavar='ACTION', type=str, choices=['plot', 'segment'])
    parser.add_argument('image', metavar='IMAGE', type=Path, help='image path (tif)')
    parser.add_argument('--debug', action='store_true')
    args = parser.parse_args()
    cfg.debug = args.debug
    logger.setLevel('DEBUG' if args.debug else 'INFO')
    assert args.image.exists(), f'{args.image} does not exist'

    port = IJPort(args.image)
    if args.action == 'plot':
        port.plot()
    else:
        port.segment()


if __name__ == '__main__':
    main()
