"""Command-line entry point: `adcssim`."""

import argparse
from collections.abc import Sequence

from adcssim import __version__


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="adcssim",
        description="CubeSat attitude determination and control simulator.",
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    build_parser().parse_args(argv)
    return 0
