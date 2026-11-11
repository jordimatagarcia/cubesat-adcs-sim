"""Command-line entry point: ``adcssim``."""

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

from adcssim import __version__
from adcssim.backends.config import VARIANTS
from adcssim.launch import RunRequest, prepare, run_batch
from adcssim.scenario import ScenarioError


def _add_overrides(p: argparse.ArgumentParser, what: str) -> None:
    p.add_argument("--backend", help=f"ADCS backend{what}, overriding the scenario (e.g. none)")
    p.add_argument(
        "--variant",
        choices=VARIANTS,
        help="ADCS algorithm variant, overriding the scenario",
    )
    p.add_argument("--duration", type=float, help="run length in seconds, overriding the scenario")
    p.add_argument(
        "--seed", type=int, help="random seed of the sensor noise, overriding the scenario"
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="adcssim",
        description="CubeSat attitude determination and control simulator.",
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    sub = parser.add_subparsers(dest="command")

    run = sub.add_parser("run", help="run a scenario and write its telemetry log")
    run.add_argument("scenario", type=Path, help="scenario file (.toml)")
    _add_overrides(run, "")
    run.add_argument(
        "--out", type=Path, default=Path("runs"), help="output directory (default: runs/)"
    )
    run.add_argument("--quiet", action="store_true", help="no progress output")

    view = sub.add_parser(
        "view", help="open the visualization: a scenario (live) or a .parquet log (replay)"
    )
    view.add_argument("path", type=Path, help="scenario file (.toml) or telemetry log (.parquet)")
    _add_overrides(view, " for a live scenario")
    view.add_argument(
        "--speed", type=float, default=60.0, help="simulated seconds per second (default: 60)"
    )
    view.add_argument("--paused", action="store_true", help="start paused")

    sub.add_parser(
        "gui", help="open the launcher: choose a scenario, implementation and run options"
    )
    return parser


def _request(path: Path, args: argparse.Namespace) -> RunRequest:
    return RunRequest(
        scenario=path,
        backend=args.backend,
        variant=args.variant,
        duration_s=args.duration,
        seed=args.seed,
    )


def _error(message: object) -> int:
    print(f"error: {message}", file=sys.stderr)
    return 2


def _run(args: argparse.Namespace) -> int:
    try:
        scenario = prepare(_request(args.scenario, args))
    except (ScenarioError, ValueError) as exc:
        return _error(exc)
    report_every = max(1, scenario.steps // 20)

    def progress(k: int, n: int) -> None:
        if not args.quiet and (k % report_every == 0 or k == n):
            print(f"\r  {k}/{n} steps ({100 * k / n:.0f}%)", end="", flush=True)

    try:
        result = run_batch(scenario, args.out, progress)
    except ValueError as exc:  # e.g. a native library that is not built
        return _error(exc)
    if not args.quiet:
        print()
    print("\n".join(result.summary()))
    return 0


_VIZ_MISSING = (
    "the {what} needs the optional 'viz' dependencies ({name} is missing).\n"
    "Install them with:  uv sync --extra viz"
)


def _view(args: argparse.Namespace) -> int:
    try:
        from adcssim.viz.app import run_viewer
    except ImportError as exc:
        return _error(_VIZ_MISSING.format(what="viewer", name=exc.name))
    if not args.path.exists():
        return _error(f"{args.path} does not exist")
    try:
        return run_viewer(
            args.path, None, args.speed, autoplay=not args.paused, request=_request(args.path, args)
        )
    except (ScenarioError, ValueError, KeyError) as exc:
        return _error(exc)


def _gui(_args: argparse.Namespace) -> int:
    try:
        from adcssim.viz.launcher import run_launcher
    except ImportError as exc:
        return _error(_VIZ_MISSING.format(what="launcher", name=exc.name))
    return run_launcher()


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command == "run":
        return _run(args)
    if args.command == "view":
        return _view(args)
    if args.command == "gui":
        return _gui(args)
    parser.print_help()
    return 0
