"""Command-line entry point: ``adcssim``."""

import argparse
import dataclasses
import math
import sys
import time
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

from adcssim import __version__
from adcssim.backends import create_backend
from adcssim.runner import Simulation
from adcssim.scenario import ScenarioError, load_scenario


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="adcssim",
        description="CubeSat attitude determination and control simulator.",
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    sub = parser.add_subparsers(dest="command")

    run = sub.add_parser("run", help="run a scenario and write its telemetry log")
    run.add_argument("scenario", type=Path, help="scenario file (.toml)")
    run.add_argument("--backend", help="ADCS backend, overriding the scenario (e.g. none)")
    run.add_argument(
        "--duration", type=float, help="run length in seconds, overriding the scenario"
    )
    run.add_argument(
        "--out", type=Path, default=Path("runs"), help="output directory (default: runs/)"
    )
    run.add_argument("--quiet", action="store_true", help="no progress output")

    view = sub.add_parser(
        "view", help="open the visualization: a scenario (live) or a .parquet log (replay)"
    )
    view.add_argument("path", type=Path, help="scenario file (.toml) or telemetry log (.parquet)")
    view.add_argument("--backend", help="ADCS backend for a live scenario (e.g. none)")
    view.add_argument(
        "--speed", type=float, default=60.0, help="simulated seconds per second (default: 60)"
    )
    view.add_argument("--paused", action="store_true", help="start paused")
    return parser


def _run(args: argparse.Namespace) -> int:
    try:
        scenario = load_scenario(args.scenario)
    except (OSError, ScenarioError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    if args.duration is not None:
        if args.duration <= 0:
            print("error: --duration must be positive", file=sys.stderr)
            return 2
        scenario = dataclasses.replace(scenario, duration_s=args.duration)
    try:
        backend = create_backend(args.backend or scenario.backend)
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    sim = Simulation(scenario, backend)
    report_every = max(1, scenario.steps // 20)

    def progress(k: int, n: int) -> None:
        if not args.quiet and (k % report_every == 0 or k == n):
            print(f"\r  {k}/{n} steps ({100 * k / n:.0f}%)", end="", flush=True)

    tic = time.perf_counter()
    telemetry = sim.run(progress)
    wall = time.perf_counter() - tic
    if not args.quiet:
        print()

    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    out = args.out / f"{scenario.name}_{backend.name}_{stamp}.parquet"
    telemetry.write(out)

    eclipse = telemetry.get("eclipse").astype(bool)
    rate = np.degrees(np.linalg.norm(telemetry.get("w_true"), axis=1))
    print(f"scenario   {scenario.name} ({backend.name})")
    print(f"simulated  {scenario.duration_s:.0f} s in {wall:.1f} s wall time")
    print(f"eclipse    {100 * eclipse.mean():.1f}% of the run")
    print(f"body rate  {rate[0]:.2f} °/s at start, {rate[-1]:.2f} °/s at end")
    error = telemetry.get("attitude_error")
    if np.any(np.isfinite(error)):
        print(f"att. error {math.degrees(float(np.nanmedian(error))):.3f}° median")
    print(f"log        {out}")
    return 0


def _view(args: argparse.Namespace) -> int:
    try:
        from adcssim.viz.app import run_viewer
    except ImportError as exc:
        print(
            f"error: the viewer needs the optional 'viz' dependencies ({exc.name} is missing).\n"
            "Install them with:  uv sync --extra viz",
            file=sys.stderr,
        )
        return 2
    if not args.path.exists():
        print(f"error: {args.path} does not exist", file=sys.stderr)
        return 2
    try:
        return run_viewer(args.path, args.backend, args.speed, autoplay=not args.paused)
    except (ScenarioError, ValueError, KeyError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command == "run":
        return _run(args)
    if args.command == "view":
        return _view(args)
    parser.print_help()
    return 0
