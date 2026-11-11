"""Describe, check and start a run, without any GUI dependency.

The command line (``adcssim run``/``view``) and the launcher window (``adcssim gui``) both go
through this module, so a run started from the GUI can always be repeated from the command
line with the same result:

- :func:`list_scenarios` and :func:`backend_status` say what can be chosen;
- :class:`RunRequest` is one choice of scenario, implementation, variant and overrides,
  and :func:`prepare` turns it into a :class:`~adcssim.scenario.Scenario` (the scenario file
  itself is never modified);
- :func:`run_batch` runs a prepared scenario to the end and writes its telemetry log.
"""

import ast
import dataclasses
import inspect
import itertools
import math
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

from adcssim.backends import BACKENDS, AdcsConfig, create_backend
from adcssim.backends.config import VARIANTS
from adcssim.runner import Simulation
from adcssim.scenario import Scenario, ScenarioError, load_scenario
from adcssim.telemetry import Telemetry

REPO = Path(__file__).resolve().parents[2]
SCENARIO_DIR = REPO / "scenarios"
RUNS_DIR = Path("runs")

BACKEND_LABELS = {
    "python-ref": "Python reference",
    "c": "C library",
    "rust": "Rust library",
    "none": "No ADCS (open loop)",
}
"""Display names, in the order the launcher lists them."""

BUILD_COMMANDS = {
    "c": "cmake -S adcs-c -B adcs-c/build && cmake --build adcs-c/build",
    "rust": "cargo build --release --manifest-path adcs-rs/Cargo.toml",
}

VARIANT_LABELS = {
    "thesis": "Thesis (FLAE + two-stage UKF)",
    "improved": "Improved (error-state UKF)",
    "mekf": "MEKF (multiplicative EKF)",
}


# ---- what can be chosen ----------------------------------------------------------------


@dataclass(frozen=True)
class ScenarioInfo:
    """A scenario file as listed by the launcher. ``error`` is set, and the other fields
    are best-effort, when the file does not load."""

    path: Path
    name: str
    description: str = ""
    duration_s: float = math.nan
    backend: str = ""
    variant: str = ""
    error: str | None = None


def list_scenarios(directory: Path = SCENARIO_DIR) -> list[ScenarioInfo]:
    """Every ``*.toml`` in ``directory``, sorted by file name. Files that fail to load are
    listed too, with the error, so the user sees why one is unusable."""
    out = []
    for path in sorted(directory.glob("*.toml")):
        try:
            sc = load_scenario(path)
        except (OSError, ScenarioError) as exc:
            out.append(ScenarioInfo(path=path, name=path.stem, error=str(exc)))
            continue
        out.append(
            ScenarioInfo(
                path=path,
                name=sc.name,
                description=sc.description,
                duration_s=sc.duration_s,
                backend=sc.backend,
                variant=sc.adcs.variant,
            )
        )
    return out


@dataclass(frozen=True)
class BackendStatus:
    name: str
    label: str
    available: bool
    reason: str = ""
    """Why it is unavailable, with the command that builds it."""
    build_command: str = ""


def backend_status() -> list[BackendStatus]:
    """Each backend and whether it can run here. The C and Rust libraries must be built
    first (or pointed to with ``ADCSSIM_C_LIB`` / ``ADCSSIM_RUST_LIB``)."""
    from adcssim.backends.native import LibraryNotFoundError, library_path

    out = []
    for name in BACKEND_LABELS:
        if name not in ("c", "rust"):
            out.append(BackendStatus(name, BACKEND_LABELS[name], True))
            continue
        try:
            library_path(name)
        except LibraryNotFoundError as exc:
            reason = f"{exc}. Build it with: {BUILD_COMMANDS[name]}"
            out.append(
                BackendStatus(name, BACKEND_LABELS[name], False, reason, BUILD_COMMANDS[name])
            )
        else:
            out.append(BackendStatus(name, BACKEND_LABELS[name], True))
    return out


# ---- ADCS settings ---------------------------------------------------------------------

SECTIONS = ("FLAE", "Two-stage UKF (thesis)", "Error-state filters (improved, MEKF)", "Control")


@dataclass(frozen=True)
class AdcsField:
    """One editable :class:`AdcsConfig` setting."""

    name: str
    kind: str
    """``float``, ``int``, ``bool`` or ``vector``."""
    default: Any
    section: str
    doc: str = ""


def _section(name: str) -> str:
    if name.startswith("flae_"):
        return SECTIONS[0]
    if name in ("process_dt", "ukf_alpha", "p_x0", "p_b0", "w_x", "w_b", "v"):
        return SECTIONS[1]
    if name.startswith(("usque_", "sigma_", "p0_", "fault_")):
        return SECTIONS[2]
    return SECTIONS[3]


def _field_docs() -> dict[str, str]:
    """The attribute docstrings of :class:`AdcsConfig` (the string after a field)."""
    tree = ast.parse(inspect.getsource(AdcsConfig))
    body = tree.body[0].body  # type: ignore[attr-defined]
    docs = {}
    for node, nxt in itertools.pairwise(body):
        if (
            isinstance(node, ast.AnnAssign)
            and isinstance(node.target, ast.Name)
            and isinstance(nxt, ast.Expr)
            and isinstance(nxt.value, ast.Constant)
            and isinstance(nxt.value.value, str)
        ):
            docs[node.target.id] = " ".join(nxt.value.value.split())
    return docs


def adcs_fields() -> list[AdcsField]:
    """The settings the launcher's advanced section shows, in declaration order. The
    variant is chosen separately and is not listed."""
    defaults = AdcsConfig()
    docs = _field_docs()
    out = []
    for f in dataclasses.fields(AdcsConfig):
        if f.name == "variant":
            continue
        value = getattr(defaults, f.name)
        if isinstance(value, bool):
            kind = "bool"
        elif isinstance(value, int):
            kind = "int"
        elif isinstance(value, np.ndarray):
            kind, value = "vector", tuple(float(v) for v in value)
        else:
            kind = "float"
        out.append(AdcsField(f.name, kind, value, _section(f.name), docs.get(f.name, "")))
    return out


def parse_adcs_value(field_: AdcsField, text: str) -> Any:
    """Typed value of a setting from its text form (vectors: numbers separated by commas
    or spaces). Raises ``ValueError`` with the setting's name on bad input."""
    text = text.strip()
    try:
        if field_.kind == "bool":
            lowered = text.lower()
            if lowered in ("true", "1", "yes", "on"):
                return True
            if lowered in ("false", "0", "no", "off"):
                return False
            raise ValueError("expected true or false")
        if field_.kind == "int":
            return int(text)
        if field_.kind == "vector":
            parts = text.replace(",", " ").split()
            if not parts:
                raise ValueError("expected numbers")
            return np.array([float(p) for p in parts])
        return float(text)
    except ValueError as exc:
        raise ValueError(f"{field_.name}: {exc}") from None


def format_adcs_value(value: Any) -> str:
    """Text form of a setting, the inverse of :func:`parse_adcs_value`."""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, np.ndarray | tuple | list):
        return ", ".join(f"{float(v):g}" for v in value)
    if isinstance(value, float):
        return f"{value:g}"
    return str(value)


# ---- a run request -------------------------------------------------------------------


@dataclass(frozen=True)
class RunRequest:
    """One run: a scenario file and what to change in it. ``None`` keeps the scenario's
    value."""

    scenario: Path
    backend: str | None = None
    variant: str | None = None
    duration_s: float | None = None
    seed: int | None = None
    adcs: Mapping[str, Any] = field(default_factory=dict)
    """ADCS settings by :class:`AdcsConfig` field name."""

    def command_line(self, mode: str = "run") -> list[str]:
        """The equivalent ``adcssim`` command (ADCS settings other than the variant have no
        command-line form and are not included)."""
        args = ["adcssim", mode, str(self.scenario)]
        for flag, value in (
            ("--backend", self.backend),
            ("--variant", self.variant),
            ("--duration", self.duration_s),
            ("--seed", self.seed),
        ):
            if value is not None:
                args += [flag, f"{value:g}" if isinstance(value, float) else str(value)]
        return args


def apply_overrides(
    scenario: Scenario,
    *,
    backend: str | None = None,
    variant: str | None = None,
    duration_s: float | None = None,
    seed: int | None = None,
    adcs: Mapping[str, Any] | None = None,
) -> Scenario:
    """``scenario`` with the given values replaced. Raises ``ValueError`` on an invalid
    value, including an ADCS configuration that no implementation would accept."""
    changes: dict[str, Any] = {}
    if backend is not None:
        if backend not in BACKENDS:
            raise ValueError(f"unknown ADCS backend {backend!r} (available: {', '.join(BACKENDS)})")
        changes["backend"] = backend
    if duration_s is not None:
        if not (math.isfinite(duration_s) and duration_s > 0):
            raise ValueError("the duration must be positive")
        changes["duration_s"] = float(duration_s)
    if seed is not None:
        if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
            raise ValueError("the seed must be a non-negative integer")
        changes["seed"] = seed
    settings = dict(adcs or {})
    if variant is not None:
        if variant not in VARIANTS:
            raise ValueError(f"unknown ADCS variant {variant!r} ({', '.join(VARIANTS)})")
        settings["variant"] = variant
    if settings:
        known = {f.name for f in dataclasses.fields(AdcsConfig)}
        unknown = sorted(set(settings) - known)
        if unknown:
            raise ValueError(f"unknown ADCS setting(s): {', '.join(unknown)}")
        try:
            changes["adcs"] = dataclasses.replace(scenario.adcs, **settings)
        except TypeError as exc:
            raise ValueError(str(exc)) from None
    out = dataclasses.replace(scenario, **changes) if changes else scenario
    if out.backend != "none":
        out.adcs.validate(out.dt_adcs_s, out.spacecraft.inertia, out.actuators.max_dipole)
    return out


def prepare(request: RunRequest) -> Scenario:
    """Load the request's scenario and apply its overrides. Raises ``ScenarioError`` for a
    file that does not load and ``ValueError`` for an invalid override."""
    try:
        scenario = load_scenario(request.scenario)
    except OSError as exc:
        raise ScenarioError(f"{request.scenario}: {exc.strerror or exc}") from None
    return apply_overrides(
        scenario,
        backend=request.backend,
        variant=request.variant,
        duration_s=request.duration_s,
        seed=request.seed,
        adcs=request.adcs,
    )


# ---- batch run -----------------------------------------------------------------------


class RunCancelledError(Exception):
    """Raised by :func:`run_batch` when ``should_cancel`` returned true."""


@dataclass(frozen=True)
class BatchResult:
    scenario: Scenario
    backend: str
    log: Path
    wall_s: float
    telemetry: Telemetry = field(repr=False)

    def summary(self) -> list[str]:
        """The report ``adcssim run`` prints."""
        sc, tel = self.scenario, self.telemetry
        eclipse = tel.get("eclipse").astype(bool)
        rate = np.degrees(np.linalg.norm(tel.get("w_true"), axis=1))
        lines = [
            f"scenario   {sc.name} ({self.backend})",
            f"simulated  {sc.duration_s:.0f} s in {self.wall_s:.1f} s wall time",
            f"eclipse    {100 * eclipse.mean():.1f}% of the run",
            f"body rate  {rate[0]:.2f} °/s at start, {rate[-1]:.2f} °/s at end",
        ]
        error = tel.get("attitude_error")
        if np.any(np.isfinite(error)):
            lines.append(f"att. error {math.degrees(float(np.nanmedian(error))):.3f}° median")
        lines.append(f"log        {self.log}")
        return lines


def log_path(out_dir: Path, scenario: Scenario, backend: str) -> Path:
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    return out_dir / f"{scenario.name}_{backend}_{stamp}.parquet"


def run_batch(
    scenario: Scenario,
    out_dir: Path = RUNS_DIR,
    progress: Callable[[int, int], None] | None = None,
    should_cancel: Callable[[], bool] | None = None,
) -> BatchResult:
    """Run ``scenario`` on its backend to the end and write the telemetry log into
    ``out_dir``. ``progress(k, n)`` is called after every control period;
    ``should_cancel`` is polled at the same time and stops the run with
    :class:`RunCancelledError` (no log is written)."""
    backend = create_backend(scenario.backend)
    sim = Simulation(scenario, backend)

    def step(k: int, n: int) -> None:
        if should_cancel is not None and should_cancel():
            raise RunCancelledError
        if progress is not None:
            progress(k, n)

    tic = time.perf_counter()
    telemetry = sim.run(step)
    wall = time.perf_counter() - tic
    out = log_path(out_dir, scenario, backend.name)
    telemetry.write(out)
    return BatchResult(scenario, backend.name, out, wall, telemetry)


def recent_logs(directory: Path = RUNS_DIR, limit: int = 20) -> Sequence[Path]:
    """The newest telemetry logs in ``directory``, newest first."""
    if not directory.is_dir():
        return []
    logs = sorted(directory.glob("*.parquet"), key=lambda p: p.stat().st_mtime, reverse=True)
    return logs[:limit]
