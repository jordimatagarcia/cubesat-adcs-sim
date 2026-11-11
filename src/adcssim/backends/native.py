"""Backends that run a compiled ADCS library (C or Rust) through ``adcs.h``, using ctypes.

The library is looked up in the environment variable ``ADCSSIM_C_LIB`` (``ADCSSIM_RUST_LIB``
for Rust), then in the default build directory of this repository::

    cmake -S adcs-c -B adcs-c/build -DCMAKE_BUILD_TYPE=Release && cmake --build adcs-c/build

The UKF variant (``improved``) exists only in the Python reference; the libraries implement
``thesis`` and ``mekf``.
"""

import ctypes
import os
import sys
from pathlib import Path

from adcssim.backends.abi import (
    ABI_VERSION,
    REAL_SIZE,
    AdcsConfigC,
    AdcsInputC,
    AdcsOutputC,
    config_to_c,
    input_to_c,
    output_from_c,
)
from adcssim.backends.base import AdcsContext, AdcsInput, AdcsOutput, AdcsStatus

REPO = Path(__file__).resolve().parents[3]

_SUFFIX = {"darwin": ".dylib", "win32": ".dll"}.get(sys.platform, ".so")
_PREFIX = "" if sys.platform == "win32" else "lib"

LIBRARIES = {
    # backend name: (environment variable, default path)
    "c": ("ADCSSIM_C_LIB", REPO / "adcs-c" / "build" / f"{_PREFIX}adcs_c{_SUFFIX}"),
    "rust": (
        "ADCSSIM_RUST_LIB",
        REPO / "adcs-rs" / "target" / "release" / f"{_PREFIX}adcs_rs{_SUFFIX}",
    ),
}


class LibraryNotFoundError(ValueError):
    """Raised when a compiled backend's library is missing. A ``ValueError`` so the CLI
    reports it as a usage error, like an unknown backend name."""


def library_path(backend: str) -> Path:
    env, default = LIBRARIES[backend]
    path = Path(os.environ[env]) if os.environ.get(env) else default
    if not path.is_file():
        raise LibraryNotFoundError(
            f"ADCS library for backend {backend!r} not found at {path}; build it or set {env}"
        )
    return path


def _load(path: Path) -> ctypes.CDLL:
    lib = ctypes.CDLL(str(path))
    lib.adcs_abi_version.restype = ctypes.c_uint32
    lib.adcs_abi_version.argtypes = []
    lib.adcs_state_size.restype = ctypes.c_size_t
    lib.adcs_state_size.argtypes = []
    lib.adcs_real_size.restype = ctypes.c_size_t
    lib.adcs_real_size.argtypes = []
    lib.adcs_init.restype = ctypes.c_int
    lib.adcs_init.argtypes = [ctypes.c_void_p, ctypes.POINTER(AdcsConfigC)]
    lib.adcs_step.restype = ctypes.c_int
    lib.adcs_step.argtypes = [
        ctypes.c_void_p,
        ctypes.POINTER(AdcsInputC),
        ctypes.POINTER(AdcsOutputC),
    ]
    version = lib.adcs_abi_version()
    if version != ABI_VERSION:
        raise RuntimeError(f"{path}: ABI version {version}, expected {ABI_VERSION}")
    real_size = lib.adcs_real_size()
    if real_size != REAL_SIZE:
        raise RuntimeError(
            f"{path}: built with {real_size}-byte reals; this binding needs {REAL_SIZE} "
            "(a single-precision build cannot be loaded)"
        )
    return lib


class NativeAdcs:
    """An ADCS backend running a shared library that implements adcs.h."""

    def __init__(self, name: str, path: Path | None = None) -> None:
        self.name = name
        self.path = path if path is not None else library_path(name)
        self.lib = _load(self.path)
        # The caller owns the state memory. An array of 8-byte integers gives the alignment
        # the library needs (it checks).
        words = (int(self.lib.adcs_state_size()) + 7) // 8
        self._state = (ctypes.c_uint64 * words)()
        self._initialised = False
        self._out = AdcsOutputC()

    def reset(self, ctx: AdcsContext) -> None:
        cfg = config_to_c(ctx.config, ctx.dt, ctx.inertia, ctx.max_dipole)
        rc = self.lib.adcs_init(ctypes.addressof(self._state), ctypes.byref(cfg))
        if rc != 0:
            raise ValueError(f"{self.name}: adcs_init rejected the configuration (code {rc})")
        self._initialised = True

    def step(self, inp: AdcsInput) -> AdcsOutput:
        if not self._initialised:
            return AdcsOutput(status=int(AdcsStatus.NOT_INITIALISED))
        c_in = input_to_c(inp)
        rc = self.lib.adcs_step(
            ctypes.addressof(self._state), ctypes.byref(c_in), ctypes.byref(self._out)
        )
        if rc != 0:
            raise RuntimeError(f"{self.name}: adcs_step returned {rc}")
        return output_from_c(self._out)
