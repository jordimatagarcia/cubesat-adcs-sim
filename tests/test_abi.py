"""The ctypes mirror must match adcs.h exactly: compile a C program that prints the real
layout and compare every size and offset."""

import ctypes
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest

from adcssim.backends import AdcsConfig, AdcsInput
from adcssim.backends.abi import (
    ABI_VERSION,
    STRUCTS,
    AdcsOutputC,
    config_to_c,
    input_to_c,
    output_from_c,
)
from tests.conftest import REPO

HEADER = REPO / "adcs-c" / "include" / "adcs.h"
CC = shutil.which("cc") or shutil.which("gcc") or shutil.which("clang")


def _layout_program() -> str:
    lines = ["#include <stdio.h>", "#include <stddef.h>", '#include "adcs.h"', "int main(void) {"]
    for cname, struct in STRUCTS.items():
        lines.append(f'  printf("{cname} sizeof %zu\\n", sizeof({cname}));')
        for field, *_ in struct._fields_:
            lines.append(f'  printf("{cname} {field} %zu\\n", offsetof({cname}, {field}));')
    lines.append('  printf("ABI %u\\n", ADCS_ABI_VERSION);')
    lines += ["  return 0;", "}"]
    return "\n".join(lines)


@pytest.fixture(scope="module")
def c_layout(tmp_path_factory: pytest.TempPathFactory) -> dict[str, int]:
    if CC is None:
        pytest.skip("no C compiler")
    d: Path = tmp_path_factory.mktemp("abi")
    (d / "layout.c").write_text(_layout_program(), encoding="utf-8")
    exe = d / "layout"
    subprocess.run(
        [CC, "-std=c11", "-I", str(HEADER.parent), str(d / "layout.c"), "-o", str(exe)],
        check=True,
    )
    out = subprocess.run([str(exe)], check=True, capture_output=True, text=True).stdout
    layout = {}
    for line in out.splitlines():
        *key, value = line.split()
        layout[" ".join(key)] = int(value)
    return layout


def test_abi_version_matches_header(c_layout: dict[str, int]) -> None:
    assert c_layout["ABI"] == ABI_VERSION


@pytest.mark.parametrize("cname", list(STRUCTS))
def test_struct_layout_matches_header(c_layout: dict[str, int], cname: str) -> None:
    struct = STRUCTS[cname]
    assert ctypes.sizeof(struct) == c_layout[f"{cname} sizeof"]
    for field, *_ in struct._fields_:
        assert getattr(struct, field).offset == c_layout[f"{cname} {field}"], field


def test_conversions_round_trip() -> None:
    cfg = AdcsConfig(control_enabled=True, kp=3e-5)
    c_cfg = config_to_c(cfg, 0.1, np.diag([1.0, 2.0, 3.0]), np.array([0.1, 0.2, 0.3]))
    assert c_cfg.control_enabled and c_cfg.kp == 3e-5 and c_cfg.inertia[4] == 2.0
    assert list(c_cfg.v)[:4] == [1e-4] * 4

    inp = AdcsInput(
        t=1.5,
        mag_b=np.array([1e-5, 2e-5, 3e-5]),
        mag_valid=True,
        gyro_b=np.array([0.1, 0.2, 0.3]),
        gyro_valid=False,
        sun_b=np.array([1.0, 0.0, 0.0]),
        sun_valid=True,
        sun_ref_i=np.array([0.0, 1.0, 0.0]),
        mag_ref_i=np.array([4e-5, 5e-5, 6e-5]),
    )
    c_in = input_to_c(inp)
    assert c_in.t == 1.5 and not c_in.gyro_valid and list(c_in.mag_ref_i) == [4e-5, 5e-5, 6e-5]

    out_c = AdcsOutputC()
    out_c.q_est[3] = 1.0
    out_c.mode = 2
    out_c.status = 0x12
    out = output_from_c(out_c)
    np.testing.assert_array_equal(out.q_est, [0.0, 0.0, 0.0, 1.0])
    assert out.mode == 2 and out.status == 0x12 and out.p_diag.shape == (19,)
