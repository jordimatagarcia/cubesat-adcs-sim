"""The launcher's GUI-free core (adcssim.launch), shared with the command line."""

import os
import time
from pathlib import Path

import numpy as np
import pytest

from adcssim.backends.config import VARIANTS
from adcssim.launch import (
    SECTIONS,
    AdcsField,
    RunCancelledError,
    RunRequest,
    adcs_fields,
    apply_overrides,
    backend_status,
    format_adcs_value,
    list_scenarios,
    parse_adcs_value,
    prepare,
    recent_logs,
    run_batch,
)
from adcssim.scenario import ScenarioError, load_scenario
from adcssim.telemetry import Telemetry
from tests.conftest import REPO

CASE_1A = REPO / "scenarios" / "case_1a.toml"


@pytest.fixture(scope="module")
def case_1a():  # type: ignore[no-untyped-def]
    return load_scenario(CASE_1A)


# ---- what can be chosen ----------------------------------------------------------------


def test_list_scenarios_reads_every_file() -> None:
    infos = {i.path.name: i for i in list_scenarios()}
    assert set(infos) == {p.name for p in (REPO / "scenarios").glob("*.toml")}
    case = infos["case_1a.toml"]
    assert case.error is None
    assert case.name == "case_1a"
    assert case.description.startswith("Thesis case 1a")
    assert case.duration_s == 3000.0
    assert (case.backend, case.variant) == ("python-ref", "thesis")


def test_list_scenarios_reports_broken_files(tmp_path: Path) -> None:
    (tmp_path / "broken.toml").write_text("name = ", encoding="utf-8")
    (tmp_path / "notes.txt").write_text("ignored", encoding="utf-8")
    [info] = list_scenarios(tmp_path)
    assert info.name == "broken" and info.error


def test_backend_status_lists_every_backend_in_order() -> None:
    status = backend_status()
    assert [s.name for s in status] == ["python-ref", "c", "rust", "none"]
    by_name = {s.name: s for s in status}
    assert by_name["python-ref"].available and by_name["none"].available


@pytest.mark.parametrize("backend", ["c", "rust"])
def test_backend_status_explains_a_missing_library(
    backend: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    env = {"c": "ADCSSIM_C_LIB", "rust": "ADCSSIM_RUST_LIB"}[backend]
    monkeypatch.setenv(env, str(tmp_path / "missing.dll"))
    status = {s.name: s for s in backend_status()}[backend]
    assert not status.available
    assert "missing.dll" in status.reason and "Build it with" in status.reason

    library = tmp_path / "present.dll"
    library.write_bytes(b"")
    monkeypatch.setenv(env, str(library))
    assert {s.name: s for s in backend_status()}[backend].available


# ---- ADCS settings ---------------------------------------------------------------------


def test_adcs_fields_describe_the_configuration() -> None:
    fields = {f.name: f for f in adcs_fields()}
    assert "variant" not in fields
    assert fields["ukf_alpha"].kind == "float" and fields["ukf_alpha"].default == 1.0
    assert fields["ukf_alpha"].section == "Two-stage UKF (thesis)"
    assert fields["p_x0"].kind == "vector" and len(fields["p_x0"].default) == 13
    assert fields["fault_steps"].kind == "int"
    assert fields["control_enabled"].kind == "bool"
    assert fields["flae_weight_sun"].section == "FLAE"
    assert fields["sigma_gyro_noise"].section == "Error-state filters (improved, MEKF)"
    assert fields["kp"].section == "Control"
    assert "rad/s" in fields["sigma_gyro_noise"].doc  # attribute docstring
    assert {f.section for f in fields.values()} == set(SECTIONS)


@pytest.mark.parametrize(
    ("kind", "text", "value"),
    [
        ("float", " 2.5e-3 ", 2.5e-3),
        ("int", "4", 4),
        ("bool", "True", True),
        ("bool", "off", False),
    ],
)
def test_parse_adcs_value(kind: str, text: str, value: object) -> None:
    f = AdcsField("x", kind, None, "Control")
    assert parse_adcs_value(f, text) == value


def test_parse_and_format_vectors_round_trip() -> None:
    f = AdcsField("p_x0", "vector", None, "Two-stage UKF (thesis)")
    v = parse_adcs_value(f, "1e-4, 1e-2 1e-7,5e-7")
    np.testing.assert_array_equal(v, [1e-4, 1e-2, 1e-7, 5e-7])
    np.testing.assert_array_equal(parse_adcs_value(f, format_adcs_value(v)), v)
    assert format_adcs_value(True) == "true" and format_adcs_value(0.5) == "0.5"


@pytest.mark.parametrize(
    ("kind", "text"), [("float", "abc"), ("int", "1.5"), ("bool", "maybe"), ("vector", "")]
)
def test_parse_adcs_value_names_the_setting_on_error(kind: str, text: str) -> None:
    with pytest.raises(ValueError, match="my_setting"):
        parse_adcs_value(AdcsField("my_setting", kind, None, "Control"), text)


# ---- overrides -------------------------------------------------------------------------


def test_apply_overrides_replaces_only_what_is_given(case_1a) -> None:  # type: ignore[no-untyped-def]
    assert apply_overrides(case_1a) is case_1a
    out = apply_overrides(
        case_1a,
        backend="none",
        variant="mekf",
        duration_s=12.0,
        seed=7,
        adcs={"ukf_alpha": 0.5, "control_enabled": True},
    )
    assert (out.backend, out.adcs.variant, out.duration_s, out.seed) == ("none", "mekf", 12.0, 7)
    assert out.adcs.ukf_alpha == 0.5 and out.adcs.control_enabled
    assert out.adcs.kp == case_1a.adcs.kp  # untouched settings keep the scenario's values
    assert case_1a.adcs.variant == "thesis"  # the loaded scenario is not modified


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"backend": "quantum"}, "unknown ADCS backend"),
        ({"variant": "kalman"}, "unknown ADCS variant"),
        ({"duration_s": 0.0}, "duration"),
        ({"duration_s": float("nan")}, "duration"),
        ({"seed": -1}, "seed"),
        ({"adcs": {"no_such_setting": 1}}, "no_such_setting"),
        ({"adcs": {"ukf_alpha": -1.0}}, "ukf_alpha"),
        ({"adcs": {"ukf_alpha": 1e-6}}, "invalid ADCS configuration"),
    ],
)
def test_apply_overrides_rejects_invalid_values(case_1a, kwargs, message) -> None:  # type: ignore[no-untyped-def]
    with pytest.raises(ValueError, match=message):
        apply_overrides(case_1a, **kwargs)


def test_open_loop_skips_the_adcs_rules(case_1a) -> None:  # type: ignore[no-untyped-def]
    # with no ADCS the settings are never used, so they are not checked against the rules
    out = apply_overrides(case_1a, backend="none", adcs={"ukf_alpha": 1e-6})
    assert out.adcs.ukf_alpha == 1e-6


def test_prepare_loads_and_overrides() -> None:
    sc = prepare(RunRequest(CASE_1A, variant="improved", seed=4))
    assert (sc.name, sc.adcs.variant, sc.seed) == ("case_1a", "improved", 4)
    with pytest.raises(ScenarioError, match=r"missing.toml"):
        prepare(RunRequest(REPO / "missing.toml"))


def test_command_line_reproduces_the_request() -> None:
    req = RunRequest(CASE_1A, backend="rust", variant="mekf", duration_s=60.0, seed=2)
    assert req.command_line() == [
        "adcssim", "run", str(CASE_1A),
        "--backend", "rust", "--variant", "mekf", "--duration", "60", "--seed", "2",
    ]  # fmt: skip
    assert RunRequest(CASE_1A).command_line("view") == ["adcssim", "view", str(CASE_1A)]


# ---- batch runs ------------------------------------------------------------------------


def test_run_batch_writes_a_log(tmp_path: Path) -> None:
    sc = prepare(RunRequest(CASE_1A, variant="mekf", duration_s=2.0, seed=5))
    calls: list[tuple[int, int]] = []
    result = run_batch(sc, tmp_path / "out", lambda k, n: calls.append((k, n)))
    assert result.log.parent == tmp_path / "out" and result.log.is_file()
    assert calls[-1] == (20, 20)
    meta = Telemetry.read(result.log).metadata
    assert (meta["backend"], meta["adcs_variant"], meta["seed"]) == ("python-ref", "mekf", "5")
    assert float(meta["duration_s"]) == 2.0
    summary = "\n".join(result.summary())
    assert "case_1a (python-ref)" in summary and str(result.log) in summary


def test_the_variant_changes_the_estimate(tmp_path: Path) -> None:
    errors = {}
    for variant in VARIANTS:
        sc = prepare(RunRequest(CASE_1A, variant=variant, duration_s=5.0))
        errors[variant] = run_batch(sc, tmp_path).telemetry.get("attitude_error")
    assert not np.array_equal(errors["thesis"], errors["improved"], equal_nan=True)
    assert not np.array_equal(errors["improved"], errors["mekf"], equal_nan=True)


def test_run_batch_can_be_cancelled(tmp_path: Path) -> None:
    sc = prepare(RunRequest(CASE_1A, duration_s=60.0))
    seen: list[int] = []
    with pytest.raises(RunCancelledError):
        run_batch(sc, tmp_path, lambda k, n: seen.append(k), should_cancel=lambda: len(seen) >= 3)
    assert len(seen) == 3
    assert not list(tmp_path.glob("*.parquet"))


def test_recent_logs_newest_first(tmp_path: Path) -> None:
    assert recent_logs(tmp_path / "none") == []
    now = time.time()
    for i, name in enumerate(["a", "b", "c"]):
        p = tmp_path / f"{name}.parquet"
        p.write_bytes(b"")
        os.utime(p, (now + i, now + i))
    assert [p.stem for p in recent_logs(tmp_path)] == ["c", "b", "a"]
    assert [p.stem for p in recent_logs(tmp_path, limit=1)] == ["c"]
