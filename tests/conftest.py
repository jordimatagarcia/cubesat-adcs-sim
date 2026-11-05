import json
from pathlib import Path
from typing import Any

import pytest

DATA = Path(__file__).parent / "data"
REPO = Path(__file__).parent.parent


def load_fixture(name: str) -> Any:
    return json.loads((DATA / name).read_text(encoding="utf-8"))


@pytest.fixture(scope="session")
def astropy_ref() -> list[dict[str, Any]]:
    """Reference values computed with astropy (see the file's ``source`` field)."""
    points: list[dict[str, Any]] = load_fixture("astropy_reference.json")["points"]
    return points


@pytest.fixture(scope="session")
def igrf_ref() -> list[dict[str, Any]]:
    """Reference field values computed with ppigrf (see the file's ``source`` field)."""
    points: list[dict[str, Any]] = load_fixture("igrf_ppigrf_reference.json")["points"]
    return points


@pytest.fixture(scope="session")
def aalto1_tle_path() -> Path:
    return REPO / "data" / "tle" / "aalto-1_2021-287.tle"
