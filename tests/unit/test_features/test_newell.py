"""Unit tests for Newell coupling (Phi_N) — compare to hand-computed values."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))

from swmi.features.newell_coupling import compute_newell_numpy  # noqa: E402


def test_newell_matches_manual_formula() -> None:
    vsw = 400.0
    by = 0.0
    bz = -5.0
    out = compute_newell_numpy(
        vsw=np.array([vsw]),
        by_gsm=np.array([by]),
        bz_gsm=np.array([bz]),
    )[0]
    bt = float(np.sqrt(by**2 + bz**2))
    theta = np.arctan2(by, bz)
    sin_term = float(np.abs(np.sin(theta / 2.0)) ** (8.0 / 3.0))
    expected = float(
        (abs(vsw) ** (4.0 / 3.0)) * (bt ** (2.0 / 3.0)) * (sin_term)
    )
    assert np.isclose(out, expected, rtol=1e-9)


def test_newell_propagates_nan() -> None:
    phi = compute_newell_numpy(
        vsw=np.array([400.0, np.nan]),
        by_gsm=np.array([0.0, 1.0]),
        bz_gsm=np.array([-5.0, -2.0]),
    )
    assert not np.isnan(phi[0])
    assert np.isnan(phi[1])
