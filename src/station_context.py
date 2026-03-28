"""
station_context.py
Utility functions to compute instantaneous magnetic local time (MLT) and
quasi-dipole latitude (QDLat) for a target SuperMAG station.

Used by build_sequences.py to append station-aware features to every
input sequence so the LSTM can learn the coupling between global
sector activity (leo_dayside / leo_nightside) and local disturbance.

Coordinate system
-----------------
- QDLat: quasi-dipole latitude (degrees).  Quasi-static; the value for
  a fixed geographic point changes slowly (~0.05 deg/year).  We use the
  value at the midpoint of the training period as a representative constant
  unless the caller passes an explicit reference time.
- MLT: magnetic local time (hours, 0–24).  Changes ~1 h per UT hour at
  fixed geographic longitude. Must be evaluated at the END of each
  120-minute input window (time T).

Dependencies
------------
Requires ``apexpy`` (already listed in pyproject.toml).

Fallback
--------
If ``apexpy`` is unavailable, approximate QDLat/MLT values are derived
from a small hard-coded lookup table (Scandinavia / Alaska / East Asia
stations used in the baseline study).

Physical rationale
------------------
An identical solar wind shock produces a much larger dB/dt spike at a
station on the nightside (substorm sector) than at the dayside. Providing
MLT sin/cos makes the sector relationship unambiguous to the LSTM.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from logger import get_logger

try:
    import apexpy
    _APEXPY_AVAILABLE = True
except ImportError:
    _APEXPY_AVAILABLE = False

log = get_logger(__name__)


# ---------------------------------------------------------------------------
# Hard-coded fallback geographic coordinates for baseline stations.
# Used when apexpy is unavailable.
# lat, lon in geographic degrees.
# ---------------------------------------------------------------------------
_STATION_GEO_COORDS: dict[str, tuple[float, float]] = {
    "ABK": (68.35, 18.82),   # Abisko, Sweden
    "TRO": (69.66, 18.94),   # Tromsø, Norway
    "BJN": (74.50, 19.20),   # Bjørnøya, Norway
    "YKC": (62.48, -114.48), # Yellowknife, Canada
    "MEA": (54.62, -113.35), # Meanook, Canada
    "KAK": (36.23, 140.19),  # Kakioka, Japan
}


def _get_apex(year: float) -> "apexpy.Apex":
    """Return an Apex object for the given decimal year (lazy init)."""
    return apexpy.Apex(date=year)


def get_station_qdlat(
    station_iaga: str,
    reference_year: float = 2018.5,
) -> float:
    """Return the quasi-dipole latitude (degrees) for a station.

    QDLat is treated as quasi-static: one value per station per study.
    Using the midpoint of the training period (2018.5 ≈ 2018-Jul) is
    appropriate for a study spanning 2015–2023.

    Parameters
    ----------
    station_iaga:
        IAGA station code (case-insensitive), e.g. ``"ABK"``.
    reference_year:
        Decimal year for the IGRF epoch used internally by apexpy.
        Defaults to the midpoint of the training period.

    Returns
    -------
    float
        QDLat in degrees (positive = north).

    Raises
    ------
    KeyError
        If the station is not found in either apexpy or the fallback table.
    """
    station = station_iaga.upper()

    if _APEXPY_AVAILABLE:
        if station not in _STATION_GEO_COORDS:
            raise KeyError(
                f"Station '{station}' not in fallback coordinate table. "
                "Add it to _STATION_GEO_COORDS in station_context.py."
            )
        glat, glon = _STATION_GEO_COORDS[station]
        apex = _get_apex(reference_year)
        qdlat, _ = apex.geo2qd(glat, glon, height=0.0)
        log.debug("QDLat(%s, year=%.1f) = %.3f via apexpy", station, reference_year, qdlat)
        return float(qdlat)
    else:
        log.warning(
            "apexpy not available; using approximate QDLat for %s. "
            "Install apexpy for accurate values.",
            station,
        )
        if station not in _STATION_GEO_COORDS:
            raise KeyError(
                f"Station '{station}' not in fallback coordinate table. "
                "Add it to _STATION_GEO_COORDS in station_context.py."
            )
        glat, glon = _STATION_GEO_COORDS[station]
        # Rough approximation: QDLat ≈ glat + 3 deg tilt correction for
        # northern Scandinavia/Alaska region (order-of-magnitude only).
        approx_qdlat = glat + 3.0
        log.debug("Approximate QDLat(%s) = %.1f (no apexpy)", station, approx_qdlat)
        return approx_qdlat


def get_station_mlt(
    station_iaga: str,
    timestamp: pd.Timestamp,
    reference_year: float = 2018.5,
) -> float:
    """Return the magnetic local time (hours, 0–24) for a station at a given time.

    Parameters
    ----------
    station_iaga:
        IAGA station code (case-insensitive), e.g. ``"ABK"``.
    timestamp:
        The UTC timestamp at which to evaluate MLT. Must be timezone-aware.
    reference_year:
        Decimal year passed to apexpy for internal IGRF model.

    Returns
    -------
    float
        MLT in hours, range [0, 24).

    Raises
    ------
    ValueError
        If ``timestamp`` is timezone-naive.
    """
    station = station_iaga.upper()

    if timestamp.tzinfo is None:
        raise ValueError(
            f"timestamp must be timezone-aware UTC, got naive {timestamp!r}."
        )

    if _APEXPY_AVAILABLE:
        if station not in _STATION_GEO_COORDS:
            raise KeyError(
                f"Station '{station}' not in _STATION_GEO_COORDS. "
                "Add geographic coordinates to station_context.py."
            )
        glat, glon = _STATION_GEO_COORDS[station]
        apex = _get_apex(reference_year)
        # apexpy.Apex.mlon2mlt converts magnetic longitude + time → MLT.
        # We first convert geo → apex magnetic longitude.
        _, mlon = apex.geo2apex(glat, glon, height=0.0)
        mlt = apex.mlon2mlt(mlon, timestamp)
        log.debug("MLT(%s, %s) = %.3f via apexpy", station, timestamp, mlt)
        return float(mlt % 24.0)
    else:
        log.warning(
            "apexpy not available; computing approximate MLT for %s. "
            "Install apexpy for accurate values.",
            station,
        )
        if station not in _STATION_GEO_COORDS:
            raise KeyError(f"Station '{station}' not in _STATION_GEO_COORDS.")
        _, glon = _STATION_GEO_COORDS[station]
        # Rough approximation: MLT ≈ UT + geographic_longitude / 15
        ut_hours = timestamp.hour + timestamp.minute / 60.0 + timestamp.second / 3600.0
        approx_mlt = (ut_hours + glon / 15.0) % 24.0
        log.debug("Approximate MLT(%s, %s) = %.2f (no apexpy)", station, timestamp, approx_mlt)
        return float(approx_mlt)


def station_mlt_encoded(
    station_iaga: str,
    timestamp: pd.Timestamp,
    reference_year: float = 2018.5,
) -> tuple[float, float]:
    """Return cyclical (sin, cos) encoding of MLT for a station at time T.

    Parameters
    ----------
    station_iaga:
        IAGA station code.
    timestamp:
        UTC timestamp (end of input window = time T).
    reference_year:
        Decimal year for apexpy IGRF epoch.

    Returns
    -------
    tuple[float, float]
        ``(station_mlt_sin, station_mlt_cos)`` — both in [-1, 1].
        Excludes these from Z-score scaling in Task 4.2.
    """
    mlt = get_station_mlt(station_iaga, timestamp, reference_year)
    angle = mlt * 2.0 * np.pi / 24.0
    return float(np.sin(angle)), float(np.cos(angle))
