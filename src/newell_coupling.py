"""
newell_coupling.py
NumPy-native implementation of the Newell coupling function (Phi_N).

Physical definition
-------------------
Phi_N = Vsw^(4/3) * Bt^(2/3) * |sin(theta_c / 2)|^(8/3)

where:
  Vsw      = solar wind speed (km/s)    — use |Vx| from OMNI
  Bt       = IMF transverse magnitude   = sqrt(By_GSM^2 + Bz_GSM^2)
  theta_c  = IMF clock angle            = arctan2(By_GSM, Bz_GSM)

Reference
---------
Newell et al. (2007), JGR, doi:10.1029/2006JA012015.

Units
-----
The output is in SI-proxy units [kV], proportional to the reconnection
rate at the dayside magnetopause. It is used in the feature matrix as a
composite driver index and in rolling 30-minute windows.

Scientific constraints
----------------------
- OMNI data is already propagated to the Earth's bow shock.
  NO additional time-shifting is applied here.
- NaN inputs propagate to NaN outputs (no sentinel fill).
- Returns a 1-D numpy array of the same length as the inputs.
"""

import numpy as np


def compute_newell_numpy(
    vsw: np.ndarray,
    by_gsm: np.ndarray,
    bz_gsm: np.ndarray,
) -> np.ndarray:
    """Compute the Newell coupling parameter element-wise.

    Parameters
    ----------
    vsw:
        Solar wind speed (km/s). Use the absolute value of Vx from OMNI.
        Shape: (N,)
    by_gsm:
        IMF By in GSM coordinates (nT). Shape: (N,)
    bz_gsm:
        IMF Bz in GSM coordinates (nT). Shape: (N,)

    Returns
    -------
    np.ndarray
        Newell coupling parameter, shape (N,).  NaN where any input is NaN.

    Raises
    ------
    ValueError
        If input arrays have different lengths.

    Examples
    --------
    >>> import numpy as np
    >>> phi = compute_newell_numpy(
    ...     vsw=np.array([400.0, np.nan]),
    ...     by_gsm=np.array([0.0, 3.0]),
    ...     bz_gsm=np.array([-5.0, -2.0]),
    ... )
    >>> np.isnan(phi[1])
    True
    """
    vsw = np.asarray(vsw, dtype=float)
    by_gsm = np.asarray(by_gsm, dtype=float)
    bz_gsm = np.asarray(bz_gsm, dtype=float)

    if not (vsw.shape == by_gsm.shape == bz_gsm.shape):
        raise ValueError(
            f"Input arrays must have the same shape. "
            f"Got vsw={vsw.shape}, by_gsm={by_gsm.shape}, bz_gsm={bz_gsm.shape}."
        )

    # IMF transverse magnitude and clock angle.
    bt = np.sqrt(by_gsm**2 + bz_gsm**2)   # nT; NaN-safe via np.sqrt
    theta_c = np.arctan2(by_gsm, bz_gsm)   # radians; NaN-safe

    # Half-clock-angle term: |sin(theta_c / 2)|^(8/3)
    sin_half = np.abs(np.sin(theta_c / 2.0))

    # Newell formula with fractional exponents.
    # np.power propagates NaN: x^p = NaN if x = NaN.
    phi = (
        np.power(np.abs(vsw), 4.0 / 3.0)
        * np.power(bt, 2.0 / 3.0)
        * np.power(sin_half, 8.0 / 3.0)
    )

    return phi
