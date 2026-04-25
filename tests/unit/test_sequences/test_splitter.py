"""
Tests for msmf.validate.validate_phase_shift (stdlib unittest; no pytest required).

TODO: Update imports; add leakage tests
"""

import unittest

import numpy as np
import pandas as pd

from msmf.validate import validate_phase_shift


class TestValidatePhaseShift(unittest.TestCase):
    def test_clean_features_pass(self) -> None:
        rng = np.random.default_rng(0)
        n = 80
        df = pd.DataFrame(
            {
                "t": pd.date_range("2015-03-17", periods=n, freq="min"),
                "f1": rng.standard_normal(n),
                "f2": rng.standard_normal(n),
                "y": rng.standard_normal(n),
            }
        )
        out = validate_phase_shift(
            df,
            feature_cols=["f1", "f2"],
            target_col="y",
            prediction_horizon_steps=60,
            time_col="t",
        )
        self.assertTrue(out["valid"])
        self.assertEqual(out["violations"], [])
        self.assertTrue(out["time_order_ok"])

    def test_detects_target_shift_echo(self) -> None:
        n = 50
        df = pd.DataFrame(
            {
                "t": pd.date_range("2015-03-17", periods=n, freq="min"),
                "y": np.linspace(0, 1, n),
                "ok_feat": np.arange(n, dtype=float),
            }
        )
        # Row i: bad equals target at row i+1  =>  bad == y.shift(-1)
        df["bad"] = df["y"].shift(-1)
        out = validate_phase_shift(
            df,
            feature_cols=["ok_feat", "bad"],
            target_col="y",
            prediction_horizon_steps=10,
            max_future_row_lag=5,
            time_col="t",
        )
        self.assertFalse(out["valid"])
        self.assertTrue(any("bad" in v and "shifted forward" in v for v in out["violations"]))

    def test_detects_raw_shift_echo_via_auxiliary(self) -> None:
        n = 40
        s = np.arange(n, dtype=float)
        df = pd.DataFrame(
            {
                "t": pd.date_range("2015-03-17", periods=n, freq="min"),
                "raw": s,
                "feat": pd.Series(s).shift(-1),
                "y": np.roll(s, -5),
            }
        )
        out = validate_phase_shift(
            df,
            feature_cols=["feat"],
            target_col="y",
            prediction_horizon_steps=5,
            max_future_row_lag=3,
            auxiliary_raw_cols=["raw"],
            time_col="t",
        )
        self.assertFalse(out["valid"])

    def test_valid_lagged_feature_not_flagged(self) -> None:
        """Backward shift (+L) uses past rows only; should not match y.shift(-k)."""
        n = 60
        rng = np.random.default_rng(1)
        base = rng.standard_normal(n)
        df = pd.DataFrame(
            {
                "t": pd.date_range("2015-03-17", periods=n, freq="min"),
                "raw": base,
                "feat_lag": pd.Series(base).shift(3),
                "y": pd.Series(base).shift(-10),
            }
        )
        df = df.dropna()
        out = validate_phase_shift(
            df,
            feature_cols=["feat_lag"],
            target_col="y",
            prediction_horizon_steps=10,
            max_future_row_lag=8,
            auxiliary_raw_cols=["raw"],
            time_col="t",
        )
        self.assertTrue(out["valid"])

    def test_target_in_feature_cols_structural_violation(self) -> None:
        df = pd.DataFrame({"f": [1.0, 2.0], "y": [3.0, 4.0]})
        out = validate_phase_shift(
            df,
            feature_cols=["f", "y"],
            target_col="y",
            prediction_horizon_steps=1,
            require_increasing_time=False,
        )
        self.assertFalse(out["valid"])
        self.assertTrue(out["target_in_features"])

    def test_non_monotonic_time_raises(self) -> None:
        df = pd.DataFrame(
            {
                "t": [pd.Timestamp("2015-01-01"), pd.Timestamp("2014-12-31")],
                "f": [1.0, 2.0],
                "y": [1.0, 2.0],
            }
        )
        with self.assertRaises(ValueError):
            validate_phase_shift(
                df,
                feature_cols=["f"],
                target_col="y",
                prediction_horizon_steps=1,
                time_col="t",
            )


if __name__ == "__main__":
    unittest.main()
