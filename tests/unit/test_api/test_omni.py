"""
test_refactor_units.py
Adversarial unit tests for the Phase 1–4 refactor infrastructure.

Tests cover:
  - logger.get_logger
  - schema.validate_output_schema (happy path, naive timestamp, duplicates, all-NaN)
  - newell_coupling.compute_newell_numpy (known value, NaN propagation)
  - config constants (type contracts)
  - dB/dt NaN-safety
  - build_sequences.audit_leakage (clean pass, violation raise)

TODO: Update imports to src.swmi.*; expand coverage
"""

import sys
import os
import math
import unittest
import logging

import numpy as np
import pandas as pd

# Resolve src/ and scripts/ on the path
_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(_ROOT, "src"))
sys.path.insert(0, os.path.join(_ROOT, "scripts"))


# ---------------------------------------------------------------------------
# 1. Logger
# ---------------------------------------------------------------------------
class TestGetLogger(unittest.TestCase):
    def test_returns_logger_instance(self):
        from logger import get_logger
        log = get_logger("test_refactor")
        self.assertIsInstance(log, logging.Logger)
        self.assertEqual(log.name, "test_refactor")

    def test_second_call_returns_same_logger(self):
        from logger import get_logger
        a = get_logger("dup_test")
        b = get_logger("dup_test")
        self.assertIs(a, b)


# ---------------------------------------------------------------------------
# 2. Schema validator
# ---------------------------------------------------------------------------
def _utc_df(n=3):
    return pd.DataFrame({
        "timestamp": pd.date_range("2015-01-01", periods=n, freq="1min", tz="UTC"),
        "value": np.random.rand(n),
    })


class TestValidateOutputSchema(unittest.TestCase):
    def test_happy_path_passes(self):
        from schema import validate_output_schema
        validate_output_schema(_utc_df(), "TEST")   # must not raise

    def test_missing_timestamp_raises_key_error(self):
        from schema import validate_output_schema
        df = pd.DataFrame({"not_timestamp": [1, 2, 3]})
        with self.assertRaises(KeyError):
            validate_output_schema(df, "BAD")

    def test_naive_timestamp_raises_type_error(self):
        from schema import validate_output_schema
        df = pd.DataFrame({
            "timestamp": pd.date_range("2015-01-01", periods=3, freq="1min"),  # no tz
            "v": [1, 2, 3],
        })
        with self.assertRaises(TypeError):
            validate_output_schema(df, "NAIVE")

    def test_wrong_tz_raises_type_error(self):
        from schema import validate_output_schema
        df = pd.DataFrame({
            "timestamp": pd.date_range("2015-01-01", periods=3, freq="1min", tz="US/Eastern"),
            "v": [1, 2, 3],
        })
        with self.assertRaises(TypeError):
            validate_output_schema(df, "WRONG_TZ")

    def test_duplicate_timestamps_raise_value_error(self):
        from schema import validate_output_schema
        ts = pd.Timestamp("2015-01-01 00:00:00", tz="UTC")
        df = pd.DataFrame({
            "timestamp": [ts, ts, ts],
            "v": [1, 2, 3],
        })
        with self.assertRaises(ValueError):
            validate_output_schema(df, "DUPES")

    def test_all_nan_column_does_not_raise(self):
        """All-NaN column emits WARNING but must not raise."""
        from schema import validate_output_schema
        df = _utc_df(5)
        df["all_nan"] = float("nan")
        # Should log a warning, not raise
        validate_output_schema(df, "ALL_NAN")


# ---------------------------------------------------------------------------
# 3. Newell coupling
# ---------------------------------------------------------------------------
class TestNewellCoupling(unittest.TestCase):
    def test_known_value(self):
        """Verify formula against manual calculation for a well-known input."""
        from newell_coupling import compute_newell_numpy
        vsw   = np.array([400.0])
        by    = np.array([0.0])
        bz    = np.array([-5.0])   # southward; theta = pi, sin(pi/2)=1

        phi = compute_newell_numpy(vsw, by, bz)
        # Expected: 400^(4/3) * 5^(2/3) * 1^(8/3)
        expected = (400.0 ** (4/3)) * (5.0 ** (2/3)) * (1.0 ** (8/3))
        self.assertAlmostEqual(float(phi[0]), expected, places=3)

    def test_nan_propagation(self):
        """NaN input → NaN output (no crash, no sentinel fill)."""
        from newell_coupling import compute_newell_numpy
        phi = compute_newell_numpy(
            np.array([float("nan"), 400.0]),
            np.array([0.0, 0.0]),
            np.array([-5.0, -5.0]),
        )
        self.assertTrue(math.isnan(float(phi[0])))
        self.assertFalse(math.isnan(float(phi[1])))

    def test_zero_bt_gives_zero(self):
        """When By=Bz=0, Bt=0 → phi=0."""
        from newell_coupling import compute_newell_numpy
        phi = compute_newell_numpy(
            np.array([400.0]),
            np.array([0.0]),
            np.array([0.0]),
        )
        self.assertAlmostEqual(float(phi[0]), 0.0, places=6)

    def test_shape_mismatch_raises(self):
        from newell_coupling import compute_newell_numpy
        with self.assertRaises(ValueError):
            compute_newell_numpy(np.array([1.0, 2.0]), np.array([1.0]), np.array([1.0]))


# ---------------------------------------------------------------------------
# 4. Config type contracts
# ---------------------------------------------------------------------------
class TestConfigConstants(unittest.TestCase):
    def test_horizon_is_int(self):
        import config
        self.assertIsInstance(config.FORECAST_HORIZON_MIN, int)
        self.assertEqual(config.FORECAST_HORIZON_MIN, 60)

    def test_stations_is_list_of_strings(self):
        import config
        self.assertIsInstance(config.SUPERMAG_STATIONS, list)
        self.assertTrue(all(isinstance(s, str) for s in config.SUPERMAG_STATIONS))
        self.assertGreater(len(config.SUPERMAG_STATIONS), 0)

    def test_decay_halflife_is_positive_float(self):
        import config
        self.assertIsInstance(config.DECAY_HALFLIFE_MIN, float)
        self.assertGreater(config.DECAY_HALFLIFE_MIN, 0.0)

    def test_split_buffer_is_non_negative_int(self):
        import config
        self.assertIsInstance(config.SPLIT_BUFFER_DAYS, int)
        self.assertGreaterEqual(config.SPLIT_BUFFER_DAYS, 0)

    def test_scaler_version_is_string(self):
        import config
        self.assertIsInstance(config.SCALER_VERSION, str)


# ---------------------------------------------------------------------------
# 5. dB/dt NaN-safety (from retrieve_supermag logic)
# ---------------------------------------------------------------------------
class TestDbdtNanSafety(unittest.TestCase):
    """Verify backward-diff dB/dt never zero-fills missing values."""

    def _run_dbdt(self, b_n, b_e):
        # Reproduces the exact logic from retrieve_supermag._compute_dbdt
        b_n_s = pd.Series(b_n, dtype=float)
        b_e_s = pd.Series(b_e, dtype=float)
        dbn_dt = b_n_s.diff() / 60.0
        dbe_dt = b_e_s.diff() / 60.0

        dbdt_sq = np.where(
            dbn_dt.isna() | dbe_dt.isna(),
            float("nan"),
            dbn_dt.fillna(0) ** 2 + dbe_dt.fillna(0) ** 2,
        )
        dbdt_magnitude = pd.Series(np.sqrt(dbdt_sq))
        dbdt_magnitude[dbn_dt.isna() | dbe_dt.isna()] = float("nan")
        return dbn_dt, dbe_dt, dbdt_magnitude

    def test_first_row_is_nan(self):
        dbn, _, _ = self._run_dbdt([1.0, 2.0, 3.0], [0.0, 1.0, 2.0])
        self.assertTrue(math.isnan(float(dbn.iloc[0])))

    def test_nan_b_n_propagates_to_magnitude(self):
        _, _, mag = self._run_dbdt([float("nan"), 2.0], [0.0, 1.0])
        # row 1: diff of NaN and 2.0 → NaN; magnitude must be NaN
        self.assertTrue(math.isnan(float(mag.iloc[1])))

    def test_no_zero_fill_on_nan_input(self):
        """If b_n is NaN, dbdt_magnitude must not be 0.0 (zero is a real value)."""
        _, _, mag = self._run_dbdt([float("nan"), float("nan")], [1.0, 2.0])
        for v in mag:
            self.assertNotEqual(v, 0.0)   # must be NaN, not 0

    def test_clean_inputs_give_correct_value(self):
        dbn, _, mag = self._run_dbdt([0.0, 6.0], [0.0, 0.0])
        # dbn_dt at row 1 = (6-0)/60 = 0.1 nT/min; mag = 0.1
        self.assertAlmostEqual(float(dbn.iloc[1]), 0.1, places=6)
        self.assertAlmostEqual(float(mag.iloc[1]), 0.1, places=6)


# ---------------------------------------------------------------------------
# 6. Anti-leakage audit
# ---------------------------------------------------------------------------
class TestAuditLeakage(unittest.TestCase):
    def _make_ts(self, n, start="2015-01-01"):
        return pd.date_range(start, periods=n, freq="1min", tz="UTC")

    def test_clean_sequences_pass(self):
        from build_sequences import audit_leakage
        import config as cfg
        n = 20
        feat_ts = np.array(self._make_ts(n, "2015-06-01"))
        tgt_ts  = np.array(self._make_ts(n, "2015-06-01") + pd.Timedelta(hours=1))
        boundaries = {
            "train_end": pd.Timestamp("2021-12-31 23:59:00", tz="UTC"),
            "val_start": pd.Timestamp(cfg.VAL_START, tz="UTC"),
        }
        # Should not raise
        audit_leakage(feat_ts, tgt_ts, boundaries)

    def test_feature_after_target_raises(self):
        from build_sequences import audit_leakage
        n = 5
        # Features are 2 hours AFTER the targets — a hard leakage violation
        feat_ts = np.array(self._make_ts(n, "2015-06-01") + pd.Timedelta(hours=2))
        tgt_ts  = np.array(self._make_ts(n, "2015-06-01"))
        with self.assertRaises(AssertionError):
            audit_leakage(feat_ts, tgt_ts, {})


if __name__ == "__main__":
    unittest.main(verbosity=2)
