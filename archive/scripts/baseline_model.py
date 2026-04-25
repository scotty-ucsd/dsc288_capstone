"""
baseline_model.py

M0/M1/M2 baseline models for ABK March 2015 MVP dataset.

Refactored to use src/logger and src/config for paths.
ML model logic is unchanged per refactor specifications.
"""

import os
import sys
import json
import numpy as np
import pandas as pd

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import config
from logger import get_logger

log = get_logger(__name__)


# Paths use config base dirs; MVP sequences keep the legacy path structure
_SEQ_BASE = os.path.join("data", "processed", "sequences", "2015", "03")
SEQ_INDEX_PATH   = os.path.join(_SEQ_BASE, "sequence_index_abk_201503.parquet")
X_PATH           = os.path.join(_SEQ_BASE, "X_abk_201503.npy")
Y_PATH           = os.path.join(_SEQ_BASE, "y_abk_201503.npy")
FEATURE_LIST_PATH = os.path.join(_SEQ_BASE, "feature_columns_abk_201503.txt")

OUT_DIR           = os.path.join("data", "results", "2015", "03")
OUT_METRICS       = os.path.join(OUT_DIR, "baseline_metrics_abk_201503.csv")
OUT_PREDS         = os.path.join(OUT_DIR, "baseline_predictions_abk_201503.parquet")
OUT_IMPORTANCE    = os.path.join(OUT_DIR, "m2_feature_importance_proxy_abk_201503.csv")
OUT_CONFIG        = os.path.join(OUT_DIR, "baseline_run_config_abk_201503.json")
OUT_SPLIT_DIAGNOSTICS = os.path.join(OUT_DIR, "split_diagnostics_abk_201503.csv")


def load_inputs():
    if not os.path.exists(SEQ_INDEX_PATH):
        raise FileNotFoundError(f"Missing sequence index: {SEQ_INDEX_PATH}")
    if not os.path.exists(X_PATH):
        raise FileNotFoundError(f"Missing X array: {X_PATH}")
    if not os.path.exists(Y_PATH):
        raise FileNotFoundError(f"Missing y array: {Y_PATH}")
    if not os.path.exists(FEATURE_LIST_PATH):
        raise FileNotFoundError(f"Missing feature list: {FEATURE_LIST_PATH}")

    seq_index = pd.read_parquet(SEQ_INDEX_PATH)
    X = np.load(X_PATH)
    y = np.load(Y_PATH)

    with open(FEATURE_LIST_PATH, "r", encoding="utf-8") as f:
        feature_cols = [line.strip() for line in f if line.strip()]

    seq_index["start_timestamp"] = pd.to_datetime(seq_index["start_timestamp"], utc=True)
    seq_index["end_timestamp"] = pd.to_datetime(seq_index["end_timestamp"], utc=True)
    seq_index["target_timestamp"] = pd.to_datetime(seq_index["target_timestamp"], utc=True)

    if len(seq_index) != len(X) or len(seq_index) != len(y):
        raise ValueError("Sequence index, X, and y lengths do not match.")

    return seq_index, X, y, feature_cols


def build_splits(seq_index):
    n = len(seq_index)
    train_end = int(n * 0.70)
    val_end = int(n * 0.85)

    split = np.array(["test"] * n, dtype=object)
    split[:train_end] = "train"
    split[train_end:val_end] = "val"

    seq_index = seq_index.copy()
    seq_index["split"] = split
    return seq_index


def extract_last_timestep_df(X, feature_cols):
    return pd.DataFrame(X[:, -1, :], columns=feature_cols)


def rmse(y_true, y_pred):
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def event_metrics(y_true, y_pred, threshold):
    obs = y_true >= threshold
    pred = y_pred >= threshold

    tp = int(np.sum(obs & pred))
    tn = int(np.sum((~obs) & (~pred)))
    fp = int(np.sum((~obs) & pred))
    fn = int(np.sum(obs & (~pred)))

    pod = tp / (tp + fn) if (tp + fn) > 0 else np.nan
    far = fp / (tp + fp) if (tp + fp) > 0 else np.nan

    denom = ((tp + fn) * (fn + tn) + (tp + fp) * (fp + tn))
    hss = (2 * (tp * tn - fn * fp) / denom) if denom > 0 else np.nan

    return {
        "threshold": threshold,
        "count_obs_events": int(obs.sum()),
        "count_pred_events": int(pred.sum()),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "pod": pod,
        "far": far,
        "hss": hss,
    }


def regression_metrics(y_true, y_pred):
    out = {
        "rmse": rmse(y_true, y_pred),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "r2": float(r2_score(y_true, y_pred)),
    }

    p95 = np.nanpercentile(y_true, 95.0)
    tail_mask = y_true >= p95
    out["tail_threshold_p95"] = float(p95)
    out["tail_count"] = int(tail_mask.sum())
    out["tail_rmse_p95"] = rmse(y_true[tail_mask], y_pred[tail_mask]) if tail_mask.sum() > 0 else np.nan

    for thr in [2.0, 5.0, 10.0]:
        ev = event_metrics(y_true, y_pred, thr)
        out[f"events_obs_ge_{thr:g}"] = ev["count_obs_events"]
        out[f"events_pred_ge_{thr:g}"] = ev["count_pred_events"]
        out[f"pod_ge_{thr:g}"] = ev["pod"]
        out[f"far_ge_{thr:g}"] = ev["far"]
        out[f"hss_ge_{thr:g}"] = ev["hss"]

    return out


def compute_persistence_predictions(X, feature_cols):
    last_df = extract_last_timestep_df(X, feature_cols)

    if "abk_dbdt_mag_t0" not in last_df.columns:
        raise ValueError(
            "Proper M0 persistence requires 'abk_dbdt_mag_t0' in the last timestep features."
        )

    return last_df["abk_dbdt_mag_t0"].to_numpy(dtype=np.float32), "abk_dbdt_mag_t0"


def build_m1_features(X, feature_cols):
    last_df = extract_last_timestep_df(X, feature_cols)

    desired = [
        "ut_hour_sin",
        "ut_hour_cos",
        "month_sin",
        "month_cos",
        "omni_sym_h",
        "omni_al",
        "omni_au",
        "leo_index_global",
        "decay_age_min",
        "leo_is_fresh",
        "leo_count",
        "goes_bz_gsm",
        "omni_bz_gsm",
        "omni_vx",
    ]

    use_cols = [c for c in desired if c in last_df.columns]
    if not use_cols:
        raise ValueError("No M1 feature columns available.")

    return last_df[use_cols].copy(), use_cols


def build_m2_features(X, feature_cols):
    last_df = extract_last_timestep_df(X, feature_cols)
    return last_df.copy(), list(last_df.columns)


def evaluate_model(name, split_name, y_true, y_pred, persistence_rmse_test=None):
    row = {"model": name, "split": split_name}
    row.update(regression_metrics(y_true, y_pred))
    if persistence_rmse_test is not None and split_name == "test":
        row["rmse_skill_vs_m0"] = 1.0 - (row["rmse"] / persistence_rmse_test if persistence_rmse_test > 0 else np.nan)
    else:
        row["rmse_skill_vs_m0"] = np.nan
    return row


def fit_and_predict_log_linear(X_train, y_train, X_val, X_test):
    y_train_log = np.log1p(np.clip(y_train, a_min=0.0, a_max=None))
    model = LinearRegression()
    model.fit(X_train, y_train_log)

    train_pred = np.expm1(model.predict(X_train))
    val_pred = np.expm1(model.predict(X_val))
    test_pred = np.expm1(model.predict(X_test))

    train_pred = np.clip(train_pred, a_min=0.0, a_max=None)
    val_pred = np.clip(val_pred, a_min=0.0, a_max=None)
    test_pred = np.clip(test_pred, a_min=0.0, a_max=None)

    return model, train_pred, val_pred, test_pred


def fit_and_select_hgbr(X_train, y_train, X_val, y_val, X_test):
    candidates = [
        {"max_depth": 3, "learning_rate": 0.05, "max_iter": 300, "min_samples_leaf": 20},
        {"max_depth": 5, "learning_rate": 0.05, "max_iter": 300, "min_samples_leaf": 20},
        {"max_depth": 5, "learning_rate": 0.03, "max_iter": 500, "min_samples_leaf": 30},
    ]

    best_model = None
    best_val_pred = None
    best_test_pred = None
    best_cfg = None
    best_rmse = np.inf

    for cfg in candidates:
        model = HistGradientBoostingRegressor(
            loss="squared_error",
            random_state=42,
            early_stopping=True,
            validation_fraction=None,
            **cfg,
        )
        model.fit(X_train, y_train)
        val_pred = model.predict(X_val)
        val_pred = np.clip(val_pred, a_min=0.0, a_max=None)
        val_rmse = rmse(y_val, val_pred)

        if val_rmse < best_rmse:
            best_rmse = val_rmse
            best_model = model
            best_val_pred = val_pred
            best_test_pred = np.clip(model.predict(X_test), a_min=0.0, a_max=None)
            best_cfg = cfg

    return best_model, best_cfg, best_val_pred, best_test_pred


def permutation_importance_proxy(model, X_val, y_val, feature_names, top_n=25):
    baseline = rmse(y_val, np.clip(model.predict(X_val), a_min=0.0, a_max=None))
    records = []

    rng = np.random.default_rng(42)
    X_work = X_val.copy()

    for j, col in enumerate(feature_names):
        original = X_work[:, j].copy()
        rng.shuffle(X_work[:, j])
        score = rmse(y_val, np.clip(model.predict(X_work), a_min=0.0, a_max=None))
        X_work[:, j] = original

        records.append({
            "feature": col,
            "rmse_increase": float(score - baseline),
        })

    imp = pd.DataFrame(records).sort_values("rmse_increase", ascending=False).head(top_n)
    return imp


def build_split_diagnostics(seq_index, X, y, feature_cols):
    last_df = extract_last_timestep_df(X, feature_cols)
    diag_df = seq_index[["split", "start_timestamp", "end_timestamp", "target_timestamp"]].copy()
    diag_df["y_true"] = y

    if "omni_sym_h" in last_df.columns:
        diag_df["omni_sym_h_last"] = last_df["omni_sym_h"].to_numpy(dtype=np.float32)
    else:
        diag_df["omni_sym_h_last"] = np.nan

    rows = []
    for split_name, grp in diag_df.groupby("split", sort=False):
        yvals = grp["y_true"].to_numpy(dtype=float)
        symh = grp["omni_sym_h_last"].to_numpy(dtype=float)

        row = {
            "split": split_name,
            "n_sequences": int(len(grp)),
            "start_min": grp["start_timestamp"].min(),
            "start_max": grp["start_timestamp"].max(),
            "target_min": grp["target_timestamp"].min(),
            "target_max": grp["target_timestamp"].max(),
            "y_mean": float(np.nanmean(yvals)),
            "y_median": float(np.nanmedian(yvals)),
            "y_p90": float(np.nanpercentile(yvals, 90)),
            "y_p95": float(np.nanpercentile(yvals, 95)),
            "y_p99": float(np.nanpercentile(yvals, 99)),
            "count_y_ge_0_1": int(np.sum(yvals >= 0.1)),
            "count_y_ge_0_2": int(np.sum(yvals >= 0.2)),
            "count_y_ge_0_5": int(np.sum(yvals >= 0.5)),
            "count_y_ge_1_0": int(np.sum(yvals >= 1.0)),
            "count_y_ge_2_0": int(np.sum(yvals >= 2.0)),
            "symh_mean": float(np.nanmean(symh)),
            "symh_median": float(np.nanmedian(symh)),
            "symh_min": float(np.nanmin(symh)),
            "symh_p05": float(np.nanpercentile(symh, 5)),
            "count_symh_le_-30": int(np.sum(symh <= -30)),
            "count_symh_le_-50": int(np.sum(symh <= -50)),
            "count_symh_le_-100": int(np.sum(symh <= -100)),
            "count_quiet_symh_gt_-30": int(np.sum(symh > -30)),
        }
        rows.append(row)

    return pd.DataFrame(rows)


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    log.info("Loading inputs...")
    seq_index, X, y, feature_cols = load_inputs()
    seq_index = build_splits(seq_index)

    split = seq_index["split"].to_numpy()
    train_mask = split == "train"
    val_mask = split == "val"
    test_mask = split == "test"

    log.info("Train sequences: %d", int(train_mask.sum()))
    log.info("Val sequences:   %d", int(val_mask.sum()))
    log.info("Test sequences:  %d", int(test_mask.sum()))

    log.info("Computing split diagnostics...")
    split_diag_df = build_split_diagnostics(seq_index, X, y, feature_cols)
    split_diag_df.to_csv(OUT_SPLIT_DIAGNOSTICS, index=False)

    preds_df = seq_index[[
        "sequence_id",
        "start_timestamp",
        "end_timestamp",
        "target_timestamp",
        "target_abk_dbdt_mag_tplus60",
        "split",
    ]].copy()
    preds_df["y_true"] = y

    last_df = extract_last_timestep_df(X, feature_cols)
    if "omni_sym_h" in last_df.columns:
        preds_df["omni_sym_h_last"] = last_df["omni_sym_h"].to_numpy(dtype=np.float32)

    metrics_rows = []

    log.info("Running M0 persistence baseline...")
    m0_pred_all, m0_source = compute_persistence_predictions(X, feature_cols)
    m0_pred_all = np.clip(m0_pred_all, a_min=0.0, a_max=None)
    preds_df["m0_pred"] = m0_pred_all

    m0_test_rmse = rmse(y[test_mask], m0_pred_all[test_mask])

    for split_name, mask in [("train", train_mask), ("val", val_mask), ("test", test_mask)]:
        metrics_rows.append(
            evaluate_model("M0_persistence", split_name, y[mask], m0_pred_all[mask], persistence_rmse_test=m0_test_rmse)
        )

    log.info("Running M1 log-linear context baseline...")
    X_m1_df, m1_cols = build_m1_features(X, feature_cols)

    X_m1_train = X_m1_df.loc[train_mask].to_numpy(dtype=np.float32)
    y_train = y[train_mask]
    X_m1_val = X_m1_df.loc[val_mask].to_numpy(dtype=np.float32)
    y_val = y[val_mask]
    X_m1_test = X_m1_df.loc[test_mask].to_numpy(dtype=np.float32)

    m1_model, m1_train_pred, m1_val_pred, m1_test_pred = fit_and_predict_log_linear(
        X_m1_train, y_train, X_m1_val, X_m1_test
    )

    preds_df.loc[train_mask, "m1_pred"] = m1_train_pred
    preds_df.loc[val_mask, "m1_pred"] = m1_val_pred
    preds_df.loc[test_mask, "m1_pred"] = m1_test_pred

    for split_name, mask in [("train", train_mask), ("val", val_mask), ("test", test_mask)]:
        metrics_rows.append(
            evaluate_model("M1_log_linear_context", split_name, y[mask], preds_df.loc[mask, "m1_pred"].to_numpy(), persistence_rmse_test=m0_test_rmse)
        )

    log.info("Running M2 gradient-boosted tree baseline...")
    X_m2_df, m2_cols = build_m2_features(X, feature_cols)

    X_m2_train = X_m2_df.loc[train_mask].to_numpy(dtype=np.float32)
    X_m2_val = X_m2_df.loc[val_mask].to_numpy(dtype=np.float32)
    X_m2_test = X_m2_df.loc[test_mask].to_numpy(dtype=np.float32)

    m2_model, m2_cfg, m2_val_pred, m2_test_pred = fit_and_select_hgbr(
        X_m2_train, y_train, X_m2_val, y_val, X_m2_test
    )

    m2_train_pred = np.clip(m2_model.predict(X_m2_train), a_min=0.0, a_max=None)

    preds_df.loc[train_mask, "m2_pred"] = m2_train_pred
    preds_df.loc[val_mask, "m2_pred"] = m2_val_pred
    preds_df.loc[test_mask, "m2_pred"] = m2_test_pred

    for split_name, mask in [("train", train_mask), ("val", val_mask), ("test", test_mask)]:
        metrics_rows.append(
            evaluate_model("M2_hgbr_last_timestep", split_name, y[mask], preds_df.loc[mask, "m2_pred"].to_numpy(), persistence_rmse_test=m0_test_rmse)
        )

    log.info("Computing M2 permutation-importance proxy...")
    imp_df = permutation_importance_proxy(
        model=m2_model,
        X_val=X_m2_val.copy(),
        y_val=y_val,
        feature_names=m2_cols,
        top_n=25,
    )

    metrics_df = pd.DataFrame(metrics_rows).sort_values(["model", "split"]).reset_index(drop=True)
    metrics_df.to_csv(OUT_METRICS, index=False)
    preds_df.to_parquet(OUT_PREDS, index=False)
    imp_df.to_csv(OUT_IMPORTANCE, index=False)

    run_config = {
        "m0_source": m0_source,
        "m1_model": "LinearRegression on log1p(target), inverse transformed with expm1 and clipped at zero",
        "m1_features": m1_cols,
        "m2_candidate_selection_metric": "validation_rmse",
        "m2_selected_config": m2_cfg,
        "split_strategy": "chronological_70_15_15_within_month",
        "thresholds": [2.0, 5.0, 10.0],
        "split_diagnostics_symh_thresholds": [-30, -50, -100],
    }
    with open(OUT_CONFIG, "w", encoding="utf-8") as f:
        json.dump(run_config, f, indent=2)

    log.info("Saved metrics       → %s", OUT_METRICS)
    log.info("Saved predictions   → %s", OUT_PREDS)
    log.info("Saved M2 importance → %s", OUT_IMPORTANCE)
    log.info("Saved run config    → %s", OUT_CONFIG)
    log.info("Saved split diag    → %s", OUT_SPLIT_DIAGNOSTICS)

    log.info("\nSplit diagnostics:\n%s", split_diag_df.to_string(index=False))
    log.info("\nMetrics summary:\n%s", metrics_df.to_string(index=False))
    log.info("\nTop M2 validation importance:\n%s", imp_df.to_string(index=False))


if __name__ == "__main__":
    main()
