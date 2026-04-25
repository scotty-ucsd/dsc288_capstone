"""Model interfaces for the SWMI forecasting pipeline."""

from swmi.models.baseline_lstm import (
    MultiOutputBaseline,
    PerStationGradientBoostingBaseline,
    PersistenceBaseline,
    RidgeMultiOutputBaseline,
    SequenceDataset,
    load_sequence_npz,
    train_and_evaluate_baselines,
)

__all__ = [
    "MultiOutputBaseline",
    "PerStationGradientBoostingBaseline",
    "PersistenceBaseline",
    "RidgeMultiOutputBaseline",
    "SequenceDataset",
    "load_sequence_npz",
    "train_and_evaluate_baselines",
]
