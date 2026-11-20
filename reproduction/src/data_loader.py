"""NASA C-MAPSS loading, preprocessing and the exact train/val/test split.

Pipeline (identical to the experiments that produced the manuscript tables):

1. Read train_FD00X.txt / test_FD00X.txt / RUL_FD00X.txt.
2. Piecewise-linear RUL cap at 125 cycles for both train and test labels.
3. Feature columns = all 3 operating settings + 21 sensors, minus columns
   whose training variance is ~0 (determined on the training split only).
4. Min-max normalise using training statistics only.
5. Sliding windows over every training engine: label = RUL at the window's
   last cycle.  85/15 random split of *windows* with random_state=42.
6. Test set = the final window of each test engine (one prediction per engine).

The exact window indices of step 5 are shipped under ``splits/`` so the split
can be audited without relying on the sklearn version.  The recipe above
recomputes them when no index file is supplied.
"""

import json
import os
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

COLUMN_NAMES = [
    "engine_id", "cycle",
    "op_setting_1", "op_setting_2", "op_setting_3",
    "sensor_1", "sensor_2", "sensor_3", "sensor_4", "sensor_5",
    "sensor_6", "sensor_7", "sensor_8", "sensor_9", "sensor_10",
    "sensor_11", "sensor_12", "sensor_13", "sensor_14", "sensor_15",
    "sensor_16", "sensor_17", "sensor_18", "sensor_19", "sensor_20",
    "sensor_21",
]


def get_feature_cols(df: pd.DataFrame) -> List[str]:
    """Operating settings + non-constant sensors (variance > 1e-6)."""
    cols = [c for c in COLUMN_NAMES if c not in ("engine_id", "cycle")]
    return [c for c in cols if df[c].std() > 1e-6]


def load_raw_data(data_dir: str, dataset_id: int = 1):
    """Return (train_df, test_df, rul_array) for FD00``dataset_id``."""
    sid = f"FD00{dataset_id}"
    train_df = pd.read_csv(os.path.join(data_dir, f"train_{sid}.txt"),
                           sep=r"\s+", header=None, names=COLUMN_NAMES)
    test_df = pd.read_csv(os.path.join(data_dir, f"test_{sid}.txt"),
                          sep=r"\s+", header=None, names=COLUMN_NAMES)
    rul_array = pd.read_csv(os.path.join(data_dir, f"RUL_{sid}.txt"),
                            sep=r"\s+", header=None).values.ravel()
    return train_df, test_df, rul_array


def compute_train_rul(df: pd.DataFrame, cap: int = 125) -> pd.DataFrame:
    """RUL = max_cycle - cycle, clipped to [0, cap] (piecewise-linear target)."""
    df = df.copy()
    max_cycles = df.groupby("engine_id")["cycle"].max()
    df["rul"] = (df["engine_id"].map(max_cycles) - df["cycle"]).clip(upper=cap)
    return df


def attach_test_rul(df: pd.DataFrame, rul_array: np.ndarray, cap: int = 125) -> pd.DataFrame:
    """Attach the official test RUL (engine ids start at 1) and cap it."""
    df = df.copy()
    rul_df = pd.DataFrame({"engine_id": range(1, len(rul_array) + 1),
                           "rul": rul_array})
    df = df.merge(rul_df, on="engine_id", how="left")
    df["rul"] = df["rul"].clip(upper=cap)
    return df


class Normalizer:
    """Min-max / z-score scaler fitted on training data only."""

    def __init__(self, method: str = "minmax"):
        self.method = method
        self.scalers: Dict[str, dict] = {}
        self.feature_cols: List[str] = []
        self.fitted = False

    def fit(self, train_df: pd.DataFrame, feature_cols: List[str]) -> "Normalizer":
        self.feature_cols = feature_cols
        for col in feature_cols:
            vals = train_df[col].values.astype(np.float32)
            if self.method == "minmax":
                self.scalers[col] = {"min": vals.min(), "max": vals.max()}
            else:
                self.scalers[col] = {"mean": vals.mean(), "std": vals.std() + 1e-8}
        self.fitted = True
        return self

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        if not self.fitted:
            raise RuntimeError("Normalizer must be fitted before transform().")
        out = np.zeros((len(df), len(self.feature_cols)), dtype=np.float32)
        for i, col in enumerate(self.feature_cols):
            s = self.scalers[col]
            vals = df[col].values.astype(np.float32)
            if self.method == "minmax":
                out[:, i] = (vals - s["min"]) / (s["max"] - s["min"] + 1e-8)
            else:
                out[:, i] = (vals - s["mean"]) / s["std"]
        return out


def create_sequences(df: pd.DataFrame, feature_cols: List[str],
                     window_size: int = 30, step: int = 1) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """One sample per (engine, end-cycle) sliding window; label = final-cycle RUL."""
    X_list, y_list, id_list = [], [], []
    for eid in df["engine_id"].unique():
        sub = df[df["engine_id"] == eid].sort_values("cycle").reset_index(drop=True)
        feats = sub[feature_cols].values.astype(np.float32)
        rul = sub["rul"].values.astype(np.float32)
        for start in range(0, len(sub) - window_size + 1, step):
            end = start + window_size
            X_list.append(feats[start:end])
            y_list.append(rul[end - 1])
            id_list.append(eid)
    return (np.stack(X_list, axis=0), np.array(y_list, dtype=np.float32),
            np.array(id_list, dtype=np.int32))


def _capped_table(data_dir: str, dataset_id: int, cap: int, norm_method: str):
    """Load, cap and normalise; return (train_norm, test_norm, feature_cols, scaler)."""
    train_df, test_df, rul_array = load_raw_data(data_dir, dataset_id)
    train_df = compute_train_rul(train_df, cap=cap)
    test_df = attach_test_rul(test_df, rul_array, cap=cap)
    feature_cols = get_feature_cols(train_df)
    normalizer = Normalizer(method=norm_method).fit(train_df, feature_cols)

    def to_frame(df):
        norm = pd.DataFrame(normalizer.transform(df), columns=feature_cols)
        norm["engine_id"] = df["engine_id"].values
        norm["cycle"] = df["cycle"].values
        norm["rul"] = df["rul"].values
        return norm

    return to_frame(train_df), to_frame(test_df), feature_cols, normalizer


def build_split_indices(data_dir: str, dataset_id: int, window_size: int,
                        cap: int = 125, norm_method: str = "minmax",
                        val_ratio: float = 0.15, split_seed: int = 42) -> dict:
    """Recompute and return the exact 85/15 window split indices.

    Returns a dict with ``train_idx`` and ``val_idx`` (int32 arrays), plus a
    small human-readable manifest.  These are the indices shipped in
    ``splits/FD00X_split.npz``.
    """
    train_norm, test_norm, feature_cols, _ = _capped_table(
        data_dir, dataset_id, cap, norm_method)
    X_all, y_all, _ = create_sequences(train_norm, feature_cols, window_size)
    tr_idx, va_idx = train_test_split(
        np.arange(len(X_all)), test_size=val_ratio, random_state=split_seed)
    manifest = {
        "dataset": f"FD00{dataset_id}",
        "window_size": window_size,
        "rul_cap": cap,
        "norm_method": norm_method,
        "split_rule": "sklearn.model_selection.train_test_split",
        "test_size": val_ratio,
        "random_state": split_seed,
        "n_windows_total": int(len(X_all)),
        "n_train_windows": int(len(tr_idx)),
        "n_val_windows": int(len(va_idx)),
        "n_test_engines": int(test_norm["engine_id"].nunique()),
        "n_features": int(len(feature_cols)),
        "feature_cols": feature_cols,
    }
    return {"train_idx": tr_idx.astype(np.int32),
            "val_idx": va_idx.astype(np.int32),
            "manifest": manifest}


def preprocess_pipeline(data_dir: str, dataset_id: int = 1, window_size: int = 30,
                        cap: int = 125, norm_method: str = "minmax",
                        val_ratio: float = 0.15, random_seed: int = 42,
                        split_indices: dict = None) -> dict:
    """End-to-end preprocessing.  Returns arrays used by train.py.

    ``split_indices`` may be the dict produced by :func:`build_split_indices`
    (or loaded from ``splits/FD00X_split.npz``) to guarantee bit-identical
    train/val partitions.
    """
    train_norm, test_norm, feature_cols, normalizer = _capped_table(
        data_dir, dataset_id, cap, norm_method)
    X_all, y_all, _ = create_sequences(train_norm, feature_cols, window_size)

    if split_indices is None:
        tr_idx, va_idx = train_test_split(
            np.arange(len(X_all)), test_size=val_ratio, random_state=random_seed)
    else:
        tr_idx = np.asarray(split_indices["train_idx"], dtype=np.int64)
        va_idx = np.asarray(split_indices["val_idx"], dtype=np.int64)
    X_train, y_train = X_all[tr_idx], y_all[tr_idx]
    X_val, y_val = X_all[va_idx], y_all[va_idx]

    # Test: last window of every engine, capped RUL at its final cycle.
    X_test_list, y_test_list = [], []
    for eid in test_norm["engine_id"].unique():
        sub = test_norm[test_norm["engine_id"] == eid].sort_values("cycle")
        feats = sub[feature_cols].values.astype(np.float32)
        rul = sub["rul"].values.astype(np.float32)
        if len(feats) < window_size:
            print(f"[preprocess] skip engine {eid}: {len(feats)} rows < window {window_size}")
            continue
        X_test_list.append(feats[-window_size:])
        y_test_list.append(rul[-1])
    X_test = np.stack(X_test_list, axis=0).astype(np.float32)
    y_test = np.array(y_test_list, dtype=np.float32)

    print(f"[preprocess] FD00{dataset_id}: train={len(X_train)} val={len(X_val)} "
          f"test={len(X_test)} | features={len(feature_cols)} | window={window_size} | cap={cap}")
    return {
        "X_train": X_train, "y_train": y_train,
        "X_val": X_val, "y_val": y_val,
        "X_test": X_test, "y_test": y_test,
        "normalizer": normalizer, "feature_cols": feature_cols,
        "cap": cap, "window_size": window_size, "n_features": len(feature_cols),
        "train_idx": tr_idx, "val_idx": va_idx,
    }


def save_split(path, split: dict) -> None:
    """Persist split indices + manifest as a compressed .npz file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path, train_idx=split["train_idx"], val_idx=split["val_idx"],
        manifest_json=json.dumps(split["manifest"], sort_keys=True))
    print(f"[split] wrote {path}")


def load_split(path) -> dict:
    """Load split indices written by :func:`save_split`."""
    with np.load(path, allow_pickle=False) as d:
        return {"train_idx": d["train_idx"], "val_idx": d["val_idx"],
                "manifest": json.loads(str(d["manifest_json"]))}
