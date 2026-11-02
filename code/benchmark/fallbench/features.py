"""Window features for the fall-detection models."""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

from .data import build_index, load_recording
from .silhouette import FEATURES

SHAPE = ["cx", "cy", "top", "bottom", "width", "height", "log_aspect", "area", "extent",
         "angle", "elongation", "major", "minor"]
VELOCITY = ["cy", "top", "bottom", "height", "log_aspect", "angle", "area"]
ACCEL = ["cy", "top", "angle"]


def frame_signals(df: pd.DataFrame, fps: float, blank_rate: float = 0.0, rng=None) -> pd.DataFrame:
    df = df.copy()
    if blank_rate:
        df.loc[(rng or np.random.default_rng(0)).random(len(df)) < blank_rate, "present"] = 0.0
    df.loc[df["present"] < 0.5, FEATURES] = np.nan
    df["log_aspect"] = np.log(df["aspect"].clip(lower=1e-3))
    sig = df[SHAPE].ffill()
    for c in VELOCITY:
        sig[f"v_{c}"] = sig[c].diff() * fps
    for c in ACCEL:
        sig[f"a_{c}"] = sig[f"v_{c}"].diff() * fps
    sig["present"] = df["present"].to_numpy()
    return sig


def summarise(seq: np.ndarray, names: list[str], feature_set: str) -> tuple[np.ndarray, list[str]]:
    """Convert windowed sequences into summary features."""
    col = {n: seq[:, :, i] for i, n in enumerate(names)}
    stats = {"mean": np.nanmean, "min": np.nanmin, "max": np.nanmax}
    feats = {f"{c}_{k}": fn(col[c], axis=1) for c in SHAPE for k, fn in stats.items()}
    feats["missing_fraction"] = 1 - col["present"].mean(axis=1)
    if feature_set == "shape_motion":
        for c in SHAPE:
            feats[f"{c}_std"] = np.nanstd(col[c], axis=1)
            feats[f"{c}_change"] = col[c][:, -1] - col[c][:, 0]
        for c in VELOCITY:
            v = col[f"v_{c}"]
            feats.update({f"v_{c}_max": np.nanmax(v, 1), f"v_{c}_min": np.nanmin(v, 1),
                          f"v_{c}_absmean": np.nanmean(np.abs(v), 1)})
        for c in ACCEL:
            feats[f"a_{c}_absmax"] = np.nanmax(np.abs(col[f"a_{c}"]), 1)
        h = np.nanmax(col["height"], 1)
        h = np.where(h > 0, h, np.nan)
        feats["top_drop_rel"] = (col["top"][:, -1] - col["top"][:, 0]) / h
        feats["cy_drop_rel"] = (col["cy"][:, -1] - col["cy"][:, 0]) / h
        feats["height_last_rel"] = col["height"][:, -1] / h
        feats["v_top_max_rel"] = np.nanmax(col["v_top"], 1) / h
    X = np.stack(list(feats.values()), axis=1).astype(np.float32)
    return np.where(np.isfinite(X), X, np.nan), list(feats)


def build_windows(cfg: dict, subjects: list[str], index: pd.DataFrame | None = None,
                  blank_rate: float = 0.0) -> dict:
    """Build all windows for the given subjects."""
    w = cfg["windows"]
    W, S, fset = w["length"], w["stride"], w["features"]
    index = build_index(cfg) if index is None else index
    rng = np.random.default_rng(cfg["seed"])
    metas, seqs = [], []
    for rec in index[index["usable"] & index["subject_id"].isin(subjects)].itertuples():
        df = load_recording(cfg, rec)
        sig = frame_signals(df, cfg["fps"], blank_rate, rng)
        if len(df) < W:
            continue
        idx = np.arange(0, len(df) - W + 1, S)[:, None] + np.arange(W)
        frac = df["label"].to_numpy()[idx].mean(axis=1)
        frames = df["frame"].to_numpy()
        metas.append(pd.DataFrame(dict(
            recording_id=rec.recording_id, subject_id=rec.subject_id, trial_uid=rec.trial_uid,
            camera=rec.camera, activity_name=rec.activity_name, category=rec.category,
            start_frame=frames[idx[:, 0]], end_frame=frames[idx[:, -1]], fall_fraction=frac,
            label=(frac >= w["min_fall_fraction"]).astype(int))))
        seqs.append(sig.to_numpy(np.float32)[idx])
    names = list(sig.columns)
    seq = np.concatenate(seqs)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        X_sum, sum_names = summarise(seq, names, fset)
    keep = [i for i, n in enumerate(names) if fset == "shape_motion" or n in SHAPE or n == "present"]
    return dict(meta=pd.concat(metas, ignore_index=True), X_sum=X_sum, sum_names=sum_names,
                X_seq=seq[:, :, keep], seq_names=[names[i] for i in keep])
