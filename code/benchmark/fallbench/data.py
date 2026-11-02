"""Dataset index: one row per recording (subject x trial x camera).

Reads the cleaned fall_labels.csv, the per-frame labels from
`fall_labels.py export` and the frame folders. Never writes to the dataset.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd


def camera_label(folder: str) -> str:
    """cctv-a / Cam_A -> CCTV-A (same rule as fall_labels.py)."""
    n = folder.lower().replace("_", "").replace("-", "")
    return {"a": "CCTV-A", "b": "CCTV-B"}.get(n[-1:], folder.upper())


def list_frames(folder) -> list[Path]:
    folder = Path(folder)
    if not folder.is_dir():
        return []
    files = [p for p in folder.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"} and p.stem.isdigit()]
    return sorted(files, key=lambda p: int(p.stem))


def build_index(cfg: dict) -> pd.DataFrame:
    rows = []
    for subject, root in cfg["subjects"].items():
        root = Path(root)
        cams = {camera_label(d.name): d for d in (root / "Frames").iterdir() if d.is_dir()} \
            if (root / "Frames").is_dir() else {}
        labels = pd.read_csv(root / "fall_labels.csv")
        for r in labels.itertuples(index=False):
            rid = f"{r.trial_uid}_{r.camera}"
            frames_dir = cams.get(r.camera, root / "Frames" / "?") / f"Activity{r.activity_id}Trial{r.trial_id}"
            label_file = root / "frame_labels" / f"{rid}.csv"
            n = int(r.n_frames or 0)
            issue = ("no frames (missing or corrupted video)" if n <= 0 else
                     f"frame folder not found: {frames_dir}" if not frames_dir.is_dir() else
                     f"{label_file.name} not found - run fall_labels.py export" if not label_file.exists() else "")
            rows.append(dict(subject_id=subject, trial_uid=r.trial_uid, camera=r.camera, recording_id=rid,
                             activity_id=r.activity_id, activity_name=r.activity_name, category=r.category,
                             n_frames=n, fall_start=r.fall_start_frame, impact=r.impact_frame,
                             fall_end=r.fall_end_frame, frames_dir=str(frames_dir),
                             label_file=str(label_file), usable=issue == "", issue=issue))
    return pd.DataFrame(rows)


def load_recording(cfg: dict, rec) -> pd.DataFrame:
    """Per-frame labels joined with the per-frame silhouette features."""
    labels = pd.read_csv(rec.label_file)
    feats = Path(cfg["features_dir"]) / rec.subject_id / f"{rec.recording_id}.csv"
    if not feats.exists():
        raise FileNotFoundError(f"{feats} missing - run `python run.py extract` first")
    df = labels.merge(pd.read_csv(feats), on="frame", how="left")
    df["present"] = df["present"].fillna(0.0)
    return df
