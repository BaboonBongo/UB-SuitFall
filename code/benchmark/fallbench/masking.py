"""Mask source selection for participant- vs training-derived silhouettes."""
from __future__ import annotations

import copy
from pathlib import Path


def mode(cfg: dict) -> str:
    m = cfg.get("mask_source", "participant")
    if m not in ("participant", "training"):
        raise ValueError(f"mask_source must be 'participant' or 'training', not {m!r}")
    return m


def training_subjects(cfg: dict, test_subject: str) -> list[str]:
    return [s for s in sorted(cfg["subjects"]) if s != test_subject]


def source_tag(subjects: list[str]) -> str:
    return "+".join(sorted(subjects))


def fold_cfg(cfg: dict, test_subject: str) -> dict:
    """Return the config for the fold that tests `test_subject`."""
    if mode(cfg) == "participant":
        return cfg
    c = copy.copy(cfg)
    src = training_subjects(cfg, test_subject)
    c["features_dir"] = str(Path(cfg["features_dir"]) / f"mask_from_{source_tag(src)}")
    c["mask_from"] = src
    c["fold_test_subject"] = test_subject
    return c


def mask_png(cfg: dict, subject: str, camera: str) -> Path:
    """Path to the static mask used for this subject and camera."""
    root = Path(cfg["features_dir"]) / "static"
    if mode(cfg) == "participant":
        return root / f"{subject}_{camera}.png"
    return root / f"{source_tag(cfg['mask_from'])}_{camera}.png"


def camera_check_path(cfg: dict) -> Path:
    return Path(cfg["features_dir"]) / "camera_check.json"
