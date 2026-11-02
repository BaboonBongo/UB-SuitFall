"""Where the static green-background mask comes from.

Two modes (config key `mask_source`):

participant (original, transductive)
    One mask per participant and camera, estimated from all of that
    participant's recordings. Features live in <features_dir>/<subject>/.
    When a participant is the test subject, its own unlabelled frames
    contributed to the mask used on it.

training (corrected, inductive)
    For each leave-one-subject-out fold, one mask per camera estimated only
    from the TRAINING participant(s) of that fold, and applied to the
    recordings of both the training and the test participant. Each fold has
    its own feature folder, <features_dir>/mask_from_<train subjects>/, so a
    mask or feature file is never reused across folds.

The masking algorithm itself (4 evenly spaced frames per recording, suit-green
in >= 50% of sampled frames, 11x11 dilation, mask skipped when the frame size
differs) is identical in both modes; only the source recordings change.
"""
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
    """Config whose features_dir holds the features to use in the fold that tests `test_subject`.

    In participant mode this is the original shared feature folder; in training
    mode it is the fold's own folder, built from training-participant masks only.
    """
    if mode(cfg) == "participant":
        return cfg
    c = copy.copy(cfg)
    src = training_subjects(cfg, test_subject)
    c["features_dir"] = str(Path(cfg["features_dir"]) / f"mask_from_{source_tag(src)}")
    c["mask_from"] = src
    c["fold_test_subject"] = test_subject
    return c


def mask_png(cfg: dict, subject: str, camera: str) -> Path:
    """Path of the static mask applied to `subject`'s recordings of `camera` under `cfg`.

    `cfg` must be the fold config (fold_cfg) in training mode.
    """
    root = Path(cfg["features_dir"]) / "static"
    if mode(cfg) == "participant":
        return root / f"{subject}_{camera}.png"
    return root / f"{source_tag(cfg['mask_from'])}_{camera}.png"


def camera_check_path(cfg: dict) -> Path:
    return Path(cfg["features_dir"]) / "camera_check.json"
