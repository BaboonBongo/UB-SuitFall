"""Silhouette extraction and static-mask generation."""
from __future__ import annotations

import json
import math
import multiprocessing as mp
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from . import masking
from .data import build_index, list_frames

FEATURES = ["cx", "cy", "top", "bottom", "width", "height", "aspect", "area", "extent",
            "angle", "elongation", "major", "minor"]
EMPTY = {"present": 0.0, **{k: np.nan for k in FEATURES}}


def read(path, width: int) -> np.ndarray | None:
    img = cv2.imread(str(path))
    if img is not None and img.shape[1] != width:
        img = cv2.resize(img, (width, round(img.shape[0] * width / img.shape[1])), interpolation=cv2.INTER_AREA)
    return img


def suit_mask(img: np.ndarray, ranges) -> np.ndarray:
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    return np.bitwise_or.reduce([cv2.inRange(hsv, np.array(lo, np.uint8), np.array(hi, np.uint8))
                                 for lo, hi in ranges])


def static_green(videos: list[list[Path]], s: dict, info: dict | None = None) -> np.ndarray:
    """Estimate the static green background mask from sampled frames."""
    masks, sizes = [], set()
    for frames in videos:
        for i in np.unique(np.linspace(0, len(frames) - 1, s["static_samples_per_video"]).round().astype(int)):
            img = read(frames[i], s["width"])
            if img is not None:
                masks.append(suit_mask(img, s["suit_hsv"]) > 0)
                sizes.add(img.shape[:2])
    shape = {m.shape for m in masks}
    if info is not None:
        info.update(recordings=len(videos), sampled_frames=len(masks),
                    frame_sizes=sorted(f"{w}x{h}" for h, w in sizes), consistent_size=len(shape) == 1)
    if len(shape) != 1:
        return np.zeros((1, 1), np.uint8)
    static = (np.mean(masks, axis=0) >= s["static_min_fraction"]).astype(np.uint8) * 255
    return cv2.dilate(static, np.ones((11, 11), np.uint8))


def shape_features(mask: np.ndarray) -> dict:
    H, W = mask.shape
    ys, xs = np.nonzero(mask)
    w, h = np.ptp(xs) + 1.0, np.ptp(ys) + 1.0
    m = cv2.moments(mask, binaryImage=True)
    a, b, c = m["mu20"] / m["m00"], m["mu02"] / m["m00"], m["mu11"] / m["m00"]
    root = math.sqrt(4 * c * c + (a - b) ** 2)
    l1, l2 = max((a + b + root) / 2, 1e-9), max((a + b - root) / 2, 1e-9)
    theta = 0.5 * math.degrees(math.atan2(2 * c, a - b))
    return dict(cx=xs.mean() / W, cy=ys.mean() / H, top=ys.min() / H, bottom=ys.max() / H,
                width=w / H, height=h / H, aspect=h / w, area=len(xs) / (W * H), extent=len(xs) / (w * h),
                angle=90 - abs(theta), elongation=math.sqrt(l1 / l2), major=4 * math.sqrt(l1) / H, minor=4 * math.sqrt(l2) / H)


class Tracker:
    """Track the suit silhouette frame by frame."""

    def __init__(self, static: np.ndarray | None, s: dict):
        self.s, self.static, self.prev = s, static, None
        self.merge = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (s["merge_px"],) * 2)

    def __call__(self, img: np.ndarray) -> tuple[dict, np.ndarray | None]:
        H, W = img.shape[:2]
        green = suit_mask(img, self.s["suit_hsv"])
        if self.static is not None and self.static.shape == green.shape:
            green[self.static > 0] = 0
        green = cv2.morphologyEx(green, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        n, labels, stats, cents = cv2.connectedComponentsWithStats(cv2.dilate(green, self.merge))
        min_px, best, best_key = self.s["min_area"] * W * H, None, None
        for k in range(1, n):
            if stats[k, cv2.CC_STAT_AREA] < min_px:
                continue
            pixels = np.count_nonzero(green[labels == k])
            near = self.prev is None or math.dist(cents[k], self.prev) < 0.25 * math.hypot(W, H)
            if pixels >= min_px and (best_key is None or (near, pixels) > best_key):
                best, best_key = k, (near, pixels)
        if best is None:
            return dict(EMPTY), None
        person = ((labels == best) & (green > 0)).astype(np.uint8)
        f = shape_features(person)
        self.prev = (f["cx"] * W, f["cy"] * H)
        return {"present": 1.0, **f}, person


def _overlay(img, person, f, frame):
    img = img.copy()
    if person is not None:
        img[person > 0] = np.minimum(0.5 * img[person > 0] + (127, 0, 127), 255).astype(np.uint8)
        ys, xs = np.nonzero(person)
        cv2.rectangle(img, (xs.min(), ys.min()), (xs.max(), ys.max()), (0, 255, 255), 2)
    text = f"f{frame} angle {f['angle']:.0f}" if person is not None else f"f{frame} NO SILHOUETTE"
    cv2.putText(img, text, (6, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
    return img


def _extract(job):
    rec, static_png, s, out_csv, qa_jpg = job
    cv2.setNumThreads(1)
    frames = list_frames(rec["frames_dir"])
    track = Tracker(cv2.imread(static_png, cv2.IMREAD_GRAYSCALE), s)
    if rec["category"] == "FALL" and rec["fall_start"] == rec["fall_start"]:
        fs, fi, fe = int(rec["fall_start"]), int(rec["impact"]), int(rec["fall_end"])
        qa = {max(1, fs - 10), fs, (fs + fi) // 2, fi, fe, min(len(frames), fe + 10)}
    else:
        qa = set(np.linspace(1, len(frames), 6).round().astype(int).tolist())
    rows, tiles = [], []
    for path in frames:
        img = read(path, s["width"])
        f, person = track(img) if img is not None else (dict(EMPTY), None)
        rows.append({"frame": int(path.stem), **f})
        if img is not None and int(path.stem) in qa:
            tiles.append(_overlay(img, person, f, int(path.stem)))
    df = pd.DataFrame(rows)
    df.to_csv(out_csv, index=False)
    if tiles:
        tiles += [np.zeros_like(tiles[0])] * (-len(tiles) % 3)
        cv2.imwrite(qa_jpg, np.vstack([np.hstack(tiles[i:i + 3]) for i in range(0, len(tiles), 3)]))
    return rec["recording_id"], float(df["present"].mean())


def _frame_size(frames_dir, width: int) -> str | None:
    frames = list_frames(frames_dir)
    img = read(frames[0], width) if frames else None
    return None if img is None else f"{img.shape[1]}x{img.shape[0]}"


def _run_jobs(jobs: list, root: Path) -> None:
    print(f"extracting {len(jobs)} recording(s) -> {root}")
    with ProcessPoolExecutor(os.cpu_count(), mp_context=mp.get_context("spawn")) as pool:
        for rid, present in pool.map(_extract, jobs):
            print(f"  {rid}: silhouette in {100 * present:.0f}% of frames")


def extract(cfg: dict, force: bool = False, accept_camera_shift: bool = False) -> None:
    if masking.mode(cfg) == "training":
        return _extract_training_masks(cfg, force, accept_camera_shift)
    s, root = cfg["silhouette"], Path(cfg["features_dir"])
    (root / "qa").mkdir(parents=True, exist_ok=True)
    (root / "static").mkdir(exist_ok=True)
    index = build_index(cfg)
    index = index[index["usable"]]
    jobs = []
    for (subj, cam), grp in index.groupby(["subject_id", "camera"]):
        png = masking.mask_png(cfg, subj, cam)
        if force or not png.exists():
            mask = static_green([list_frames(d) for d in grp["frames_dir"]], s)
            cv2.imwrite(str(png), mask)
            print(f"static green {subj} {cam}: {100 * (mask > 0).mean():.2f}% of pixels")
        for rec in grp.to_dict("records"):
            out_csv = root / subj / f"{rec['recording_id']}.csv"
            out_csv.parent.mkdir(exist_ok=True)
            if force or not out_csv.exists():
                jobs.append((rec, str(png), s, str(out_csv), str(root / "qa" / f"{rec['recording_id']}.jpg")))
    print(f"({len(index) - len(jobs)} recording(s) already done)")
    _run_jobs(jobs, root)
    print(f"QA images: {root / 'qa'} - check the magenta silhouette is the suited subject")


def _check_cameras(cfg: dict, accept_camera_shift: bool) -> None:
    """Check that the transfer mask is valid before using it."""
    path = masking.camera_check_path(cfg)
    if not path.exists():
        raise SystemExit(f"{path} not found. Run tools/mask_diagnostic.py first: it verifies that the cameras of both participants have the same frame size and position.")
    check = json.loads(path.read_text())
    bad = [c for c, v in check["cameras"].items() if not v["frame_size_match"]]
    if bad:
        raise SystemExit(f"Frame sizes differ between participants for {bad}; a mask cannot be transferred. Stop and check the camera correspondence (see camera_check.json).")
    moved = [c for c, v in check["cameras"].items() if v["max_abs_shift_px"] > check["max_shift_px"]]
    if moved and not accept_camera_shift:
        raise SystemExit(f"The scene appears shifted between participants for {moved} (> {check['max_shift_px']} px, see camera_check.json and the overlay images). Confirm that it is the same camera installation, then re-run with --accept-camera-shift.")


def _extract_training_masks(cfg: dict, force: bool, accept_camera_shift: bool) -> None:
    """Generate fold-specific masks from training subjects only."""
    _check_cameras(cfg, accept_camera_shift)
    s = cfg["silhouette"]
    index = build_index(cfg)
    index = index[index["usable"]]
    log, per_rec = [], []
    for test_subject in sorted(cfg["subjects"]):
        fcfg = masking.fold_cfg(cfg, test_subject)
        src, root = fcfg["mask_from"], Path(fcfg["features_dir"])
        (root / "qa").mkdir(parents=True, exist_ok=True)
        (root / "static").mkdir(exist_ok=True)
        info = {"test_subject": test_subject, "mask_from": src}
        for (subj, cam), grp in index.groupby(["subject_id", "camera"]):
            if subj == test_subject:
                continue
            png = masking.mask_png(fcfg, subj, cam)
            if force or not png.exists():
                mask = static_green([list_frames(d) for d in grp["frames_dir"]], s, info)
                cv2.imwrite(str(png), mask)
                log.append({
                    "fold": f"test_{test_subject}", "camera": cam, "mask_source": ",".join(src),
                    "test_subject": test_subject, "source_recordings": ",".join(sorted(grp.recording_id)),
                    "sampled_frames": info.get("sampled_frames", 0), "mask_size": list(mask.shape),
                    "consistent_size": bool(info.get("consistent_size", False)), "mask_path": str(png)
                })
            for rec in grp.to_dict("records"):
                out_csv = root / subj / f"{rec['recording_id']}.csv"
                out_csv.parent.mkdir(exist_ok=True)
                if force or not out_csv.exists():
                    per_rec.append((rec, str(png), s, str(out_csv), str(root / "qa" / f"{rec['recording_id']}.jpg")))
    if log:
        pd.DataFrame(log).to_csv(Path(cfg["features_dir"]) / "mask_log.csv", index=False)
    _run_jobs(per_rec, Path(cfg["features_dir"]))
