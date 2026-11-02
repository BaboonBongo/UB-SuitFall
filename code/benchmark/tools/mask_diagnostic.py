#!/usr/bin/env python3
"""Diagnose the original static masks before the corrected run."""

from __future__ import annotations

import argparse
import json
import math
import multiprocessing as mp
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from fallbench.config import load_config, md_table  # noqa: E402
from fallbench.data import build_index, list_frames  # noqa: E402
from fallbench.masking import camera_check_path  # noqa: E402
from fallbench.silhouette import Tracker, read, static_green  # noqa: E402

# Thresholds for reporting frames that differ materially; they are descriptive only.
CENTROID_PX = 5.0
REL_CHANGE = 0.10


def sampled(frames: list[Path], s: dict) -> list[np.ndarray]:
    idx = np.unique(np.linspace(0, len(frames) - 1, s["static_samples_per_video"]).round().astype(int))
    return [img for img in (read(frames[i], s["width"]) for i in idx) if img is not None]


def raw_size(frames_dir) -> str:
    f = list_frames(frames_dir)
    img = cv2.imread(str(f[0])) if f else None
    return "none" if img is None else f"{img.shape[1]}x{img.shape[0]}"


def phase_shift(a: np.ndarray, b: np.ndarray) -> tuple[float, float, float]:
    ga = cv2.cvtColor(a, cv2.COLOR_BGR2GRAY).astype(np.float32)
    gb = cv2.cvtColor(b, cv2.COLOR_BGR2GRAY).astype(np.float32)
    win = cv2.createHanningWindow(ga.shape[::-1], cv2.CV_32F)
    (dx, dy), response = cv2.phaseCorrelate(ga, gb, win)
    return float(dx), float(dy), float(response)


def overlay(bg: np.ndarray, m1: np.ndarray, m2: np.ndarray) -> np.ndarray:
    img = (0.6 * bg).astype(np.uint8)
    only1, only2, both = m1 & ~m2, m2 & ~m1, m1 & m2
    img[both] = (0, 220, 220)
    img[only1] = (0, 0, 255)
    img[only2] = (255, 80, 0)
    return img


def label(img: np.ndarray, text: str) -> np.ndarray:
    img = img.copy()
    cv2.putText(img, text, (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 3)
    cv2.putText(img, text, (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)
    return img


def _two_masks(job):
    rec, own_png, other_png, s = job
    cv2.setNumThreads(1)
    own, other = (cv2.imread(p, cv2.IMREAD_GRAYSCALE) for p in (own_png, other_png))
    t_own, t_other = Tracker(own, s), Tracker(other, s)
    rows = []
    for path in list_frames(rec["frames_dir"]):
        img = read(path, s["width"])
        if img is None:
            continue
        H, W = img.shape[:2]
        f1, _ = t_own(img)
        f2, _ = t_other(img)
        rows.append(dict(frame=int(path.stem), W=W, H=H,
                         **{f"{k}_own": f1.get(k, np.nan) for k in ("present", "area", "width", "height", "cx", "cy")},
                         **{f"{k}_other": f2.get(k, np.nan) for k in ("present", "area", "width", "height", "cx", "cy")} ))
    d = pd.DataFrame(rows)
    both = (d.present_own == 1) & (d.present_other == 1)
    b = d[both]
    rel = lambda k: (b[f"{k}_other"] - b[f"{k}_own"]).abs() / b[f"{k}_own"].abs().clip(lower=1e-9)  # noqa: E731
    shift = np.hypot((b.cx_other - b.cx_own) * b.W, (b.cy_other - b.cy_own) * b.H)
    material = (d.present_own != d.present_other).sum() + int(
        ((shift > CENTROID_PX) | (rel("area") > REL_CHANGE) | (rel("width") > REL_CHANGE)
         | (rel("height") > REL_CHANGE)).sum())
    identical = int(((b[[c for c in d.columns if c.endswith("_own")]].to_numpy()
                      == b[[c.replace("_own", "_other") for c in d.columns if c.endswith("_own")]].to_numpy())
                     .all(axis=1)).sum())
    return dict(recording_id=rec["recording_id"], subject_id=rec["subject_id"], camera=rec["camera"],
                frames=len(d), present_own_pct=100 * d.present_own.mean(), present_other_pct=100 * d.present_other.mean(),
                presence_differs=int((d.present_own != d.present_other).sum()),
                identical_frames=identical,
                area_rel_change_median=float(rel("area").median()) if len(b) else np.nan,
                area_rel_change_p95=float(rel("area").quantile(0.95)) if len(b) else np.nan,
                width_rel_change_p95=float(rel("width").quantile(0.95)) if len(b) else np.nan,
                height_rel_change_p95=float(rel("height").quantile(0.95)) if len(b) else np.nan,
                centroid_shift_px_median=float(shift.median()) if len(b) else np.nan,
                centroid_shift_px_p95=float(shift.quantile(0.95)) if len(b) else np.nan,
                centroid_shift_px_max=float(shift.max()) if len(b) else np.nan,
                materially_different_frames=int(material))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    root = Path(__file__).resolve().parents[1]
    ap.add_argument("--config", default=root / "config.yaml", help="ORIGINAL run config")
    ap.add_argument("--corrected", default=root / "config_foldmask.yaml", help="corrected run config")
    ap.add_argument("--out", default=None, help="default: <corrected output_dir>/../diagnostics")
    ap.add_argument("--workers", type=int, default=os.cpu_count())
    ap.add_argument("--skip-silhouettes", action="store_true", help="masks and camera check only")
    a = ap.parse_args()
    cfg, corr = load_config(a.config), load_config(a.corrected)
    s = cfg["silhouette"]
    out = Path(a.out) if a.out else Path(corr["output_dir"]).parent / "diagnostics"
    (out / "masks").mkdir(parents=True, exist_ok=True)
    index = build_index(cfg)
    index = index[index.usable]
    subjects = sorted(cfg["subjects"])
    if len(subjects) != 2:
        raise SystemExit("this diagnostic compares exactly two subjects")
    s1, s2 = subjects
    max_shift = float(corr.get("max_camera_shift_px", 3))

    mask_rows, cam_rows, cam_check = [], [], {}
    masks = {}
    for cam in sorted(index.camera.unique()):
        med, info = {}, {}
        for subj in subjects:
            grp = index[(index.subject_id == subj) & (index.camera == cam)]
            vids = [list_frames(d) for d in grp.frames_dir]
            i = {}
            m = static_green(vids, s, i) > 0
            saved = Path(cfg["features_dir"]) / "static" / f"{subj}_{cam}.png"
            saved_m = cv2.imread(str(saved), cv2.IMREAD_GRAYSCALE) if saved.exists() else None
            agree = (None if saved_m is None else
                     float(((saved_m > 0) == m).mean()) if saved_m.shape == m.shape else "shape differs")
            if saved_m is not None and saved_m.shape == m.shape:
                m_use = saved_m > 0
            else:
                m_use = m
            p = out / "masks" / f"original_{subj}_{cam}.png"
            cv2.imwrite(str(p), m_use.astype(np.uint8) * 255)
            masks[(subj, cam)] = str(p)
            frames = [img for v in vids for img in sampled(v, s)]
            sizes = {f"{f.shape[1]}x{f.shape[0]}" for f in frames}
            med[subj] = np.median(np.stack(frames), axis=0).astype(np.uint8) if len(sizes) == 1 else None
            raw = sorted({raw_size(d) for d in grp.frames_dir})
            info[subj] = dict(mask=m_use, resized=sorted(sizes), raw=raw, recordings=len(grp),
                              sampled=i["sampled_frames"], saved_mask=str(saved) if saved_m is not None else "none",
                              reproduced_agreement=agree)
            mask_rows.append(dict(camera=cam, subject=subj, recordings=len(grp), sampled_frames=i["sampled_frames"],
                                  raw_frame_size=", ".join(raw), resized_frame_size=", ".join(sorted(sizes)),
                                  static_pixels_pct=100 * m_use.mean(),
                                  saved_mask=info[subj]["saved_mask"],
                                  reproduced_vs_saved_pixel_agreement=agree))
        m1, m2 = info[s1]["mask"], info[s2]["mask"]
        size_match = info[s1]["resized"] == info[s2]["resized"] and len(info[s1]["resized"]) == 1 \
            and info[s1]["raw"] == info[s2]["raw"] and m1.shape == m2.shape
        row = dict(camera=cam, frame_size_match=size_match,
                   raw_sizes=f"{s1}: {info[s1]['raw']} / {s2}: {info[s2]['raw']}")
        if m1.shape == m2.shape:
            inter, union, xor = (m1 & m2).sum(), (m1 | m2).sum(), (m1 ^ m2).sum()
            row.update({f"static_pct_{s1}": 100 * m1.mean(), f"static_pct_{s2}": 100 * m2.mean(),
                        "iou": inter / union if union else np.nan, "xor_pixels": int(xor),
                        "xor_pct_of_frame": 100 * xor / m1.size,
                        f"only_{s1}_pixels": int((m1 & ~m2).sum()), f"only_{s2}_pixels": int((m2 & ~m1).sum())})
            cv2.imwrite(str(out / "masks" / f"xor_{cam}.png"), (m1 ^ m2).astype(np.uint8) * 255)
        if med[s1] is not None and med[s2] is not None and med[s1].shape == med[s2].shape:
            dx, dy, resp = phase_shift(med[s1], med[s2])
            mad = float(np.abs(med[s1].astype(int) - med[s2].astype(int)).mean())
            row.update(shift_dx_px=dx, shift_dy_px=dy, max_abs_shift_px=max(abs(dx), abs(dy)),
                       phase_corr_response=resp, median_background_mean_abs_diff=mad)
            panel = [label(med[s1], f"{s1} median background ({cam})"),
                     label(med[s2], f"{s2} median background ({cam})")]
            if m1.shape == med[s1].shape[:2]:
                panel.append(label(overlay(med[s1], m1, m2), f"red {s1} only, blue {s2} only, yellow both"))
            cv2.imwrite(str(out / f"camera_and_masks_{cam}.jpg"), np.hstack(panel))
        else:
            row.update(max_abs_shift_px=math.inf)
        cam_rows.append(row)
        cam_check[cam] = dict(frame_size_match=bool(size_match),
                              max_abs_shift_px=float(row.get("max_abs_shift_px", math.inf)),
                              shift_dx_px=row.get("shift_dx_px"), shift_dy_px=row.get("shift_dy_px"),
                              phase_corr_response=row.get("phase_corr_response"),
                              raw_sizes={s1: info[s1]["raw"], s2: info[s2]["raw"]},
                              resized_sizes={s1: info[s1]["resized"], s2: info[s2]["resized"]})

    check = dict(subjects=subjects, max_shift_px=max_shift, cameras=cam_check,
                 note="Written by tools/mask_diagnostic.py. Frame sizes must match for a mask to be transferred; "
                      "a shift above max_shift_px requires --accept-camera-shift after visual confirmation.")
    (out / "camera_check.json").write_text(json.dumps(check, indent=2))
    dest = camera_check_path(corr)
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(check, indent=2))
    pd.DataFrame(mask_rows).to_csv(out / "original_masks.csv", index=False)
    cams = pd.DataFrame(cam_rows)
    cams.to_csv(out / "mask_comparison.csv", index=False)

    lines = ["# Static mask diagnostic (original pipeline)", "",
             "Original masks: one per participant and camera, from all usable recordings. The script also checks camera correspondence and silhouette drift.", "",
             "## Masks per participant and camera", "", md_table(pd.DataFrame(mask_rows), "{:.3f}", index=False), "",
             "## S01 vs S02 masks and camera correspondence", "", md_table(cams, "{:.3f}", index=False), "",
             f"Images: `camera_and_masks_<camera>.jpg` and `masks/xor_<camera>.png`. Camera check saved to `{dest}` (max allowed shift {max_shift:g} px).", ""]

    if not a.skip_silhouettes:
        jobs = []
        for rec in index.to_dict("records"):
            other = s2 if rec["subject_id"] == s1 else s1
            jobs.append((rec, masks[(rec["subject_id"], rec["camera"])], masks[(other, rec["camera"])], s))
        print(f"extracting silhouettes of {len(jobs)} recordings with both masks ...")
        with ProcessPoolExecutor(a.workers, mp_context=mp.get_context("spawn")) as pool:
            per = pd.DataFrame(list(pool.map(_two_masks, jobs)))
        per.to_csv(out / "silhouette_comparison_per_recording.csv", index=False)
        g = per.groupby(["subject_id", "camera"])
        summ = pd.DataFrame({
            "recordings": g.size(), "frames": g.frames.sum(),
            "present_own_pct": g.apply(lambda d: np.average(d.present_own_pct, weights=d.frames)),
            "present_other_pct": g.apply(lambda d: np.average(d.present_other_pct, weights=d.frames)),
            "presence_differs": g.presence_differs.sum(),
            "identical_frames_pct": 100 * g.identical_frames.sum() / g.frames.sum(),
            "materially_different_frames": g.materially_different_frames.sum(),
            "materially_different_pct": 100 * g.materially_different_frames.sum() / g.frames.sum(),
            "area_rel_change_p95_max": g.area_rel_change_p95.max(),
            "centroid_shift_px_p95_max": g.centroid_shift_px_p95.max(),
            "centroid_shift_px_max": g.centroid_shift_px_max.max()}).reset_index()
        summ.to_csv(out / "silhouette_comparison_summary.csv", index=False)
        lines += ["## Silhouettes with the own mask vs the other participant's mask", "",
                  "`own` = original pipeline; `other` = the mask used when the participant is the test subject.", "",
                  f"Materially different frame: presence differs or centroid / area / box change exceeds {CENTROID_PX:g} px / {100 * REL_CHANGE:g}%.", "",
                  md_table(summ, "{:.3f}", index=False), "",
                  "Per recording: `silhouette_comparison_per_recording.csv`.", ""]
    (out / "mask_diagnostic.md").write_text("\n".join(lines))
    print("\n".join(lines))
    print(f"\ndiagnostic -> {out}")
    bad = [c for c, v in cam_check.items() if not v["frame_size_match"]]
    moved = [c for c, v in cam_check.items() if v["max_abs_shift_px"] > max_shift]
    if bad:
        print(f"\nSTOP: frame sizes differ for {bad}. Camera correspondence is not established; do not run the corrected extraction until this is clarified.")
    elif moved:
        print(f"\nCHECK: scene shift above {max_shift:g} px for {moved}. Look at camera_and_masks_<camera>.jpg; only if it is the same installation, run extract with --accept-camera-shift.")
    else:
        print("\nCamera check passed: same frame size and no scene shift above the limit for every camera.")


if __name__ == "__main__":
    main()
