#!/usr/bin/env python3
# ==============================================================
# fall_labels.py
# ==============================================================
#
# Builds the fall / non-fall metadata layer described in the
# Fall/Non-Fall Labeling Plan, WITHOUT touching any video, frame,
# CVAT XML or COCO file.
#
# Two steps:
#
#   1. suggest  - one fall_labels.csv per subject: every expected trial
#                 x camera, its category (Activity 1-5 FALL, 6-10
#                 NON_FALL), status (ok / no_frames / missing), original
#                 video filename, and - for FALL trials - SUGGESTED
#                 fall_start / impact / fall_end frames computed from the
#                 existing hip/shoulder keypoints. Also writes one review
#                 image per fall trial (fall_review/) showing the frames
#                 around the suggestion and the motion curve.
#                 You then fill in / correct fall_start_frame,
#                 impact_frame, fall_end_frame and set confirmed = yes.
#                 Re-running suggest never overwrites those columns.
#
#   2. export   - from the finished CSV, writes per-frame labels for
#                 every trial and camera: the state (PRE_FALL / FALL /
#                 POST_FALL / NON_FALL), a binary label (FALL = 1, else 0)
#                 and a 3-class label (normal = 0, FALL = 1, POST_FALL = 2),
#                 after checking every fall row is consistent.
#                 Works on the full CSV written by suggest (rows need
#                 confirmed = yes) and on a cleaned CSV that only keeps the
#                 final fall_start_frame / impact_frame / fall_end_frame
#                 columns (no confirmed / status columns: every filled-in
#                 row counts as final, rows with n_frames = 0 are skipped).
#
# Usage:
#   python fall_labels.py suggest --subject S01 /path/to/S01
#   python fall_labels.py suggest --subject S02 /path/to/S02
#   python fall_labels.py export /path/to/S01
#   python fall_labels.py export /path/to/S01 --use-suggestions   (quick experiments only)
#
# Expected folder layout (both participants' working folders follow it):
#   <subject root>/<camera>/ActivityXTrialY.mp4         (videos; optional rename_log.csv)
#   <subject root>/Frames/<camera>/ActivityXTrialY/00001.jpg ...  + annotations XML
#
# Frame numbers in the CSV are 1-BASED and match the frame filenames
# (frame 1 = 00001.jpg).
# ==============================================================

import csv
import math
import re
import sys
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

# ------------------------------------------------------------
# CONFIGURATION
# ------------------------------------------------------------
FPS = 10                       # frame extraction rate used by frame_extraction.py
ACTIVITIES = {                 # SOP-FALL-LONG-2026-V2.2, Section 8
    1: ("forward_fall_hands", "FALL"),
    2: ("forward_fall_knees", "FALL"),
    3: ("falling_backward", "FALL"),
    4: ("falling_sideward", "FALL"),
    5: ("fall_missed_chair", "FALL"),
    6: ("walking", "NON_FALL"),
    7: ("standing", "NON_FALL"),
    8: ("sitting", "NON_FALL"),
    9: ("picking_up_object", "NON_FALL"),
    10: ("lying_down", "NON_FALL"),
}
TRIALS_PER_ACTIVITY = 3
# Pose annotation file read for the suggestions, first one found wins.
# Suggestions only need rough hip/shoulder positions, so the automatic
# output is fine; a CVAT export of your corrected annotations is better.
ANNOTATION_FILES = ["annotations.xml", "annotations_v2.xml"]
LABELS_FILENAME = "fall_labels.csv"
REVIEW_DIRNAME = "fall_review"
FRAME_LABELS_DIRNAME = "frame_labels"

# Suggestion tuning
EDGE_FRACTION = 0.15           # first 15% of frames = the starting ("before") posture
TILT_MIN_TORSO_FRACTION = 0.5  # torso shorter than this x standing length on screen -> tilt ignored
GLITCH_FILTER_FRAMES = 7       # running median: removes pose glitches up to 3 frames long
FAST_SPEED = 0.025             # "fast" = covering the whole drop within ~4 s (progress per frame)
FAST_GAP_FRAMES = 2            # pauses up to this long inside the descent are bridged
MIN_PEAK_SPEED = 0.03          # peak speed (progress per frame) below this = no clear fall
GLITCH_JUMP = 0.40             # raw vs filtered progress differing by this = a pose glitch
MAX_GLITCH_FRAMES = 3
MIN_FALL_FRAMES = 2            # plausibility checks (flag rows as "check", never change them)
MAX_FALL_FRAMES = 20           # 2 s at 10 fps (controlled falls in the suit can take ~1.6 s)
MIN_SEGMENT_GAIN = 0.20        # a descent must cover >= 20% of the way down to count as the fall
ALREADY_DOWN_PROGRESS = 0.80   # descents starting after the body was already this far down are ignored
CAMERA_OFFSET_TOLERANCE = 30   # cameras placing the fall more than 3 s apart -> check
MAX_SETTLE_FRAMES = 30         # end more than 3 s after impact
CAMERA_DURATION_TOLERANCE = 5  # A and B fall durations (frames) may differ by this much
# (A and B videos of one trial may legitimately differ in length - e.g. one
#  camera recording 20 s and the other 10 s - so lengths are not compared.)
STILL_MOTION_RATIO = 0.02      # "still" = median joint movement < 2% of body size per frame
STILL_FRAMES = 5               # ... for this many consecutive frames (0.5 s at 10 fps)
MIN_HIP_CHANGE_RATIO = 0.15    # hip must move at least this x body size to be usable
MIN_TILT_CHANGE_DEG = 30.0     # torso must tilt at least this much to be usable

COLUMNS = [
    "subject_id", "trial_uid", "camera", "activity_id", "activity_name", "trial_id",
    "category", "original_filename", "status", "n_frames", "fps", "frame_numbering",
    "suggested_fall_start", "suggested_impact", "suggested_fall_end", "suggestion_quality", "check_reason",
    "fall_start_frame", "impact_frame", "fall_end_frame", "confirmed", "notes",
]
HUMAN_COLUMNS = ["fall_start_frame", "impact_frame", "fall_end_frame", "confirmed", "notes"]
TRIAL_PATTERN = re.compile(r"^Activity(\d+)Trial(\d+)$")


# ------------------------------------------------------------
# Layout helpers
# ------------------------------------------------------------
def camera_label(folder_name: str) -> str:
    """cctv-a / Cam_A / cama -> CCTV-A (same for B)."""
    n = folder_name.lower().replace("_", "").replace("-", "")
    if n.endswith("a"):
        return "CCTV-A"
    if n.endswith("b"):
        return "CCTV-B"
    return folder_name.upper()


def frame_number(filename: str) -> Optional[int]:
    stem = Path(filename).stem
    return int(stem) if stem.isdigit() else None


def list_frames(trial_dir: Path) -> List[Path]:
    if not trial_dir.is_dir():
        return []
    return sorted((p for p in trial_dir.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"}
                   and frame_number(p.name) is not None), key=lambda p: frame_number(p.name))


def read_rename_log(video_dir: Path) -> Dict[str, str]:
    """new_name -> old_name, from rename_trials.py's log (S02)."""
    log = video_dir / "rename_log.csv"
    if not log.exists():
        return {}
    with log.open(encoding="utf-8") as fh:
        return {row["new_name"]: row["old_name"] for row in csv.DictReader(fh)}


# ------------------------------------------------------------
# Pose signal from the annotation XML
# ------------------------------------------------------------
def load_pose(xml_path: Path) -> Dict[int, Dict[str, Tuple[float, float]]]:
    """frame number -> {label: (x, y)} for points with outside="0"."""
    poses: Dict[int, Dict[str, Tuple[float, float]]] = {}
    root = ET.parse(str(xml_path)).getroot()
    for image in root.findall("image"):
        n = frame_number(image.get("name", ""))
        skeleton = image.find("skeleton")
        if n is None or skeleton is None:
            continue
        pts = {}
        for p in skeleton.findall("points"):
            if p.get("outside") == "1":
                continue
            x, y = (float(v) for v in p.get("points").split(","))
            pts[p.get("label")] = (x, y)
        poses[n] = pts
    return poses


def _mid(pts: Dict[str, Tuple[float, float]], a: str, b: str) -> Optional[np.ndarray]:
    got = [pts[k] for k in (a, b) if k in pts]
    return np.mean(np.array(got), axis=0) if got else None


def _fill(values: np.ndarray) -> np.ndarray:
    """Linear-fill NaNs (edges take the nearest value)."""
    v = values.astype(float).copy()
    ok = ~np.isnan(v)
    if ok.sum() == 0:
        return v
    idx = np.arange(len(v))
    v[~ok] = np.interp(idx[~ok], idx[ok], v[ok])
    return v


def _median3(v: np.ndarray) -> np.ndarray:
    if len(v) < 3:
        return v
    p = np.pad(v, 1, mode="edge")
    return np.median(np.stack([p[:-2], p[1:-1], p[2:]]), axis=0)


def _median_k(v: np.ndarray, k: int) -> np.ndarray:
    """Running median of width k (odd); removes glitches shorter than k/2 frames."""
    if len(v) < k:
        return v
    h = k // 2
    p = np.pad(v, h, mode="edge")
    return np.median(np.stack([p[i:i + len(v)] for i in range(k)]), axis=0)


def suggest_boundaries(poses: Dict[int, Dict[str, Tuple[float, float]]], n_frames: int):
    """
    Returns (start, impact, end, quality, progress_curve, reasons) with
    1-based frame numbers (None where not found). quality: "ok", "check"
    or "no_signal"; reasons lists why a row needs checking.

    Fall progress per frame = how far the hips have dropped from their
    starting height toward the lowest point reached (0 = standing,
    1 = on the floor), with pose glitches up to 3 frames filtered out.
    Torso tilt is used instead only when the hips barely move. The FALL
    is the fast part of that drop:
      fall_start = first frame of the fast descent
      impact     = first frame after the fast descent stops (body has
                   reached the floor / mat; slow settling, sliding or lying
                   back afterwards is post-fall, not part of the fall)
      fall_end   = first frame from impact on where the body stays still
                   for 0.5 s
    Short pauses (<= FAST_GAP_FRAMES) inside the descent are bridged, so a
    fall onto the hands followed by the body going down counts as one fall.
    """
    reasons: List[str] = []
    frames = list(range(1, n_frames + 1))
    hip_y = np.full(n_frames, np.nan)
    tilt = np.full(n_frames, np.nan)
    torso_len = np.full(n_frames, np.nan)
    height = np.full(n_frames, np.nan)
    for i, f in enumerate(frames):
        pts = poses.get(f)
        if not pts:
            continue
        hip = _mid(pts, "left_hip", "right_hip")
        sh = _mid(pts, "left_shoulder", "right_shoulder")
        if hip is not None:
            hip_y[i] = hip[1]
        if hip is not None and sh is not None:
            v = sh - hip
            torso_len[i] = float(np.linalg.norm(v))
            if torso_len[i] > 1:
                tilt[i] = abs(math.degrees(math.atan2(v[0], -v[1])))   # 0 = upright, 180 = upside down
        if len(pts) >= 4:
            ys = [p[1] for p in pts.values()]
            xs = [p[0] for p in pts.values()]
            height[i] = math.hypot(max(xs) - min(xs), max(ys) - min(ys))

    if np.isnan(hip_y).all():
        return None, None, None, "no_signal", np.zeros(n_frames), ["no hip keypoints"]

    edge = max(3, int(round(EDGE_FRACTION * n_frames)))
    body = np.nanmedian(height[:edge]) if not np.isnan(height[:edge]).all() else np.nanmedian(height)
    # Tilt is meaningless when the torso points at the camera (falling
    # forward toward it): the shoulder-hip line shrinks on screen and its
    # direction flips 0 -> 180 deg in one frame. Ignore tilt in those frames.
    if not np.isnan(torso_len[:edge]).all():
        standing_torso = np.nanmedian(torso_len[:edge])
        tilt[torso_len < TILT_MIN_TORSO_FRACTION * standing_torso] = np.nan
    hip_raw, tilt_raw = _fill(hip_y), _fill(tilt)
    hip_y, tilt = _median_k(hip_raw, GLITCH_FILTER_FRAMES), _median_k(tilt_raw, GLITCH_FILTER_FRAMES)

    def extreme_change(values: np.ndarray) -> float:
        """How far the signal gets from its starting value at its furthest
        (95th percentile of the deviation, keeping its sign). Uses the
        furthest point rather than the last frames, so a subject who gets
        back up or rolls over before the video ends is still handled."""
        dev = values - np.median(values[:edge])
        hi, lo = np.percentile(dev, 95), np.percentile(dev, 5)
        return float(hi if abs(hi) >= abs(lo) else lo)

    signals, raw_signals = [], []
    hip_used = False
    hip_change = extreme_change(hip_y)
    if body > 0 and abs(hip_change) >= MIN_HIP_CHANGE_RATIO * body:
        hip_used = True
        base = np.median(hip_y[:edge])
        signals.append((hip_y - base) / hip_change)
        raw_signals.append((hip_raw - base) / hip_change)
    if not np.isnan(tilt).all():
        tilt_change = extreme_change(tilt)
        if abs(tilt_change) >= MIN_TILT_CHANGE_DEG:
            base = np.median(tilt[:edge])
            signals.append((tilt - base) / tilt_change)
            raw_signals.append((tilt_raw - base) / tilt_change)
    if not signals:
        return None, None, None, "no_signal", np.zeros(n_frames), ["hips/torso barely move"]
    # The fall itself = the hips dropping from standing height to the floor,
    # which reads the same from both cameras. Torso movement after the hips
    # are down (lying back after sitting down hard, rolling over) is
    # post-fall. Torso tilt is only used when hip height is not usable.
    progress = np.clip(signals[0], -0.5, 1.5)
    raw = np.clip(raw_signals[0], -0.5, 1.5)
    glitches = int(np.sum(np.abs(raw - progress) > GLITCH_JUMP))
    if glitches > MAX_GLITCH_FRAMES:
        reasons.append(f"pose jumps around in {glitches} frames - suggestion less reliable")

    # --- fast descent ---------------------------------------------------
    # speed smoothed over 3 frames, so a single-frame jump can't dominate
    speed = np.convolve(np.gradient(progress), np.ones(3) / 3.0, mode="same")
    peak = int(np.argmax(speed))
    if speed[peak] < MIN_PEAK_SPEED:
        return None, None, None, "check", progress, reasons + ["no clear fast descent"]
    # "Fast" is an absolute speed: progress runs 0 -> 1 over the whole drop,
    # so FAST_SPEED = 0.025 means covering it within ~4 s. Squatting,
    # sliding along the mat or slowly lying back stay below it. (A cutoff
    # relative to the trial's fastest movement failed whenever one pose
    # glitch or a camera-geometry jump was faster than the real fall.)
    fast = speed > FAST_SPEED

    def extend(i: int, step: int) -> int:
        """Walk from the peak while the descent is fast, bridging short pauses."""
        last, gap, j = i, 0, i
        while 0 <= j + step < n_frames:
            j += step
            if fast[j]:
                last, gap = j, 0
            else:
                gap += 1
                if gap > FAST_GAP_FRAMES:
                    break
        return last

    # Candidate descents = runs of fast frames (short pauses bridged). The
    # fall is the FIRST substantial descent made while the body is still up:
    # a later fast movement (lying back, rolling, or a pose glitch snapping
    # back after drifting) is not a new fall once the body is already down.
    segments = []
    i = 0
    while i < n_frames:
        if fast[i]:
            s_i = i
            e_i = extend(i, +1)
            seg_peak = float(speed[s_i:e_i + 1].max())
            imp = min(n_frames - 1, e_i + 1)
            gain = float(progress[imp] - progress[s_i])
            already_down = float(np.max(progress[:s_i + 1])) >= ALREADY_DOWN_PROGRESS
            segments.append((s_i, e_i, seg_peak, gain, already_down))
            i = e_i + 1
        else:
            i += 1
    eligible = [sg for sg in segments if sg[3] >= MIN_SEGMENT_GAIN and not sg[4]]
    if eligible:
        start_i, last_fast = eligible[0][0], eligible[0][1]
    else:
        start_i, last_fast = extend(peak, -1), extend(peak, +1)
        reasons.append("no clear single descent - picked the fastest movement")
    impact_i = min(n_frames - 1, last_fast + 1)
    if progress[impact_i] < 0.5:
        reasons.append("descent stops less than halfway down - maybe not the main fall")

    # --- stillness after impact ------------------------------------------
    motion = np.full(n_frames, np.nan)
    for i in range(n_frames - 1):
        a, b = poses.get(frames[i]), poses.get(frames[i + 1])
        if a and b:
            common = [k for k in a if k in b]
            if common:
                motion[i] = np.median([math.hypot(a[k][0] - b[k][0], a[k][1] - b[k][1]) for k in common])
    end_i = None
    limit = STILL_MOTION_RATIO * body
    for i in range(impact_i, n_frames - STILL_FRAMES + 1):
        window = motion[i:i + STILL_FRAMES]
        if np.isnan(window).sum() <= 1 and np.sum(np.nan_to_num(window, nan=0.0) >= limit) <= 1:
            end_i = i
            break
    if end_i is None:
        reasons.append("body never stays still after impact")

    duration = impact_i - start_i
    if duration < MIN_FALL_FRAMES:
        reasons.append(f"fall only {duration} frame(s) long")
    elif duration > MAX_FALL_FRAMES:
        reasons.append(f"fall {duration} frames ({duration / FPS:.1f} s) long - unusually slow")
    if end_i is not None and end_i - impact_i > MAX_SETTLE_FRAMES:
        reasons.append(f"end is {end_i - impact_i} frames after impact")
    if not hip_used:
        reasons.append("hip height barely changes - based on torso tilt only")

    quality = "check" if reasons else "ok"
    return (start_i + 1, impact_i + 1, (end_i + 1) if end_i is not None else None, quality, progress, reasons)


# ------------------------------------------------------------
# Review sheet
# ------------------------------------------------------------
def write_review_sheet(path: Path, frame_files: List[Path], progress: np.ndarray,
                       start: Optional[int], impact: Optional[int], end: Optional[int], title: str) -> None:
    n = len(frame_files)
    marks = [m for m in (start, impact, end) if m]
    if marks:
        lo, hi = max(1, min(marks) - 4), min(n, max(marks) + 4)
    else:
        lo, hi = 1, n
    chosen = list(range(lo, hi + 1))
    if len(chosen) > 30:   # keep the sheet readable, always keep the marked frames
        step = len(chosen) / 30.0
        chosen = sorted(set([chosen[int(i * step)] for i in range(30)] + marks))
    thumb_w, cols = 240, 6
    thumbs = []
    markers = [(m, nm, col) for m, nm, col in ((start, "START", (0, 220, 255)), (impact, "IMPACT", (0, 0, 255)),
                                                  (end, "END", (255, 120, 0))) if m]
    for f in chosen:
        img = cv2.imread(str(frame_files[f - 1]))
        if img is None:
            continue
        h = int(img.shape[0] * thumb_w / img.shape[1])
        t = cv2.resize(img, (thumb_w, h))
        tag = f"{f}"
        here = [(nm, col) for m, nm, col in markers if m == f]
        if here:
            cv2.rectangle(t, (0, 0), (thumb_w - 1, h - 1), here[0][1], 6)
            tag += " " + "+".join(nm for nm, _ in here)
        cv2.putText(t, tag, (6, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(t, tag, (6, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
        thumbs.append(t)
    if not thumbs:
        return
    th = thumbs[0].shape[0]
    rows = math.ceil(len(thumbs) / cols)
    sheet = np.full((rows * th + 200, cols * thumb_w, 3), 255, np.uint8)
    for i, t in enumerate(thumbs):
        r, c = divmod(i, cols)
        sheet[r * th:r * th + t.shape[0], c * thumb_w:(c + 1) * thumb_w] = t[:th]
    # progress curve over the whole trial
    y0, ch, W = rows * th + 30, 140, cols * thumb_w
    cv2.putText(sheet, title + "   (curve: fall progress over the whole trial)", (8, y0 - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1, cv2.LINE_AA)
    cv2.rectangle(sheet, (0, y0), (W - 1, y0 + ch), (200, 200, 200), 1)
    if n > 1:
        pts = [(int((i / (n - 1)) * (W - 1)), int(y0 + ch - np.clip(p, 0, 1.2) / 1.2 * ch)) for i, p in enumerate(progress)]
        cv2.polylines(sheet, [np.array(pts, np.int32)], False, (60, 60, 60), 2)
        for offset, (m, nm, col) in enumerate(markers):
            x = int(((m - 1) / (n - 1)) * (W - 1)) + 3 * offset   # nudge so coinciding lines stay visible
            cv2.line(sheet, (x, y0), (x, y0 + ch), col, 2)
        legend = "yellow = START   red = IMPACT   blue = END"
        cv2.putText(sheet, legend, (8, y0 + ch + 22), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), sheet)


# ------------------------------------------------------------
# suggest
# ------------------------------------------------------------
def cmd_suggest(subject_id: str, root: Path) -> None:
    frames_root = root / "Frames"
    if not frames_root.is_dir():
        sys.exit(f"[ERROR] No Frames folder in {root}")
    cameras = sorted(p for p in frames_root.iterdir() if p.is_dir())
    labels_path = root / LABELS_FILENAME

    existing: Dict[Tuple[str, str], Dict[str, str]] = {}
    if labels_path.exists():
        with labels_path.open(encoding="utf-8") as fh:
            for row in csv.DictReader(fh):
                existing[(row["trial_uid"], row["camera"])] = row

    rows = []
    for cam_dir in cameras:
        cam = camera_label(cam_dir.name)
        video_dir = root / cam_dir.name
        renamed = read_rename_log(video_dir)
        for act, (act_name, category) in ACTIVITIES.items():
            for trial in range(1, TRIALS_PER_ACTIVITY + 1):
                name = f"Activity{act}Trial{trial}"
                uid = f"{subject_id}_A{act:02d}_T{trial}"
                trial_dir = cam_dir / name
                frame_files = list_frames(trial_dir)
                video = video_dir / f"{name}.mp4"
                if frame_files:
                    status = "ok"
                elif video.exists():
                    status = "no_frames"
                else:
                    status = "missing"
                original = renamed.get(f"{name}.mp4", f"{name}.mp4" if video.exists() or frame_files else "")

                row = {c: "" for c in COLUMNS}
                row.update(subject_id=subject_id, trial_uid=uid, camera=cam, activity_id=act,
                           activity_name=act_name, trial_id=trial, category=category,
                           original_filename=original, status=status, n_frames=len(frame_files),
                           fps=FPS, frame_numbering="1-based (frame 1 = 00001.jpg)")

                if category == "FALL" and status == "ok":
                    xml = next((trial_dir / f for f in ANNOTATION_FILES if (trial_dir / f).exists()), None)
                    if xml is None:
                        row["suggestion_quality"] = "no_pose_file"
                    else:
                        try:
                            start, impact, end, quality, progress, reasons = suggest_boundaries(load_pose(xml), len(frame_files))
                        except ET.ParseError:
                            start = impact = end = None
                            quality, progress, reasons = "bad_pose_file", np.zeros(len(frame_files)), []
                        row.update(suggested_fall_start=start or "", suggested_impact=impact or "",
                                   suggested_fall_end=end or "", suggestion_quality=quality,
                                   check_reason="; ".join(reasons))
                        write_review_sheet(root / REVIEW_DIRNAME / f"{uid}_{cam}.jpg", frame_files, progress,
                                           start, impact, end, f"{uid} {cam} {act_name}")

                old = existing.get((uid, cam))
                if old:
                    for c in HUMAN_COLUMNS:
                        row[c] = old.get(c, "")
                rows.append(row)
                print(f"{uid} {cam:<7} {category:<8} {status:<9} frames={len(frame_files):>4} "
                      f"suggested={row['suggested_fall_start'] or '-'}/{row['suggested_impact'] or '-'}/"
                      f"{row['suggested_fall_end'] or '-'} {row['suggestion_quality']}")

    # --- cross-camera checks: A and B film the same physical trial --------
    by_trial: Dict[str, List[Dict[str, object]]] = {}
    for row in rows:
        if row["status"] == "ok":
            by_trial.setdefault(row["trial_uid"], []).append(row)
    for uid, pair in by_trial.items():
        if len(pair) != 2:
            continue
        a, b = sorted(pair, key=lambda r: r["camera"])
        extra = []
        if a["category"] == "FALL" and all(r["suggested_fall_start"] and r["suggested_impact"] for r in pair):
            da = int(a["suggested_impact"]) - int(a["suggested_fall_start"])
            db = int(b["suggested_impact"]) - int(b["suggested_fall_start"])
            if abs(da - db) > CAMERA_DURATION_TOLERANCE:
                extra.append(f"fall lasts {da} frames on {a['camera']} but {db} on {b['camera']}")
            offset = int(a["suggested_fall_start"]) - int(b["suggested_fall_start"])
            if abs(offset) > CAMERA_OFFSET_TOLERANCE:
                extra.append(f"cameras place the fall {abs(offset) / FPS:.1f} s apart - one of them picked another movement")
        for r in pair:
            if extra:
                r["check_reason"] = "; ".join([x for x in [r["check_reason"]] + extra if x])
                if r["category"] == "FALL" and r["suggestion_quality"] == "ok":
                    r["suggestion_quality"] = "check"

    with labels_path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=COLUMNS)
        w.writeheader()
        w.writerows(rows)

    falls = [r for r in rows if r["category"] == "FALL" and r["status"] == "ok"]
    print()
    print(f"Wrote {labels_path}  ({len(rows)} rows: {sum(r['status'] == 'ok' for r in rows)} ok, "
          f"{sum(r['status'] == 'no_frames' for r in rows)} no_frames, {sum(r['status'] == 'missing' for r in rows)} missing)")
    print(f"Fall rows to confirm: {len(falls)}  (quality ok: {sum(r['suggestion_quality'] == 'ok' for r in falls)}, "
          f"check: {sum(r['suggestion_quality'] != 'ok' for r in falls)})")
    print(f"Review images: {root / REVIEW_DIRNAME}/")
    to_check = [r for r in rows if r["check_reason"]]
    if to_check:
        print()
        print("Rows to look at closely (check_reason column):")
        for r in to_check:
            print(f"  {r['trial_uid']} {r['camera']:<7} {r['suggested_fall_start'] or '-'}/{r['suggested_impact'] or '-'}/"
                  f"{r['suggested_fall_end'] or '-'}  {r['check_reason']}")
    if existing:
        print("Your existing fall_start/impact/end, confirmed and notes columns were kept.")


# ------------------------------------------------------------
# export
# ------------------------------------------------------------
def _int(v: str) -> Optional[int]:
    v = (v or "").strip()
    return int(float(v)) if v else None


# 3-class scheme: normal (before a fall, or any non-fall activity) = 0,
# falling = 1, on the ground after the fall = 2.
THREE_CLASS = {"NON_FALL": 0, "PRE_FALL": 0, "FALL": 1, "POST_FALL": 2}


def cmd_export(root: Path, use_suggestions: bool) -> None:
    labels_path = root / LABELS_FILENAME
    if not labels_path.exists():
        sys.exit(f"[ERROR] {labels_path} not found - run suggest first")
    with labels_path.open(encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        rows = list(reader)
        fields = reader.fieldnames or []
    # A cleaned CSV (review columns removed) has no confirmed / status
    # columns: its fall frames are taken as final.
    needs_confirm = "confirmed" in fields
    has_status = "status" in fields

    problems, written, skipped = [], 0, []
    counts = {"NON_FALL": 0, "PRE_FALL": 0, "FALL": 0, "POST_FALL": 0}
    out_dir = root / FRAME_LABELS_DIRNAME
    for row in rows:
        key = f"{row['trial_uid']} {row['camera']}"
        n = _int(row.get("n_frames", "")) or 0
        status = row["status"] if has_status else ("ok" if n > 0 else "no frames")
        if status != "ok" or n <= 0:
            skipped.append(f"{key} ({status})")
            continue
        if row["category"] == "NON_FALL":
            states = ["NON_FALL"] * n
        else:
            confirmed = (not needs_confirm) or row.get("confirmed", "").strip().lower() == "yes"
            if use_suggestions and not confirmed:
                s, i, e = _int(row["suggested_fall_start"]), _int(row["suggested_impact"]), _int(row["suggested_fall_end"])
            else:
                if not confirmed:
                    problems.append(f"{key}: not confirmed yet")
                    continue
                s, i, e = _int(row["fall_start_frame"]), _int(row["impact_frame"]), _int(row["fall_end_frame"])
            if None in (s, i, e):
                problems.append(f"{key}: fall_start / impact / fall_end missing")
                continue
            if not (1 <= s <= i <= e <= n):
                problems.append(f"{key}: need 1 <= start ({s}) <= impact ({i}) <= end ({e}) <= {n}")
                continue
            states = ["PRE_FALL"] * (s - 1) + ["FALL"] * (e - s + 1) + ["POST_FALL"] * (n - e)
        out_dir.mkdir(exist_ok=True)
        path = out_dir / f"{row['trial_uid']}_{row['camera']}.csv"
        with path.open("w", newline="", encoding="utf-8") as fh:
            w = csv.writer(fh)
            w.writerow(["frame", "filename", "state", "label", "label_3class"])
            for f, st in enumerate(states, start=1):
                w.writerow([f, f"{f:05d}.jpg", st, 1 if st == "FALL" else 0, THREE_CLASS[st]])
                counts[st] += 1
        written += 1

    print(f"Frame-label files written: {written}  -> {out_dir}/")
    total = sum(counts.values())
    if total:
        print("Frames per state: " + ", ".join(f"{k} {v} ({100 * v / total:.1f}%)" for k, v in counts.items()))
    if skipped:
        shown = ", ".join(skipped[:6]) + (f", ... (+{len(skipped) - 6} more)" if len(skipped) > 6 else "")
        print(f"Skipped {len(skipped)} row(s) without frames: {shown}")
    if problems:
        print(f"NOT written ({len(problems)}) - fix these rows in {LABELS_FILENAME}:")
        for p in problems:
            print("  " + p)
    if use_suggestions:
        print("NOTE: unconfirmed fall rows used the automatic suggestions - fine for quick experiments, "
              "not as ground truth.")


def main() -> None:
    args = sys.argv[1:]
    usage = ("Usage:\n  python fall_labels.py suggest --subject S01 <subject root>\n"
             "  python fall_labels.py export <subject root> [--use-suggestions]")
    if len(args) >= 4 and args[0] == "suggest" and args[1] == "--subject":
        cmd_suggest(args[2], Path(args[3]))
    elif len(args) >= 2 and args[0] == "export":
        cmd_export(Path(args[1]), "--use-suggestions" in args)
    else:
        sys.exit(usage)


if __name__ == "__main__":
    main()
