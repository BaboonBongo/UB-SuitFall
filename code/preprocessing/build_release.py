#!/usr/bin/env python3
"""Assemble the public UB-SuitFall release from the two working folders."""
import argparse
import csv
import os
import re
import shutil
import sys
from pathlib import Path

CAMERA_DIRS = {
    "CCTV-A": ["CCTV-A", "cctv-a", "Cam_A", "cam_a"],
    "CCTV-B": ["CCTV-B", "cctv-b", "Cam_B", "cam_b"],
}
FRAME_RE = re.compile(r"^\d{5}\.jpg$")


def find_dir(parent: Path, names):
    for n in names:
        if (parent / n).is_dir():
            return parent / n
    return None


_LOGS = {}


def renamed_to(vid_dir: Path, original: str):
    """Return the renamed video name recorded in rename_log.csv, if any."""
    if vid_dir not in _LOGS:
        mapping = {}
        log = vid_dir / "rename_log.csv"
        if log.is_file():
            with open(log, newline="") as f:
                for row in csv.reader(f):
                    names = [Path(v.strip()).name for v in row if v.strip().lower().endswith((".mp4", ".avi", ".mkv", ".mov"))]
                    if len(names) >= 2:
                        mapping[names[0]] = names[-1]
        _LOGS[vid_dir] = mapping
    return _LOGS[vid_dir].get(original)


def plan_subject(subject: str, root: Path):
    """Return the release jobs, problems, and skipped records for one subject."""
    jobs, problems, skipped = [], [], []
    labels = root / "fall_labels.csv"
    if not labels.is_file():
        return [], [f"{subject}: {labels} not found"], []
    with open(labels, newline="") as f:
        rows = list(csv.DictReader(f))

    for r in rows:
        if r["subject_id"] != subject:
            problems.append(f"{subject}: fall_labels.csv contains subject_id {r['subject_id']}")
            break

    jobs.append(("file", labels, Path(subject) / "fall_labels.csv"))

    for r in rows:
        cam, uid = r["camera"], r["trial_uid"]
        n = int(float(r["n_frames"] or 0))
        a, t = int(r["activity_id"]), int(r["trial_id"])
        trial = f"Activity{a}Trial{t}"
        if n == 0:
            skipped.append(f"{uid} {cam}")
            continue

        vid_dir = find_dir(root, CAMERA_DIRS[cam])
        frm_dir = find_dir(root / "Frames", CAMERA_DIRS[cam])
        if vid_dir is None or frm_dir is None:
            problems.append(f"{uid} {cam}: camera folder not found")
            continue

        src_vid = vid_dir / r["original_filename"]
        if not src_vid.is_file():
            renamed = sorted(vid_dir.glob(trial + ".*"))
            renamed = [p for p in renamed if p.suffix.lower() in (".mp4", ".avi", ".mkv", ".mov")]
            if len(renamed) == 1:
                logged = renamed_to(vid_dir, r["original_filename"])
                if logged is not None and logged != renamed[0].name:
                    problems.append(f"{uid} {cam}: rename_log.csv says {r['original_filename']} -> {logged}, but expected {renamed[0].name}")
                src_vid = renamed[0]
        if src_vid.is_file():
            dst_vid = Path(subject) / cam / (trial + src_vid.suffix.lower())
            jobs.append(("file", src_vid, dst_vid))
        else:
            problems.append(f"{uid} {cam}: video {src_vid} not found")
            src_vid = None

        src_frames = frm_dir / trial
        frames = sorted(p.name for p in src_frames.iterdir() if FRAME_RE.match(p.name)) if src_frames.is_dir() else []
        expected = [f"{i:05d}.jpg" for i in range(1, n + 1)]
        if frames != expected:
            have = len(frames)
            missing = sorted(set(expected) - set(frames))[:3]
            extra = sorted(set(frames) - set(expected))[:3]
            problems.append(f"{uid} {cam}: {have} frames in {src_frames}, expected {n}"
                            + (f"; missing e.g. {missing}" if missing else "")
                            + (f"; extra e.g. {extra}" if extra else ""))
        for name in frames:
            jobs.append(("file", src_frames / name, Path(subject) / "Frames" / cam / trial / name))

        fl = root / "frame_labels" / f"{uid}_{cam}.csv"
        if fl.is_file():
            jobs.append(("file", fl, Path(subject) / "frame_labels" / fl.name))
        else:
            problems.append(f"{uid} {cam}: {fl} not found")

        jobs.append(("manifest", None, {
            "subject_id": subject, "trial_uid": uid, "camera": cam,
            "activity_id": a, "trial_id": t, "category": r["category"],
            "video": str(Path(subject) / cam / (trial + (src_vid.suffix.lower() if src_vid else ".mp4"))),
            "frames_dir": str(Path(subject) / "Frames" / cam / trial),
            "n_frames": n,
            "frame_labels": str(Path(subject) / "frame_labels" / f"{uid}_{cam}.csv"),
        }))
    return jobs, problems, skipped


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--s01", required=True, type=Path, help="S01's working folder")
    ap.add_argument("--s02", required=True, type=Path, help="S02's working folder")
    ap.add_argument("--out", required=True, type=Path, help="new release folder (must not exist or be empty)")
    ap.add_argument("--check", action="store_true", help="only verify, copy nothing")
    ap.add_argument("--link", action="store_true", help="hard-link instead of copy")
    args = ap.parse_args()

    out = args.out.resolve()
    for src in (args.s01, args.s02):
        s = src.resolve()
        if out == s or s in out.parents:
            sys.exit(f"--out must not be inside a source folder ({s})")
    if out.exists() and any(out.iterdir()) and not args.check:
        sys.exit(f"{out} already exists and is not empty; choose a new folder")

    jobs, problems, skipped = [], [], []
    for subject, root in (("S01", args.s01), ("S02", args.s02)):
        j, p, s = plan_subject(subject, root.resolve())
        jobs += j
        problems += p
        skipped += s

    files = [j for j in jobs if j[0] == "file"]
    manifest = [j[2] for j in jobs if j[0] == "manifest"]
    size = sum(src.stat().st_size for _, src, _ in files)
    print(f"Recordings to release: {len(manifest)} "
          f"(S01 {sum(m['subject_id']=='S01' for m in manifest)}, "
          f"S02 {sum(m['subject_id']=='S02' for m in manifest)})")
    print(f"Frames: {sum(m['n_frames'] for m in manifest):,}")
    print(f"Files: {len(files):,}, total {size/1e9:.2f} GB")
    print(f"Skipped (n_frames = 0): {len(skipped)}" + (": " + ", ".join(skipped) if skipped else ""))
    if problems:
        print(f"\nPROBLEMS ({len(problems)}):")
        for p in problems:
            print("  -", p)
    if args.check:
        print("\n--check: nothing copied.")
        return
    if problems:
        sys.exit("\nNothing copied. Fix the problems above and run again.")

    for i, (_, src, rel) in enumerate(files, 1):
        dst = out / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        if args.link:
            try:
                os.link(src, dst)
                continue
            except OSError:
                pass
        shutil.copy2(src, dst)
        if i % 5000 == 0:
            print(f"  {i:,}/{len(files):,} files")

    with open(out / "MANIFEST.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(manifest[0]))
        w.writeheader()
        w.writerows(manifest)
    print(f"\nDone: {out}")


if __name__ == "__main__":
    main()