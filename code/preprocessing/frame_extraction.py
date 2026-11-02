#!/usr/bin/env python3
"""Extract frames at 10 fps from the UB-SuitFall videos with ffmpeg.

This is the script that produced the Frames/ folders of the release. Both
participants were processed with the same settings; only the folder names
differed between the working copies, so camera folders are found
automatically (CCTV-A / cctv-a / Cam_A, and the same for B).

Videos must already be named ActivityXTrialY.mp4 (see rename_trials.py).

Output:
    <root>/Frames/<camera>/<video name>/00001.jpg, 00002.jpg, ...

Usage:
    python frame_extraction.py --root /path/to/S01 --skip Activity7Trial2.mp4
    python frame_extraction.py --root /path/to/S02

During extraction, S01's Activity7Trial2.mp4 (both cameras) was skipped
because the video was corrupted; it is not part of the release. Recordings
whose frame folder already contains .jpg files are skipped.

Requires ffmpeg on the PATH. Frames decoded with a different ffmpeg version
may differ slightly from the released frames, which are the reference.
"""
import argparse
import subprocess
from pathlib import Path

FPS = 10  # frame extraction rate

# camera folder names used in the release and in the working copies
CAMERA_DIRS = {
    "CCTV-A": ["CCTV-A", "cctv-a", "Cam_A", "cam_a"],
    "CCTV-B": ["CCTV-B", "cctv-b", "Cam_B", "cam_b"],
}


def find_camera_folders(root: Path) -> list:
    """Return the existing camera folder names under root, one per camera."""
    found = []
    for camera, names in CAMERA_DIRS.items():
        name = next((n for n in names if (root / n).is_dir()), None)
        if name is None:
            print(f"[WARNING] No folder for {camera} in {root} (looked for {', '.join(names)})")
        else:
            found.append(name)
    return found


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", required=True, type=Path,
                    help="participant folder containing the camera folders")
    ap.add_argument("--skip", nargs="*", default=[], metavar="VIDEO",
                    help="video file names to skip (known corrupted videos)")
    ap.add_argument("--fps", type=float, default=FPS,
                    help=f"extraction rate (default {FPS}, as used for the release)")
    args = ap.parse_args()

    root = args.root
    skip_files = set(args.skip)
    processed = skipped_existing = skipped_corrupted = failed = 0

    print("=" * 60)
    print("Starting frame extraction...")
    print("=" * 60)

    for camera in find_camera_folders(root):
        camera_folder = root / camera
        print(f"\nProcessing {camera}...")

        for video in sorted(camera_folder.glob("*.mp4")):
            if video.name in skip_files:
                print(f"[SKIP] Corrupted video: {video.name}")
                skipped_corrupted += 1
                continue

            output_folder = root / "Frames" / camera / video.stem
            if output_folder.exists() and any(output_folder.glob("*.jpg")):
                print(f"[SKIP] Already extracted: {video.name}")
                skipped_existing += 1
                continue
            output_folder.mkdir(parents=True, exist_ok=True)

            command = [
                "ffmpeg", "-y",
                "-i", str(video),
                "-vf", f"fps={args.fps:g}",
                str(output_folder / "%05d.jpg"),
            ]
            print(f"[PROCESS] {video.name}")
            try:
                subprocess.run(command, check=True,
                               stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                processed += 1
            except subprocess.CalledProcessError:
                print(f"[FAILED] {video.name}")
                failed += 1

    print("\n" + "=" * 60)
    print("Extraction Complete")
    print("=" * 60)
    print(f"Successfully extracted : {processed}")
    print(f"Skipped (already done) : {skipped_existing}")
    print(f"Skipped (corrupted)    : {skipped_corrupted}")
    print(f"Failed                 : {failed}")
    print("=" * 60)


if __name__ == "__main__":
    main()
