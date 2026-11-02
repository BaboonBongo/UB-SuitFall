#!/usr/bin/env python3
"""Extract frames from video files at a fixed FPS."""
import argparse
import subprocess
from pathlib import Path

FPS = 10

CAMERA_DIRS = {
    "CCTV-A": ["CCTV-A", "cctv-a", "Cam_A", "cam_a"],
    "CCTV-B": ["CCTV-B", "cctv-b", "Cam_B", "cam_b"],
}


def find_camera_folders(root: Path) -> list:
    found = []
    for camera, names in CAMERA_DIRS.items():
        name = next((n for n in names if (root / n).is_dir()), None)
        if name is None:
            print(f"[WARNING] No folder for {camera} in {root} (looked for {', '.join(names)})")
        else:
            found.append(name)
    return found


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", required=True, type=Path, help="participant folder containing the camera folders")
    ap.add_argument("--skip", nargs="*", default=[], metavar="VIDEO", help="video file names to skip")
    ap.add_argument("--fps", type=float, default=FPS, help=f"extraction rate (default {FPS})")
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
                subprocess.run(command, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
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
