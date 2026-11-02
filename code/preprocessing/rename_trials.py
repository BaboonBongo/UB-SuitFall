#!/usr/bin/env python3
import csv
import re
import sys
from pathlib import Path
from typing import List, Tuple

ACTIVITY_NUMBER_REMAP = {11: 10}
SOURCE_NAME_PATTERN = re.compile(r"^(\d+)_(\d+)$")
LOG_FILENAME = "rename_log.csv"


def planned_renames(folder: Path) -> Tuple[List[Tuple[Path, Path]], List[str]]:
    """Return planned renames for entries named <N>_<T> inside a folder."""
    renames: List[Tuple[Path, Path]] = []
    skipped: List[str] = []

    for entry in sorted(folder.iterdir(), key=lambda p: p.name):
        stem = entry.stem if entry.is_file() else entry.name
        suffix = entry.suffix if entry.is_file() else ""
        match = SOURCE_NAME_PATTERN.match(stem)
        if not match:
            if entry.name != LOG_FILENAME and not entry.name.startswith("Activity"):
                skipped.append(entry.name)
            continue
        activity = int(match.group(1))
        trial = int(match.group(2))
        activity = ACTIVITY_NUMBER_REMAP.get(activity, activity)
        target = folder / f"Activity{activity}Trial{trial}{suffix}"
        renames.append((entry, target))

    return renames, skipped


def check_conflicts(renames: List[Tuple[Path, Path]]) -> List[str]:
    problems = []
    targets = [t for _, t in renames]
    for t in set(targets):
        if targets.count(t) > 1:
            sources = [s.name for s, tt in renames if tt == t]
            problems.append(f"{', '.join(sources)} would all become {t.name}")
    for s, t in renames:
        if t.exists():
            problems.append(f"{t.name} already exists (from {s.name}) - not overwriting")
    return problems


def apply_renames(folder: Path, renames: List[Tuple[Path, Path]]) -> None:
    log_path = folder / LOG_FILENAME
    new_log = not log_path.exists()
    with log_path.open("a", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        if new_log:
            writer.writerow(["old_name", "new_name"])
        for source, target in renames:
            source.rename(target)
            writer.writerow([source.name, target.name])


def undo_renames(folder: Path, apply: bool) -> int:
    log_path = folder / LOG_FILENAME
    if not log_path.exists():
        print(f"  no {LOG_FILENAME} here - nothing to undo")
        return 0
    rows = list(csv.DictReader(log_path.open(encoding="utf-8")))
    count = 0
    for row in reversed(rows):
        current, original = folder / row["new_name"], folder / row["old_name"]
        if not current.exists():
            print(f"  [missing] {current.name} (skipped)")
            continue
        if original.exists():
            print(f"  [conflict] {original.name} already exists (skipped)")
            continue
        print(f"  {current.name}  ->  {original.name}")
        if apply:
            current.rename(original)
        count += 1
    if apply:
        log_path.rename(log_path.with_name(LOG_FILENAME + ".undone"))
    return count


def main() -> None:
    args = sys.argv[1:]
    apply = "--apply" in args
    undo = "--undo" in args
    folders = [Path(a) for a in args if not a.startswith("--")]

    if not folders:
        print("Usage: python rename_trials.py <folder> [<folder> ...] [--apply | --undo]")
        sys.exit(1)

    total = 0
    blocked = False
    for folder in folders:
        print(f"\n{folder}")
        if not folder.is_dir():
            print("  [ERROR] not a folder")
            blocked = True
            continue

        if undo:
            total += undo_renames(folder, apply=True)
            continue

        renames, skipped = planned_renames(folder)
        problems = check_conflicts(renames)
        for source, target in renames:
            print(f"  {source.name:<16} ->  {target.name}")
        for name in skipped:
            print(f"  [ignored] {name} (not <N>_<T> format)")
        if problems:
            blocked = True
            for p in problems:
                print(f"  [CONFLICT] {p}")
            print("  Nothing renamed in this folder because of the conflict(s) above.")
            continue
        if apply and renames:
            apply_renames(folder, renames)
        total += len(renames)

    print()
    if undo:
        print(f"Reverted {total} item(s).")
    elif apply:
        print(f"Renamed {total} item(s). Log: {LOG_FILENAME} in each folder (use --undo to revert).")
    else:
        print(f"DRY RUN: {total} item(s) would be renamed. Re-run with --apply to do it.")
    if blocked:
        sys.exit(2)


if __name__ == "__main__":
    main()
