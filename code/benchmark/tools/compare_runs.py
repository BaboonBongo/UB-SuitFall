#!/usr/bin/env python3
"""Validate the corrected run and compare it with the original run (read-only).

    python tools/compare_runs.py
    python tools/compare_runs.py --original config.yaml --corrected config_foldmask.yaml

Writes to <corrected output_dir>/comparison/:
  validation.md              checklist: mask sources, camera check, extraction, windows,
                             partitions, seeds, result files (PASS / FAIL / CHECK)
  original_vs_corrected.md   comparison tables (overall, per fold, paired tests,
                             generalisation gap, missing silhouettes, ablations, cost)
  original_vs_corrected.csv  overall table
Exit code 1 if any validation item FAILS.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from fallbench import masking  # noqa: E402
from fallbench.config import load_config, md_table  # noqa: E402
from fallbench.data import build_index  # noqa: E402

MODELS = ["RF", "SVM", "LSTM"]
RESULT_FILES = ["summary.csv", "per_fold.csv", "confidence_intervals.csv", "paired_tests.csv",
                "paired_tests_summary.csv", "events.csv", "per_activity.csv", "per_camera.csv",
                "missing_silhouette.csv", "generalisation_gap.csv", "benchmark.csv", "benchmark_info.json"]


def r(cfg, *p) -> Path:
    return Path(cfg["output_dir"]).joinpath(*p)


def validate(corr: dict, orig: dict) -> list[tuple[str, str, str]]:
    checks = []
    add = lambda status, item, detail="": checks.append((status, item, detail))  # noqa: E731
    subjects = sorted(corr["subjects"])
    index = build_index(corr)
    usable = index[index.usable]
    base = Path(corr["features_dir"])

    add("PASS" if masking.mode(corr) == "training" else "FAIL", "corrected config uses mask_source = training",
        masking.mode(corr))
    # camera correspondence
    cc = masking.camera_check_path(corr)
    if cc.exists():
        c = json.loads(cc.read_text())
        sizes = all(v["frame_size_match"] for v in c["cameras"].values())
        add("PASS" if sizes else "FAIL", "frame sizes identical between subjects for every camera",
            ", ".join(f"{k}: {v['raw_sizes']}" for k, v in c["cameras"].items()))
        shifts = {k: v["max_abs_shift_px"] for k, v in c["cameras"].items()}
        add("PASS" if all(v <= c["max_shift_px"] for v in shifts.values()) else "CHECK",
            f"scene shift between subjects <= {c['max_shift_px']} px", json.dumps(shifts))
    else:
        add("FAIL", "camera_check.json present (run tools/mask_diagnostic.py)", str(cc))
    # mask sources
    log = base / "mask_log.csv"
    if log.exists():
        m = pd.read_csv(log)
        leak = [f"{x.fold}/{x.camera}" for x in m.itertuples()
                if x.test_subject in str(x.mask_source).split("+")
                or any(rid.startswith(x.test_subject + "_") for rid in str(x.source_recordings).split(","))]
        add("FAIL" if leak else "PASS", "every fold's masks estimated from training-subject recordings only",
            "leaks: " + ", ".join(leak) if leak else
            "; ".join(f"{x.fold} {x.camera}: from {x.mask_source}, {x.recordings} recordings, "
                      f"{x.sampled_frames} frames, {x.mask_size}" for x in m.itertuples()))
        exp = {(f"test_{t}", cam) for t in subjects for cam in usable.camera.unique()}
        got = set(zip(m.fold, m.camera))
        add("PASS" if exp == got else "FAIL", "one mask per fold and camera", f"missing {sorted(exp - got)}" if exp - got else "")
        pngs = {x.mask_path for x in m.itertuples()}
        add("PASS" if len(pngs) == len(m) and all(Path(p).exists() for p in pngs) else "FAIL",
            "mask files exist and are not shared between folds")
        add("PASS" if all(m.consistent_size) else "FAIL", "sampled source frames have one frame size per mask")
    else:
        add("FAIL", "mask_log.csv present (run.py extract with config_foldmask.yaml)", str(log))
    app = base / "mask_applied_per_recording.csv"
    if app.exists():
        a = pd.read_csv(app)
        skipped = a[~a.mask_applied]
        add("PASS" if skipped.empty else "CHECK", "mask applied to every recording (no size mismatch)",
            "skipped: " + ", ".join(f"{x.fold}/{x.recording_id}" for x in skipped.itertuples()) if len(skipped) else "")
    # extraction completeness
    for t in subjects:
        f = masking.fold_cfg(corr, t)
        missing = [x.recording_id for x in usable.itertuples()
                   if not (Path(f["features_dir"]) / x.subject_id / f"{x.recording_id}.csv").exists()]
        add("FAIL" if missing else "PASS", f"fold test_{t}: features for all {len(usable)} usable recordings",
            ", ".join(missing[:10]))
        if not missing:
            pres = [pd.read_csv(Path(f["features_dir"]) / x.subject_id / f"{x.recording_id}.csv")["present"].mean()
                    for x in usable.itertuples()]
            add("PASS" if min(pres) > 0 else "CHECK", f"fold test_{t}: silhouette found in every recording",
                f"mean {100 * sum(pres) / len(pres):.1f}% of frames, lowest recording {100 * min(pres):.1f}%")
    # only the mask source may differ: training-subject masks and features must equal the original run's
    obase = Path(orig["features_dir"])
    if masking.mode(orig) == "participant" and (obase / "static").exists():
        import cv2
        import numpy as np
        for t in subjects:
            f = masking.fold_cfg(corr, t)
            for trn in f["mask_from"]:
                for cam in sorted(usable.camera.unique()):
                    a_, b_ = (cv2.imread(str(x), cv2.IMREAD_GRAYSCALE) for x in
                              (masking.mask_png(f, t, cam), obase / "static" / f"{trn}_{cam}.png"))
                    same = a_ is not None and b_ is not None and a_.shape == b_.shape and np.array_equal(a_, b_)
                    add("PASS" if same else "FAIL",
                        f"fold test_{t}: {cam} mask identical to the original {trn} mask (same procedure, same recordings)")
                diff = []
                for x in usable[usable.subject_id == trn].itertuples():
                    o, c = (obase / trn / f"{x.recording_id}.csv", Path(f["features_dir"]) / trn / f"{x.recording_id}.csv")
                    if o.exists() and c.exists():
                        if not pd.read_csv(o).equals(pd.read_csv(c)):
                            diff.append(x.recording_id)
                add("FAIL" if diff else "PASS",
                    f"fold test_{t}: training-subject ({trn}) features identical to the original run",
                    ", ".join(diff[:10]))
    # folds, windows, partitions
    tot = {}
    for name, cfg in (("original", orig), ("corrected", corr)):
        n = n_fall = 0
        for t in subjects:
            fj = r(cfg, "folds", f"test_{t}.json")
            if not fj.exists():
                add("FAIL", f"{name}: folds/test_{t}.json present")
                continue
            d = json.loads(fj.read_text())
            n, n_fall = n + d["n_test"], n_fall + d["n_test_fall"]
            if name == "corrected":
                ok = t not in d["train_subjects"] and d["test_subject"] == t
                trials = {tr for v in d["inner_cv_trials"].values() for tr in v}
                ok_cv = all(not tr.startswith(t + "_") for tr in trials)
                src = d.get("mask_estimated_from")
                add("PASS" if ok and ok_cv else "FAIL",
                    f"fold test_{t}: test subject absent from training and inner CV",
                    f"train {d['train_subjects']}, inner-CV trials from {sorted({x.split('_')[0] for x in trials})}")
                add("PASS" if isinstance(src, list) and t not in src else "FAIL",
                    f"fold test_{t}: features built with masks from training subject(s)", str(src))
        tot[name] = (n, n_fall)
    if len(tot) == 2:
        add("PASS" if tot["original"] == tot["corrected"] else "FAIL",
            "window and FALL-window counts identical to original (windows come from labels, not masks)",
            f"original {tot['original']}, corrected {tot['corrected']}")
    pred = r(corr, "predictions.csv")
    if pred.exists():
        p = pd.read_csv(pred)
        combos = p.groupby(["fold", "model"]).seed.nunique()
        add("PASS" if combos.min() == len(corr["tuning"]["seeds"]) and len(combos) == 2 * len(MODELS) else "FAIL",
            f"predictions for {len(MODELS)} models x {len(subjects)} folds x {len(corr['tuning']['seeds'])} seeds",
            combos.to_dict().__repr__())
        wrong = p[p.subject_id != p.fold.str.removeprefix("test_")]
        add("FAIL" if len(wrong) else "PASS", "each fold's predictions contain only its test subject")
        thr = p.groupby(["fold", "model"]).threshold.nunique()
        add("PASS" if thr.max() == 1 else "FAIL", "one training-derived threshold per fold and model")
    else:
        add("FAIL", "corrected predictions.csv present")
    for f in RESULT_FILES:
        add("PASS" if r(corr, "results", f).exists() else "FAIL", f"results/{f} regenerated")
    abl = r(corr, "results", "ablations.csv")
    add("PASS" if abl.exists() else "CHECK", "results/ablations.csv regenerated (run.py ablations)")
    if pred.exists():
        t0 = pred.stat().st_mtime
        stale = [str(x.relative_to(r(corr))) for d in ("results", "figures", "tables")
                 for x in r(corr, d).glob("*") if x.is_file() and x.stat().st_mtime < t0]
        add("FAIL" if stale else "PASS", "results, figures and tables newer than the corrected predictions",
            ", ".join(stale[:10]))
    return checks


def summary(cfg) -> pd.DataFrame:
    return pd.read_csv(r(cfg, "results", "summary.csv"), index_col=0).reindex(MODELS)


def compare(orig: dict, corr: dict) -> tuple[pd.DataFrame, list[str]]:
    rows = []
    subjects = sorted(corr["subjects"])
    for name, cfg in (("original", orig), ("corrected", corr)):
        n = n_fall = 0
        for t in subjects:
            d = json.loads(r(cfg, "folds", f"test_{t}.json").read_text())
            n, n_fall = n + d["n_test"], n_fall + d["n_test_fall"]
        rows += [dict(metric="Number of windows", run=name, value=n), dict(metric="FALL windows", run=name, value=n_fall)]
    so, sc = summary(orig), summary(corr)
    metrics = [("f1", "F1"), ("pr_auc", "PR-AUC"), ("roc_auc", "ROC-AUC"), ("precision", "Precision"),
               ("recall", "Recall"), ("specificity", "Specificity"), ("balanced_accuracy", "Balanced accuracy"),
               ("event_recall", "Event recall"), ("median_delay_s", "Median detection delay (s)"),
               ("false_alarms_per_hour", "False alarms per hour"),
               ("false_alarms_in_fall_recordings", "False alarms in fall recordings")]
    for m in MODELS:
        for k, lab in metrics:
            for name, s in (("original", so), ("corrected", sc)):
                rows.append(dict(metric=f"{m} {lab}", run=name, value=s.loc[m, f"{k}_mean"], sd=s.loc[m, f"{k}_std"]))
    long = pd.DataFrame(rows)
    wide = long.pivot_table(index="metric", columns="run", values="value", sort=False)[["original", "corrected"]]
    sd = long.pivot_table(index="metric", columns="run", values="sd", sort=False)
    wide["absolute_difference"] = wide.corrected - wide.original
    wide["sd_original"] = sd.get("original")
    wide["sd_corrected"] = sd.get("corrected")
    lines = ["# Original vs corrected pipeline", "",
             "Original: static mask per participant from all of that participant's recordings (transductive). "
             "Corrected: per fold, mask from the training participant only. Values are means over seeds of the "
             "mean over both folds; SD over seeds (SVM is deterministic). Differences are not tested here; see the "
             "paired tests below for the model comparisons within each run.", "",
             md_table(wide, "{:.3f}"), ""]

    def side(name, file, idx, cols, agg=None):
        o, c = pd.read_csv(r(orig, "results", file)), pd.read_csv(r(corr, "results", file))
        if agg:
            o, c = agg(o), agg(c)
        t = o.set_index(idx)[cols].join(c.set_index(idx)[cols], lsuffix="_original", rsuffix="_corrected")
        return [f"## {name}", "", md_table(t, "{:.3f}"), ""]

    lines += side("Per fold (mean over seeds)", "per_fold.csv", ["fold", "model"],
                  ["recall", "precision", "f1", "pr_auc", "roc_auc", "false_alarms_per_hour"],
                  lambda d: d.groupby(["fold", "model"], as_index=False)[
                      ["recall", "precision", "f1", "pr_auc", "roc_auc", "false_alarms_per_hour"]].mean())
    lines += side("Paired model comparisons (seeds significant out of seeds)", "paired_tests_summary.csv",
                  ["metric", "model_a", "model_b"], ["mean_diff", "seeds_significant"])
    lines += side("Within-subject vs cross-subject PR-AUC", "generalisation_gap.csv", ["fold", "model"],
                  ["within_subject_cv_pr_auc", "cross_subject_test_pr_auc", "gap"])
    lines += side("Missing-silhouette robustness (F1, mean of folds)", "missing_silhouette.csv",
                  ["model", "missing_rate"], ["f1"],
                  lambda d: d.groupby(["model", "missing_rate"], as_index=False)["f1"].mean())
    if r(orig, "results", "ablations.csv").exists() and r(corr, "results", "ablations.csv").exists():
        lines += side("Ablations", "ablations.csv", ["ablation", "model"],
                      ["f1_mean", "pr_auc_mean", "event_recall_mean", "false_alarms_per_hour_mean"])
    bo, bc = pd.read_csv(r(orig, "results", "benchmark.csv")), pd.read_csv(r(corr, "results", "benchmark.csv"))
    for b in (bo, bc):
        if "estimated_end_to_end_ms" not in b:
            b["estimated_end_to_end_ms"] = b["end_to_end_ms"]
    t = bo.set_index(["model", "device"])[["latency_ms", "preprocessing_ms", "estimated_end_to_end_ms"]].join(
        bc.set_index(["model", "device"])[["latency_ms", "preprocessing_ms", "estimated_end_to_end_ms"]],
        lsuffix="_original", rsuffix="_corrected")
    lines += ["## Computational cost (timings vary between runs of the same code)", "", md_table(t, "{:.3f}"), ""]
    return wide, lines


def main():
    root = Path(__file__).resolve().parents[1]
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--original", default=root / "config.yaml")
    ap.add_argument("--corrected", default=root / "config_foldmask.yaml")
    a = ap.parse_args()
    orig, corr = load_config(a.original), load_config(a.corrected)
    dest = r(corr, "comparison")
    dest.mkdir(parents=True, exist_ok=True)
    checks = validate(corr, orig)
    v = ["# Validation of the corrected run", "", "| Status | Check | Detail |", "|---|---|---|"]
    v += [f"| {s} | {i} | {d.replace('|', '/')} |" for s, i, d in checks]
    (dest / "validation.md").write_text("\n".join(v))
    print("\n".join(v))
    fails = [c for c in checks if c[0] == "FAIL"]
    try:
        wide, lines = compare(orig, corr)
        wide.to_csv(dest / "original_vs_corrected.csv")
        (dest / "original_vs_corrected.md").write_text("\n".join(lines))
        print("\n" + "\n".join(lines))
    except FileNotFoundError as e:
        print(f"\ncomparison not written, a result file is missing: {e}")
    print(f"\n-> {dest}")
    if fails:
        print(f"\n{len(fails)} validation item(s) FAILED - do not use these results until fixed.")
        sys.exit(1)


if __name__ == "__main__":
    main()
