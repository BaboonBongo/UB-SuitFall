"""Evaluate model predictions on the held-out test set."""
from __future__ import annotations

import itertools
import json

import numpy as np
import pandas as pd

from . import masking
from .config import out
from .data import build_index
from .features import build_windows
from .metrics import (bootstrap, event_metrics, fold_mean, holm, paired_p, percentile_ci, window_metrics)
from .models import Model

MODELS = ["RF", "SVM", "LSTM"]
WINDOW = ["accuracy", "precision", "recall", "specificity", "f1", "macro_f1", "balanced_accuracy",
          "roc_auc", "pr_auc", "fpr", "fnr"]
EVENT = ["event_recall", "median_delay_s", "false_alarms_per_hour", "false_alarms_in_fall_recordings"]


def _aligned(preds: pd.DataFrame, seed: int):
    """Align every model’s window scores for one seed on the same ordering."""
    p = preds[preds.seed == seed].sort_values(["model", "fold", "recording_id", "start_frame"])
    ref = p[p.model == MODELS[0]].reset_index(drop=True)
    scores = {m: p[p.model == m]["score"].to_numpy() for m in MODELS}
    thr = {m: p[p.model == m].groupby("subject_id")["threshold"].first().to_dict() for m in MODELS}
    return ref, scores, thr


def run(cfg: dict, robustness: bool = True) -> None:
    preds = pd.read_csv(out(cfg, "predictions.csv"))
    index = build_index(cfg)
    ev = cfg["evaluation"]
    res = lambda name: out(cfg, "results", name)

    rows, events = [], []
    for (fold, model, seed), g in preds.groupby(["fold", "model", "seed"]):
        em, per = event_metrics(g, index, cfg["fps"], ev["alarm_gap"])
        rows.append(dict(fold=fold, model=model, seed=seed,
                         **window_metrics(g["label"], g["score"], g["threshold"].iloc[0]), **em))
        events.append(per.assign(fold=fold, model=model, seed=seed))
    per_fold = pd.DataFrame(rows)
    per_fold.to_csv(res("per_fold.csv"), index=False)
    pd.concat(events).to_csv(res("events.csv"), index=False)

    by_seed = per_fold.groupby(["model", "seed"])[WINDOW + EVENT].mean()
    summary = by_seed.groupby("model").agg(["mean", "std"])
    summary.columns = [f"{a}_{b}" for a, b in summary.columns]
    summary.to_csv(res("summary.csv"))

    metrics, seeds = ev["metrics"], sorted(preds.seed.unique())
    ci_rows, tests = [], []
    for seed in seeds:
        ref, scores, thr = _aligned(preds, seed)
        y, subj = ref["label"].to_numpy(), ref["subject_id"].to_numpy()
        boots = bootstrap(ref, y, scores, thr, metrics, ev["bootstrap"], cfg["seed"] + int(seed))
        point = {m: fold_mean(y, scores[m], subj, thr[m]) for m in MODELS}
        if seed == seeds[0]:
            for m in MODELS:
                for j, k in enumerate(metrics):
                    ci_rows.append(dict(model=m, metric=k, estimate=point[m][k],
                                        **dict(zip(["ci_low", "ci_high"], percentile_ci(boots[m][:, j])))))
        for j, k in enumerate(metrics):
            pairs = list(itertools.combinations(MODELS, 2))
            diffs = [boots[a][:, j] - boots[b][:, j] for a, b in pairs]
            p_adj = holm([paired_p(d) for d in diffs])
            for (a, b), d, pa in zip(pairs, diffs, p_adj):
                lo, hi = percentile_ci(d)
                tests.append(dict(seed=seed, metric=k, model_a=a, model_b=b, diff=point[a][k] - point[b][k],
                                  ci_low=lo, ci_high=hi, p_holm=pa, significant=pa < 0.05))
    pd.DataFrame(ci_rows).to_csv(res("confidence_intervals.csv"), index=False)
    tests = pd.DataFrame(tests)
    tests.to_csv(res("paired_tests.csv"), index=False)
    tests.groupby(["metric", "model_a", "model_b"]).agg(
        mean_diff=("diff", "mean"), seeds=("seed", "count"), seeds_significant=("significant", "sum"),
        seeds_a_better=("diff", lambda d: int((d > 0).sum()))).to_csv(res("paired_tests_summary.csv"))

    p0 = preds[preds.seed == seeds[0]]
    pd.DataFrame([dict(model=m, activity_name=a, category=c, windows=len(g), fall_windows=g.label.sum(),
                       false_positive_rate=((g.pred == 1) & (g.label == 0)).sum() / max(1, (g.label == 0).sum()),
                       missed_fall_windows=((g.pred == 0) & (g.label == 1)).sum())
                  for (m, a, c), g in p0.groupby(["model", "activity_name", "category"])]) \
        .to_csv(res("per_activity.csv"), index=False)
    pd.DataFrame([dict(model=m, fold=f, camera=c, **window_metrics(g.label, g.score, g.threshold.iloc[0]))
                  for (m, f, c), g in p0.groupby(["model", "fold", "camera"])]) .to_csv(res("per_camera.csv"), index=False)

    if robustness:
        _robustness(cfg, index)
    print(f"results -> {out(cfg, 'results')}")


def _robustness(cfg: dict, index: pd.DataFrame) -> None:
    """Check robustness to missing silhouettes and distribution shift."""
    selected = json.loads(out(cfg, "selected.json").read_text())
    log = pd.read_csv(out(cfg, "training_log.csv"))
    log = log[log.seed == log.seed.min()]
    per_fold = pd.read_csv(out(cfg, "results", "per_fold.csv"))
    rows, gap = [], []
    for fold, models in selected.items():
        for rate in [0.0] + cfg["ablations"]["missing_rates"]:
            subj = fold.removeprefix("test_")
            test = build_windows(masking.fold_cfg(cfg, subj), [subj], index, blank_rate=rate)
            for m, sel in models.items():
                model = Model.load(log[(log.fold == fold) & (log.model == m)]["path"].iloc[0])
                s = model.score(test["X_seq" if model.input == "seq" else "X_sum"])
                wm = window_metrics(test["meta"]["label"], s, sel["threshold"])
                rows.append(dict(fold=fold, model=m, missing_rate=rate, f1=wm["f1"], recall=wm["recall"],
                                 specificity=wm["specificity"], pr_auc=wm["pr_auc"]))
        for m, sel in models.items():
            test_ap = per_fold[(per_fold.fold == fold) & (per_fold.model == m)]["pr_auc"].mean()
            gap.append(dict(fold=fold, model=m, within_subject_cv_pr_auc=sel["cv_pr_auc"],
                            cross_subject_test_pr_auc=test_ap, gap=sel["cv_pr_auc"] - test_ap))
    pd.DataFrame(rows).to_csv(out(cfg, "results", "missing_silhouette.csv"), index=False)
    pd.DataFrame(gap).to_csv(out(cfg, "results", "generalisation_gap.csv"), index=False)
