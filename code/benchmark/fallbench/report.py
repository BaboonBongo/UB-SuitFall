"""Generate dataset inspection and result summaries."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from sklearn.metrics import precision_recall_curve, roc_curve

from . import masking
from .config import md_table, out
from .data import build_index, list_frames

MODELS = ["RF", "SVM", "LSTM"]
COLORS = {"RF": "#2a6f97", "SVM": "#c8553d", "LSTM": "#588157"}


def inspect(cfg: dict) -> None:
    fps, W, S = cfg["fps"], cfg["windows"]["length"], cfg["windows"]["stride"]
    index = build_index(cfg)
    ok = index[index.usable]
    lines = ["# Dataset inspection", "", "## Recordings", "",
             md_table(index.groupby(["subject_id", "camera", "category"])["usable"].agg(["size", "sum"]).rename(columns={"size": "expected", "sum": "usable"})), ""]
    lines += [f"- excluded: {r.recording_id} ({r.activity_name}): {r.issue}" for r in index[~index.usable].itertuples()]
    coverage = ok.pivot_table(index="activity_name", columns="subject_id", values="recording_id", aggfunc="count", fill_value=0)
    lines += ["", "## Usable recordings per activity", "", md_table(coverage, "{:.0f}"), ""]
    issues = [f"'{a}' has no usable recording for {s}: a model trained on {s} never sees it."
              for a, row in coverage.iterrows() for s, v in row.items() if v == 0]

    frames, falls, n_win, n_pos, present = 0, 0, 0, 0, []
    for r in ok.itertuples():
        y = pd.read_csv(r.label_file)["label"].to_numpy()
        n_files = len(list_frames(r.frames_dir))
        if not n_files == len(y) == r.n_frames:
            issues.append(f"{r.recording_id}: {r.n_frames} frames in fall_labels.csv, {n_files} images, {len(y)} label rows")
        frames, falls = frames + len(y), falls + y.sum()
        frac = y[np.arange(0, len(y) - W + 1, S)[:, None] + np.arange(W)].mean(1) if len(y) >= W else []
        n_win, n_pos = n_win + len(frac), n_pos + int((np.asarray(frac) >= cfg["windows"]["min_fall_fraction"]).sum())
        for test_subject in (sorted(cfg["subjects"]) if masking.mode(cfg) == "training" else [None]):
            fcfg = masking.fold_cfg(cfg, test_subject) if test_subject else cfg
            f = Path(fcfg["features_dir"]) / r.subject_id / f"{r.recording_id}.csv"
            if f.exists():
                present.append(dict(mask=("from " + masking.source_tag(fcfg["mask_from"])) if test_subject else "own subject",
                                    subject_id=r.subject_id, camera=r.camera,
                                    recording_id=r.recording_id, present=pd.read_csv(f)["present"].mean()))
    fall = ok[ok.category == "FALL"]
    timing = pd.DataFrame({"start to impact (s)": (fall.impact - fall.fall_start) / fps,
                           "impact to end (s)": (fall.fall_end - fall.impact) / fps,
                           "whole fall (s)": (fall.fall_end - fall.fall_start + 1) / fps}).describe().T
    lines += ["## Frames, labels and windows", "",
              f"- {len(ok)} usable recordings, {frames} frames ({frames / fps / 60:.1f} min), FALL frames {100 * falls / frames:.1f}%",
              f"- windows of {W} frames ({W / fps:g} s), stride {S} ({S / fps:g} s): {n_win} windows, {n_pos} FALL ({100 * n_pos / max(n_win, 1):.1f}%) - accuracy alone would be dominated by NON_FALL",
              "", "Fall timing from the manual labels:", "",
              md_table(timing[["mean", "std", "min", "50%", "max"]], "{:.2f}"), ""]
    if present:
        p = pd.DataFrame(present)
        lines += ["## Silhouette detection (% of frames)", "",
                  md_table(100 * p.groupby(["mask", "subject_id", "camera"])["present"].mean().to_frame(), "{:.1f}"), ""]
        low = p[p.present < 0.8]
        if len(low):
            issues.append(f"silhouette found in < 80% of frames: {', '.join(low.recording_id)}")
    else:
        lines += ["Silhouettes not extracted yet - run `python run.py extract`.", ""]
    lines += ["## Issues", ""] + [f"- {i}" for i in issues or ["none"]]
    out(cfg, "inspection.md").write_text("\n".join(lines))
    print("\n".join(lines))


def _save(fig, cfg, name):
    path = out(cfg, "figures", name)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def figures(cfg: dict) -> None:
    preds = pd.read_csv(out(cfg, "predictions.csv"))
    p0 = preds[preds.seed == preds.seed.min()]
    fig, axes = plt.subplots(1, 3, figsize=(9.6, 3.2))
    for ax, m in zip(axes, MODELS):
        g = p0[p0.model == m]
        cm = pd.crosstab(g.label, g.pred).reindex(index=[0, 1], columns=[0, 1], fill_value=0).to_numpy()
        ax.imshow(cm, cmap="Blues")
        for (i, j), v in np.ndenumerate(cm):
            ax.text(j, i, v, ha="center", va="center", color="white" if v > cm.max() / 2 else "black")
        ax.set(title=m, xticks=[0, 1], yticks=[0, 1], xticklabels=["NON_FALL", "FALL"],
               yticklabels=["NON_FALL", "FALL"], xlabel="Predicted")
    axes[0].set_ylabel("True")
    fig.suptitle("Window-level confusion matrices, both test subjects", fontsize=10)
    _save(fig, cfg, "confusion_matrices.png")

    folds = sorted(p0.fold.unique())
    for kind in ("roc", "pr"):
        fig, axes = plt.subplots(1, len(folds), figsize=(4.6 * len(folds), 4.2), squeeze=False)
        for ax, fold in zip(axes[0], folds):
            for m in MODELS:
                g = p0[(p0.model == m) & (p0.fold == fold)]
                x, y = (roc_curve(g.label, g.score)[:2] if kind == "roc"
                        else precision_recall_curve(g.label, g.score)[1::-1])
                ax.plot(x, y, label=m, color=COLORS[m])
            ax.set(title=fold.replace("test_", "Test subject "),
                   xlabel="False-positive rate" if kind == "roc" else "Recall (FALL)",
                   ylabel="Recall (FALL)" if kind == "roc" else "Precision (FALL)")
            ax.grid(alpha=0.3)
            ax.legend()
        _save(fig, cfg, f"{kind}_curves.png")

    bench = out(cfg, "results", "benchmark.csv")
    if bench.exists():
        b = pd.read_csv(bench)
        s = pd.read_csv(out(cfg, "results", "summary.csv"), index_col=0)
        fig, ax = plt.subplots(figsize=(5.5, 4))
        for r in b.itertuples():
            ax.errorbar(r.latency_ms, s.loc[r.model, "f1_mean"], s.loc[r.model, "f1_std"], fmt="o",
                        color=COLORS[r.model], capsize=3)
            ax.annotate(f"{r.model} ({r.device})", (r.latency_ms, s.loc[r.model, "f1_mean"]), fontsize=8,
                        xytext=(4, 4), textcoords="offset points")
        ax.set(xscale="log", xlabel="model latency per window (ms, median, log)", ylabel="F1 (FALL)")
        ax.grid(alpha=0.3)
        _save(fig, cfg, "latency_vs_f1.png")

    miss = out(cfg, "results", "missing_silhouette.csv")
    if miss.exists():
        d = pd.read_csv(miss).groupby(["model", "missing_rate"])["f1"].mean().unstack(0)
        ax = (100 * d.set_axis(100 * d.index)).plot(marker="o", figsize=(5, 3.6),
                                                     color=[COLORS[m] for m in d.columns])
        ax.set(xlabel="test frames without silhouette (%)", ylabel="F1 (FALL) x 100")
        _save(ax.figure, cfg, "missing_silhouette.png")


def tables(cfg: dict) -> None:
    s = pd.read_csv(out(cfg, "results", "summary.csv"), index_col=0).reindex(MODELS)
    pm = lambda k, d=3: [f"{s.loc[m, k + '_mean']:.{d}f} ± {np.nan_to_num(s.loc[m, k + '_std']):.{d}f}"  # noqa: E731
                         for m in MODELS]
    cols = {"Precision": "precision", "Recall": "recall", "Specificity": "specificity", "F1": "f1",
            "Bal. acc.": "balanced_accuracy", "ROC-AUC": "roc_auc", "PR-AUC": "pr_auc",
            "Event recall": "event_recall", "False alarms/h": "false_alarms_per_hour"}
    perf = pd.DataFrame({"Model": MODELS, **{c: pm(k, 1 if k == "false_alarms_per_hour" else 3)
                                             for c, k in cols.items()}})
    perf.to_csv(out(cfg, "tables", "performance.csv"), index=False)
    lines = ["# Paper tables", "", "Mean ± SD over seeds of the mean over both leave-one-subject-out folds; "
             "window level except event recall and false alarms per hour.", "", md_table(perf, index=False), ""]
    ci = pd.read_csv(out(cfg, "results", "confidence_intervals.csv"))
    ci["value"] = [f"{e:.3f} [{lo:.3f}, {hi:.3f}]" for e, lo, hi in zip(ci.estimate, ci.ci_low, ci.ci_high)]
    lines += ["95% recording-level bootstrap CIs (first seed):", "",
              md_table(ci.pivot(index="model", columns="metric", values="value").reindex(MODELS)), ""]
    bench = out(cfg, "results", "benchmark.csv")
    if bench.exists():
        b = pd.read_csv(bench)
        cost = pd.DataFrame({"Model": b.model + " (" + b.device + ")", "Structure": b.structure,
                             "Train (s)": b.train_seconds.round(2), "Latency (ms)": b.latency_ms.round(3),
                             "p95 (ms)": b.latency_p95_ms.round(3), "Throughput (win/s)": b.throughput_per_s.round().astype(int),
                             "Memory (MB)": b.memory_increase_mb.round(1), "File (MB)": b.model_file_mb.round(3),
                             "Ops/window": [f"{v:,.0f} {u}" for v, u in zip(b.ops_per_window, b.ops_unit)],
                             "Estimated end-to-end (ms)": (b.estimated_end_to_end_ms if "estimated_end_to_end_ms" in b
                                                           else b.end_to_end_ms).round(2)})
        cost.to_csv(out(cfg, "tables", "computational_cost.csv"), index=False)
        lines += ["Computational cost (first fold, first seed; protocol in results/benchmark_info.json). "
                  "Frame reading = stored JPEG frames (disk read + decode + resize), not live video decoding; "
                  "static-mask generation is offline and excluded; estimated end-to-end = sum of separately "
                  "measured medians (stride x per-frame preprocessing + window features + model latency).", "",
                  md_table(cost, index=False), ""]
    out(cfg, "tables", "paper_tables.md").write_text("\n".join(lines))
    perf.to_latex(out(cfg, "tables", "performance.tex"), index=False)


def reports(cfg: dict) -> None:
    res = lambda n: out(cfg, "results", n)  # noqa: E731
    lines = ["# Reproducibility", ""]
    if res("benchmark_info.json").exists():
        lines += ["```json", res("benchmark_info.json").read_text(), "```", ""]
    for f in sorted(Path(cfg["output_dir"]).glob("folds/*.json")):
        d = json.loads(f.read_text())
        lines.append(f"- {f.stem}: train {d['train_subjects']}, test {d['test_subject']}, {d['n_train']} train "
                     f"windows ({d['n_train_fall']} FALL), {d['n_test']} test windows ({d['n_test_fall']} FALL); "
                     f"inner-CV trials in folds/{f.name}")
    lines += ["", "Selected hyperparameters and thresholds (training subjects only):", "", "```json",
              out(cfg, "selected.json").read_text(), "```", "", "Configuration:", "", "```yaml",
              yaml.safe_dump({k: v for k, v in cfg.items() if k not in ("overrides",)}, sort_keys=False), "```"]
    out(cfg, "reports", "reproducibility.md").write_text("\n".join(lines))

    s = pd.read_csv(res("summary.csv"), index_col=0).reindex(MODELS)
    lines = ["# Results summary (generated from the result files)", ""]
    for k in ["f1", "pr_auc", "balanced_accuracy", "recall", "specificity", "false_alarms_per_hour"]:
        vals = s[f"{k}_mean"]
        best = vals.idxmin() if k == "false_alarms_per_hour" else vals.idxmax()
        lines.append(f"- {k}: " + ", ".join(f"{m} {vals[m]:.3f} ± {np.nan_to_num(s.loc[m, k + '_std']):.3f}"
                                            for m in MODELS) + f" (best point estimate: {best})")
    t = pd.read_csv(res("paired_tests_summary.csv"))
    lines += ["", "Paired comparisons (Holm-corrected; 'significant' only if p < 0.05 in every seed):", ""]
    lines += [f"- {r.metric}: {r.model_a} - {r.model_b} = {r.mean_diff:+.3f}; significant in "
              f"{r.seeds_significant}/{r.seeds} seeds" for r in t.itertuples()]
    act = pd.read_csv(res("per_activity.csv"))
    lines += ["", "False-positive window rate per non-fall activity (first seed):", "",
              md_table(act[act.category == "NON_FALL"].pivot(index="activity_name", columns="model",
                                                             values="false_positive_rate")[MODELS]), ""]
    for name, title in [("generalisation_gap.csv", "Within-subject CV vs cross-subject test PR-AUC"),
                        ("missing_silhouette.csv", "Missing-silhouette robustness"),
                        ("ablations.csv", "Ablations (reduced search; compare with `reference`)")]:
        if res(name).exists():
            d = pd.read_csv(res(name))
            if name == "missing_silhouette.csv":
                d = d.groupby(["model", "missing_rate"])[["f1", "recall", "specificity", "pr_auc"]].mean().reset_index()
            if name == "ablations.csv":
                d = d[d.model.isin(MODELS)][["ablation", "model", "f1_mean", "pr_auc_mean", "recall_mean",
                                            "specificity_mean", "false_alarms_per_hour_mean"]]
            lines += [f"{title}:", "", md_table(d, index=False), ""]
    out(cfg, "reports", "results_summary.md").write_text("\n".join(lines))


def run(cfg: dict) -> None:
    figures(cfg)
    tables(cfg)
    reports(cfg)
    print(f"figures, tables, reports -> {cfg['output_dir']}")
