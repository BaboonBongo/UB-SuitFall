"""Window-level and event-level metrics with bootstrap uncertainty."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.metrics import precision_recall_curve


def choose_threshold(y, s, rule: str = "max_f1", recall_target: float = 0.9) -> float:
    """Pick the threshold for a model from its training-subject scores."""
    prec, rec, thr = precision_recall_curve(y, s)
    prec, rec = prec[:-1], rec[:-1]
    if rule == "recall_at":
        ok = np.nonzero(rec >= recall_target)[0]
        return float(thr[ok[-1]] if len(ok) else thr[0])
    return float(thr[np.argmax(2 * prec * rec / np.maximum(prec + rec, 1e-12))])


def roc_auc(y, s) -> float:
    """AUC using the Mann-Whitney U formulation."""
    n1 = y.sum()
    n0 = len(y) - n1
    return float((rankdata(s)[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)) if n1 and n0 else np.nan


def pr_auc(y, s) -> float:
    """Average precision for binary labels."""
    n1 = y.sum()
    if not 0 < n1 < len(y):
        return np.nan
    order = np.argsort(-s, kind="mergesort")
    last = np.r_[np.nonzero(np.diff(s[order]))[0], len(s) - 1]
    tp = np.cumsum(y[order])[last]
    return float(np.sum(np.diff(np.r_[0.0, tp / n1]) * tp / (last + 1)))


def window_metrics(y, s, thr: float) -> dict:
    y, s = np.asarray(y, int), np.asarray(s, float)
    p = s >= thr
    tp, fp = int((p & (y == 1)).sum()), int((p & (y == 0)).sum())
    fn, tn = int(y.sum()) - tp, int((y == 0).sum()) - fp
    div = lambda a, b: a / b if b else np.nan
    f1 = div(2 * tp, 2 * tp + fp + fn) if tp + fp + fn else 0.0
    f1_neg = div(2 * tn, 2 * tn + fp + fn) if tn + fp + fn else 0.0
    rec, spec = div(tp, tp + fn), div(tn, tn + fp)
    return dict(n_windows=len(y), n_fall=int(y.sum()), tp=tp, fp=fp, tn=tn, fn=fn,
                accuracy=div(tp + tn, len(y)), precision=div(tp, tp + fp), recall=rec, specificity=spec,
                f1=f1, macro_f1=(f1 + f1_neg) / 2, balanced_accuracy=np.nanmean([rec, spec]),
                roc_auc=roc_auc(y, s), pr_auc=pr_auc(y, s), fpr=div(fp, fp + tn), fnr=div(fn, fn + tp))


def event_metrics(windows: pd.DataFrame, index: pd.DataFrame, fps: float, gap: int) -> tuple[dict, pd.DataFrame]:
    """Merge positive windows into alarm episodes and summarize event-level metrics."""
    info = index.set_index("recording_id")
    rows = []
    for rid, g in windows.groupby("recording_id"):
        g = g.sort_values("start_frame")
        pos = np.nonzero(g["pred"].to_numpy())[0]
        episodes = np.split(pos, np.nonzero(np.diff(pos) > gap + 1)[0] + 1) if len(pos) else []
        r = info.loc[rid]
        st, en = g["start_frame"].to_numpy(), g["end_frame"].to_numpy()
        row = dict(recording_id=rid, subject_id=r.subject_id, activity_name=r.activity_name,
                   category=r.category, hours=r.n_frames / fps / 3600, detected=np.nan, delay_s=np.nan)
        if r.category == "FALL":
            fs, fe = r.fall_start, r.fall_end
            hits = [e for e in episodes if st[e[0]] <= fe and en[e[-1]] >= fs]
            overlap = pos[(st[pos] <= fe) & (en[pos] >= fs)]
            row.update(detected=int(bool(hits)), delay_s=(en[overlap[0]] - fs) / fps if len(overlap) else np.nan,
                       false_alarms=len(episodes) - len(hits))
        else:
            row["false_alarms"] = len(episodes)
        rows.append(row)
    per = pd.DataFrame(rows)
    falls, nonfall = per[per.category == "FALL"], per[per.category == "NON_FALL"]
    hours = nonfall["hours"].sum()
    return dict(event_recall=falls["detected"].mean(), median_delay_s=falls["delay_s"].median(),
                false_alarms_per_hour=nonfall["false_alarms"].sum() / hours if hours else np.nan,
                false_alarms_in_fall_recordings=falls["false_alarms"].sum()), per


def fold_mean(y, s, subjects, thresholds: dict, rows: dict | None = None) -> dict:
    """Aggregate per-subject metrics across folds."""
    per = [window_metrics(y[i], s[i], thresholds[subj]) for subj, i in
           (rows or {k: np.nonzero(subjects == k)[0] for k in thresholds}).items()]
    return {k: float(np.nanmean([d[k] for d in per])) for k in per[0]}


def bootstrap(meta: pd.DataFrame, y, scores: dict, thresholds: dict, metrics: list, n: int, seed: int) -> dict:
    """Bootstrap whole recordings for paired comparison of models."""
    rng = np.random.default_rng(seed)
    meta = meta.reset_index(drop=True)
    strata = {subj: [list(g.groupby("recording_id").indices.values()) for _, g in gs.groupby("category")]
                     for subj, gs in meta.groupby("subject_id")}
    offsets = {subj: [gs.index.to_numpy() for _, gs in meta[meta.subject_id == subj].groupby("category")]
               for subj in strata}
    out = {m: np.full((n, len(metrics)), np.nan) for m in scores}
    for b in range(n):
        rows = {subj: np.concatenate([off[recs[j]] for recs, off in zip(cats, offsets[subj])
                                      for j in rng.integers(0, len(recs), len(recs))])
                for subj, cats in strata.items()}
        for m, s in scores.items():
            fm = fold_mean(y, s, None, thresholds[m], rows)
            out[m][b] = [fm[k] for k in metrics]
    return out


def percentile_ci(a, level: float = 0.95) -> tuple[float, float]:
    a = a[~np.isnan(a)]
    return tuple(np.percentile(a, [50 * (1 - level), 50 * (1 + level)])) if len(a) else (np.nan, np.nan)


def paired_p(diff) -> float:
    """Two-sided bootstrap p-value for paired differences."""
    d = diff[~np.isnan(diff)]
    return float(max(2 * min((d <= 0).mean(), (d >= 0).mean()), 1 / len(d))) if len(d) else np.nan


def holm(p) -> np.ndarray:
    p = np.asarray(p, float)
    order = np.argsort(p)
    adj = np.minimum(1, np.maximum.accumulate((len(p) - np.arange(len(p))) * p[order]))
    out = np.empty_like(p)
    out[order] = adj
    return out
