"""Leave-one-subject-out experiment: tuning, final training, test predictions.

For each fold (one subject held out for testing):
  1. search each model's hyperparameters with StratifiedGroupKFold inside the
     training subject (groups = trial, so both camera views of one fall stay
     together), scored by PR-AUC;
  2. choose the decision threshold on the out-of-fold scores of the best
     configuration;
  3. retrain the best configuration on the whole training subject once per
     seed and score every window of the held-out subject.
The test subject's windows are used in step 3 only. With mask_source = training
(masking.py), the static background mask used for both subjects in a fold is
estimated from the training subject only, so no test-subject frame is used
before step 3.
"""
from __future__ import annotations

import json
import math

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score
from sklearn.model_selection import ParameterGrid, ParameterSampler, StratifiedGroupKFold

from . import masking
from .config import out, save_json
from .data import build_index
from .features import build_windows
from .metrics import choose_threshold
from .models import NAMES, make_model


def candidates(spec: dict, scale: float, seed: int) -> list[dict]:
    grid = ParameterGrid(spec["space"])
    n = math.ceil((spec["n_iter"] or len(grid)) * scale)
    return list(grid) if n >= len(grid) else list(ParameterSampler(spec["space"], n, random_state=seed))


def tune(name: str, X, y, groups, splits, cfg: dict, scale: float):
    """Best params, their out-of-fold scores, and a table of every candidate."""
    rows, best = [], None
    for params in candidates(cfg["models"][name], scale, cfg["seed"]):
        oof = np.zeros(len(y))
        aps = []
        for tr, va in splits:
            oof[va] = make_model(name, params, cfg, cfg["seed"]).fit(X[tr], y[tr], groups[tr]).score(X[va])
            aps.append(average_precision_score(y[va], oof[va]))
        rows.append(dict(model=NAMES[name], params=json.dumps(params), cv_pr_auc=np.mean(aps), cv_sd=np.std(aps)))
        if best is None or np.mean(aps) > best[0]:
            best = (np.mean(aps), params, oof)
    return best[1], best[2], pd.DataFrame(rows)


def run(cfg: dict, scale: float = 1.0, seeds: list | None = None) -> None:
    seeds = seeds or cfg["tuning"]["seeds"]
    t = cfg["tuning"]
    index = build_index(cfg)
    subjects = sorted(cfg["subjects"])
    preds, tuning, log, selected = [], [], [], {}
    for test_subject in subjects:
        fold = f"test_{test_subject}"
        fcfg = masking.fold_cfg(cfg, test_subject)    # features made with this fold's masks
        train = build_windows(fcfg, [s for s in subjects if s != test_subject], index)
        test = build_windows(fcfg, [test_subject], index)
        y, groups = train["meta"]["label"].to_numpy(), train["meta"]["trial_uid"].to_numpy()
        splits = list(StratifiedGroupKFold(t["inner_folds"], shuffle=True, random_state=cfg["seed"])
                      .split(y, y, groups))
        print(f"\n=== {fold}: {len(y)} training windows ({y.sum()} FALL), "
              f"{len(test['meta'])} test windows ({test['meta']['label'].sum()} FALL) ===")
        save_json(dict(train_subjects=[s for s in subjects if s != test_subject], test_subject=test_subject,
                       n_train=len(y), n_train_fall=y.sum(), n_test=len(test["meta"]),
                       n_test_fall=test["meta"]["label"].sum(), mask_source_mode=masking.mode(cfg),
                       mask_estimated_from=fcfg.get("mask_from", "each subject's own recordings"),
                       features_dir=fcfg["features_dir"], summary_features=train["sum_names"],
                       sequence_features=train["seq_names"],
                       inner_cv_trials={k: sorted(set(groups[va])) for k, (_, va) in enumerate(splits)}),
                  out(cfg, "folds", f"{fold}.json"))
        selected[fold] = {}
        for name in ("rf", "svm", "lstm"):
            key = "X_seq" if name == "lstm" else "X_sum"
            print(f"  {NAMES[name]}: tuning ...", flush=True)
            params, oof, table = tune(name, train[key], y, groups, splits, cfg, scale)
            thr = choose_threshold(y, oof, t["threshold"], t["recall_target"])
            tuning.append(table.assign(fold=fold))
            selected[fold][NAMES[name]] = dict(params=params, threshold=thr, cv_pr_auc=table["cv_pr_auc"].max())
            print(f"  {NAMES[name]}: {params}  CV PR-AUC {table['cv_pr_auc'].max():.3f}  threshold {thr:.3g}")
            for seed in seeds:
                model = make_model(name, params, cfg, seed).fit(train[key], y, groups)
                path = out(cfg, "models", fold, f"{NAMES[name]}_seed{seed}.joblib")
                model.save(path)
                s = model.score(test[key])
                preds.append(test["meta"].assign(fold=fold, model=NAMES[name], seed=seed, score=s,
                                                 threshold=thr, pred=(s >= thr).astype(int)))
                log.append(dict(fold=fold, model=NAMES[name], seed=seed, path=str(path),
                                train_seconds=model.train_seconds, **model.describe()))
            if name == "rf":
                rf = model.pipe[-1]
                pd.Series(rf.feature_importances_, train["sum_names"]).sort_values(ascending=False) \
                    .to_csv(out(cfg, "folds", f"{fold}_rf_importance.csv"), header=["impurity_importance"])
    pd.concat(preds).to_csv(out(cfg, "predictions.csv"), index=False)
    pd.concat(tuning).to_csv(out(cfg, "tuning.csv"), index=False)
    pd.DataFrame(log).to_csv(out(cfg, "training_log.csv"), index=False)
    save_json(selected, out(cfg, "selected.json"))
    print(f"\npredictions -> {out(cfg, 'predictions.csv')}")


def ablations(cfg: dict) -> pd.DataFrame:
    """Each ablation is a full LOSO run with a reduced search; `reference`
    uses the same budget with default settings, so comparisons are like-for-like."""
    from .config import load_config
    from . import evaluate
    ab = cfg["ablations"]
    rows = []
    for name, override in ab["runs"].items():
        sub = load_config(cfg["config_path"], *cfg["overrides"], override, {"evaluation": {"bootstrap": 300}},
                          output_dir=str(out(cfg, "ablations", name, "x").parent))
        if not out(sub, "results", "summary.csv").exists():
            print(f"\n##### ablation {name}: {override or 'default settings'}")
            run(sub, scale=ab["search_scale"], seeds=ab["seeds"])
            evaluate.run(sub, robustness=False)
        rows.append(pd.read_csv(out(sub, "results", "summary.csv")).assign(ablation=name, setting=json.dumps(override)))
    res = pd.concat(rows, ignore_index=True)
    res.to_csv(out(cfg, "results", "ablations.csv"), index=False)
    return res
