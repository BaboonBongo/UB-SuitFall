# Fall detection benchmark: RF vs SVM vs LSTM

Binary fall detection (FALL vs NON_FALL) on the two-camera suit dataset,
from suit-silhouette features (no pose model, no skeleton annotations).
All settings are in `config.yaml`; all outputs go to `runs/`. The dataset
folders are only read.

```
config.yaml            every setting (paths, windows, search spaces, ablations)
run.py                 command line: one command per stage
fallbench/
  data.py              dataset index: recordings, labels, frame folders
  silhouette.py        suit silhouette + shape features per frame
  features.py          windows, window labels, RF/SVM summaries, LSTM sequences
  models.py            RF, SVM, LSTM behind one interface (fit / score / save)
  metrics.py           window + event metrics, bootstrap, paired tests
  experiment.py        leave-one-subject-out tuning, training, predictions, ablations
  evaluate.py          results tables, CIs, paired tests, robustness
  benchmark.py         latency, throughput, memory, size, operation counts
  report.py            dataset report, figures, paper tables, reports
METHODS.md             design decisions, written for the paper's Methods section
```

## Static-mask source (original vs corrected)

`config.yaml` reproduces the original run: one static green-background mask per
participant and camera, estimated from all of that participant's recordings,
including the held-out participant's (transductive). `config_foldmask.yaml`
is the corrected run: per leave-one-subject-out fold, one mask per camera
estimated from the training participant only and applied to both participants
(`fallbench/masking.py`). It inherits every other setting and writes to
`runs/foldmask/`. Commands: `RUN_CORRECTED.md`. Tools: `tools/mask_diagnostic.py`
(original masks, camera check, silhouettes with each mask) and
`tools/compare_runs.py` (validation and original-vs-corrected tables).

## Before running

* Frames in `<subject root>/Frames/<camera>/ActivityXTrialY/00001.jpg`
* The cleaned `fall_labels.csv` in each subject root
* Per-frame labels from `python fall_labels.py export <subject root>`
* Check the two paths under `subjects:` in `config.yaml`
* `pip install -r requirements.txt` in your venv (keep your CUDA build of torch)

## Run (from this folder)

```bash
python run.py all --quick     # test the setup on a tiny search (runs/quick, not for the paper)
python run.py extract         # silhouettes for every recording
python run.py inspect         # runs/main/inspection.md
#   check a few images in runs/features/qa/: the magenta region must be the suited subject
python run.py experiment      # longest step; over VNC use: nohup python run.py experiment > exp.log 2>&1 &
python run.py evaluate
python run.py benchmark       # keep the PC idle while timing
python run.py report
python run.py ablations       # optional, after the main run
```

## Outputs (`runs/main/`)

| | |
|---|---|
| `inspection.md` | dataset structure, label balance, fall timing, silhouette quality, issues |
| `folds/` | split per fold (subjects, inner-CV trial assignment), RF feature importance |
| `tuning.csv`, `selected.json` | every configuration tried; chosen hyperparameters and thresholds |
| `models/`, `training_log.csv` | saved models (preprocessing included), training time, structure |
| `predictions.csv` | every test window: recording, frames, true label, score, threshold, prediction |
| `results/` | `summary.csv`, `per_fold.csv`, `confidence_intervals.csv`, `paired_tests*.csv`, `events.csv`, `per_activity.csv`, `per_camera.csv`, `benchmark.csv`, `missing_silhouette.csv`, `generalisation_gap.csv`, `ablations.csv` |
| `figures/` | confusion matrices, ROC, PR, latency vs F1, missing-silhouette robustness |
| `tables/` | `paper_tables.md`, `performance.csv/.tex`, `computational_cost.csv` |
| `reports/` | `results_summary.md`, `reproducibility.md` |
