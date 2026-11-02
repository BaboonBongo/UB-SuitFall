# Methods and design decisions

This document explains what the pipeline does and why, in the order of the
experimental workflow. It is written so the relevant parts can be adapted
into the Methods section of the paper. Names in brackets refer to
`config.yaml` keys.

## 1. Data and labels

* Two participants (S01, S02) performed 10 activities × 3 trials while
  wearing an ageing-simulation suit, recorded by two fixed CCTV cameras
  (CCTV-A, CCTV-B; not frame-synchronised). Activities 1-5 are simulated
  falls, 6-10 are activities of daily living (walking, standing, sitting,
  picking up an object, lying down).
* Frames were extracted at 10 fps. For every fall recording, the fall start,
  impact and fall end frames were set manually per camera; frames from fall
  start to fall end are FALL (1), all others NON_FALL (0). Non-fall
  recordings are NON_FALL throughout.
* Unusable recordings (missing or corrupted video) are listed by
  `run.py inspect` and excluded everywhere. For S01, picking-up trials 2-3
  and all lying-down trials are missing; this matters for cross-subject
  testing (a model trained on S01 has never seen intentional lying down).

## 2. Input representation: suit silhouette (no pose)

The skeleton annotations are not used. Instead, each frame is reduced to
the silhouette of the suit, which is green with red patches:

1. Frame resized to `silhouette.width` pixels wide, converted to HSV.
2. Suit pixels = HSV range of the suit green (the same range used to
   identify the subject in the earlier pose pipeline).
3. Static green scenery (e.g. green crates) is removed with a static mask:
   pixels that are suit-green in ≥ 50% of frames sampled evenly (4 per video)
   from the camera's videos, dilated 11x11. The subject moves between trials,
   scenery does not. Which videos are sampled is set by `mask_source`:
   `participant` (original run) = all videos of the same participant and
   camera, including the held-out participant's; `training` (corrected run) =
   per fold, only the training participant's videos of that camera, applied
   to both participants.
4. Suit patches separated by the red parts are merged by dilation; the
   merged region with the most suit pixels is the subject, preferring the
   region nearest the previous frame's position (simple tracking). Other
   people (no suit) are ignored by construction.
5. Shape features of the subject's suit pixels: centroid (cx, cy), top and
   bottom, bounding-box width/height, aspect ratio, area, extent (fill
   ratio), orientation of the major axis from vertical (image moments),
   elongation, major/minor axis length; all normalised by frame size.
   A `present` flag marks frames without a silhouette.

Visual QA sheets (`runs/features/qa/`) show the extracted
silhouette on frames around each fall and across each non-fall recording;
they must be checked before the experiment and a few can go into the paper.

**Mapping to the feature types in the plan.** Normalised coordinates →
centroid / top / bottom; relative positions → box width/height, aspect;
body inclination → orientation angle, elongation; body-centre displacement
→ centroid change; hip/shoulder drop → top and centroid drop relative to
body height; velocities and accelerations → per-frame differences; posture
change over time → window SD and first-to-last change; missing indicators →
`present`, `missing_fraction`. Joint-level features cannot be computed
without keypoints and are therefore not used.

## 3. Windows

* 20-frame windows (2.0 s) with a 5-frame stride (0.5 s) [windows].
  Rationale: at 10 fps the manually labelled falls last ~1-4 s from start
  to end, with start→impact typically ~1 s (see the fall-timing table in
  the inspection report), so a 2 s window contains the descent and
  impact. A 0.5 s stride gives an alarm decision every 0.5 s.
* Window label = FALL if ≥ 50% of its frames are FALL.
* Overlapping windows are strongly correlated. This is handled by (i)
  keeping every window of a trial on the same side of every split and (ii)
  computing uncertainty by resampling whole recordings. Non-overlapping
  windows are run as an ablation.
* All features are causal: a window uses only frames up to its last frame;
  missing silhouettes are forward-filled from earlier frames only.

## 4. Same information for all three models

* **RF and SVM:** one summary vector per window: mean/min/max of each shape
  signal, missing fraction, and (motion set) SD and first-to-last change of
  each signal, max/min/mean-absolute velocity, max absolute acceleration
  and body-height-relative drops (94 features).
* **LSTM:** the ordered per-frame signals of the same window (20 × 24:
  shape signals, velocities, accelerations, present flag).

Both representations come from the same per-frame signals over the same
window, so the temporal context is identical. The difference is only
whether the temporal summarisation is engineered (RF/SVM) or learned (LSTM).

**Learned preprocessing is fitted on training data only:** median imputation
(RF, SVM) and standardisation (SVM, LSTM) are part of the model object and
are fitted inside every training fold. No automatic feature selection is
used: with ~100 features and thousands of windows RF and regularised SVM do
not need it, and an extra selection step would add tuning variance. RF
impurity importances are saved for interpretation only.

## 5. Splits (leakage-resistant)

* **Outer: leave-one-subject-out.** Train on S01, test on S02, and vice
  versa. The test subject is never used for tuning, threshold selection or
  early stopping.
* **Inner (inside the training subject):** 3-fold StratifiedGroupKFold with
  groups = trial (`trial_uid`). Both camera views of one fall are therefore
  always in the same fold, and no fall event contributes windows to both
  sides of a split.
* Hyperparameters are chosen by mean inner-CV PR-AUC. The decision threshold
  is chosen on the out-of-fold scores of the chosen configuration (max F1;
  `recall_at` can enforce a minimum recall instead). The LSTM's early
  stopping uses a further 20% of training trials.
* After selection the model is refitted on the whole training subject
  (LSTM: 80% with early stopping on the remaining 20% of trials).

## 6. Models

* **Class imbalance:** all three models use the same setting
  [class_weighting]: inverse-frequency class weights for RF and SVM and the
  equivalent positive-class weight in the LSTM loss. Turning it off is an
  ablation.
* **RF** (scikit-learn): randomised search over trees, depth and min samples
  split/leaf [models.rf]. Recorded: trees, nodes, depth, size, training time,
  latency.
* **SVM** (scikit-learn SVC) on standardised features: the kernel is part
  of the search, so linear (C grid) and RBF (C × gamma grid) are both
  evaluated in inner cross-validation (every configuration's score is in
  `tuning.csv`) and the better one is used [models.svm]. Ranking metrics use the decision value; no probability
  calibration is needed because the threshold is chosen on training-subject
  scores. If the training set were too large for kernel SVM, the run stops
  and says so rather than silently subsampling.
* **LSTM** (PyTorch): LSTM over the 20-frame sequence, last hidden state →
  linear → logit; BCE-with-logits loss with positive-class weight
  (negatives/positives); Adam; gradient clipping; early stopping on
  validation PR-AUC; best checkpoint saved. Modest search over hidden size
  (16-64), layers (1-2), dropout, learning rate and batch size [models.lstm].
* **Repeated runs:** every model is retrained with 5 seeds (SVM is
  deterministic, so its SD is 0). Tables report mean ± SD over seeds.

## 7. Metrics

Window level: accuracy, precision, recall (sensitivity), specificity, F1
(FALL), macro-F1, balanced accuracy, ROC-AUC, PR-AUC, FPR, FNR, confusion
matrix. Event level: event recall (a fall is detected if an alarm episode
overlaps it), detection delay, false alarms per hour of non-fall recording.

**Why not accuracy alone:** FALL windows are a small minority (see the
inspection report). A model that never predicts FALL would still reach high
accuracy. Recall (missed falls) and false alarms are the operational costs,
so F1, balanced accuracy, PR-AUC and the event-level measures are the
primary metrics.

Each LOSO fold has its own model and threshold, so metrics are computed
per fold and averaged over folds; scores from different folds are never
pooled into one ROC curve.

## 8. Uncertainty and statistical comparison

* 95% confidence intervals: recording-level bootstrap (2000 resamples;
  recordings resampled within each test subject and category). Windows are
  never resampled individually because they are correlated.
* Model comparison: the three primary models are scored on the same
  resampled recordings in each bootstrap iteration (paired). Two-sided
  bootstrap p-values for each pair, Holm-corrected over the three pairs,
  repeated for every seed. A difference is described as significant only
  when it holds in every seed. Effect sizes (differences with CIs) are
  always reported.
* With only two subjects, between-subject variance cannot be estimated.
  These tests describe the uncertainty from the recordings of the held-out
  participants, not generalisation to the population, and must be worded
  that way.

## 9. Computational cost

Measured on the lab PC with one protocol for all models (see
`results/benchmark_info.json`): each model in a fresh process, 1 CPU thread (LSTM
additionally on GPU), 20 warm-up calls, 300 single-window calls (median and
p95 latency), batch-256 throughput, resident-memory increase during
load + inference, file size. Preprocessing (decode, silhouette, window
features) is timed separately; end-to-end per-window cost = 5 new frames ×
per-frame cost + window features + model. Operation counts are labelled:
RF node visits are measured; SVM kernel operations and LSTM
multiply-accumulates are theoretical. Theoretical complexity, measured
runtime, memory and deployment cost are reported separately; no claim about
complexity is made from the model family alone.

## 10. Ablations (`run.py ablations`)

Window length 1 s and 3 s; non-overlapping windows; shape-only vs
shape + motion features; no class weighting; robustness to 10-30% of test
frames losing their silhouette (no retraining); within-subject inner-CV vs
cross-subject test PR-AUC (generalisation gap). Design ablations use a
reduced search and are compared with a `reference` run with the same
budget.

## 11. Limitations to state in the paper

Two young participants in an ageing-simulation suit; simulated falls onto
a mat in one room; two cameras; one annotator for fall boundaries (0.1 s
resolution); silhouettes depend on the suit colour and do not transfer to
everyday clothing without a person detector or segmenter; missing S01
activities; LOSO with two folds.

## 12. A defensible contribution

Comparing three classic algorithms is not novel by itself. Defensible
contributions, depending on what the results show:

1. **The dataset:** a two-view CCTV fall dataset in an ageing-simulation
   suit with manually checked fall start / impact / end per camera and
   activities of daily living that resemble falls (sitting, lying down,
   picking up).
2. **The evaluation protocol:** subject-independent, trial-grouped,
   event-level evaluation with recording-level uncertainty. Many fall
   papers report window-level accuracy on random splits. Showing how much
   the conclusions change under this protocol (generalisation gap table) is
   itself a finding.
3. **Cost-aware comparison:** measured latency, memory and end-to-end cost
   including preprocessing. If preprocessing dominates, the choice of
   classifier matters less for deployment than usually implied.
4. **Failure analysis:** which activities and which camera view cause false
   alarms (e.g. lying down vs post-fall; top-down vs side view).
