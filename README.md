# UB-SuitFall

**A two-view CCTV fall detection dataset recorded in an age-simulation suit**

UB-SuitFall contains simulated falls and activities of daily living (ADLs) performed by two young adults wearing an age-simulation suit, recorded simultaneously by a frontal and a lateral CCTV camera. Every fall is annotated separately for each camera with its start, impact and end frame. The repository also contains the code used to prepare the data and a subject-independent baseline benchmark (Random Forest, SVM and LSTM).

> **Paper:** *UB-SuitFall: A two-view CCTV fall detection dataset recorded in an age-simulation suit.* Jonathan Arya Priguna, Rakha Apta Pradhana D R, Zahra Auliaul Rohman, Muhammad Ali Fauzi. Faculty of Computer Science, Universitas Brawijaya. *(under review)*

| | |
|---|---|
| Participants | 2 (S01, S02), wearing the Sakamoto Aged Simulation Suit II |
| Activities | 5 fall types and 5 ADLs, 3 trials each |
| Cameras | 2 fixed EyeSec CCTV cameras (frontal CCTV-A, lateral CCTV-B), 1920×1080, 30 fps |
| Recordings | 108 camera-specific recordings from 54 trials (30 falls, 24 ADLs) |
| Frames | 31,239 frames at 10 fps (52.1 min) |
| Annotations | Fall start, impact and end per fall and camera; per-frame labels for every recording |
| Recorded | 15 July 2026, Faculty of Health Sciences (FIKES), Universitas Brawijaya, Malang, Indonesia |
| Licence (data) | [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/) |

---

## Contents

- [Getting the data](#getting-the-data)
- [Recording setup](#recording-setup)
- [Activities](#activities)
- [Folder structure](#folder-structure)
- [Annotations](#annotations)
- [Missing recordings](#missing-recordings)
- [Code](#code)
- [Baseline results](#baseline-results)
- [Citation](#citation)
- [Licence](#licence)
- [Contact](#contact)

---

## Getting the data

This GitHub repository contains the **labels, the manifest and the code**. The **videos and frames** (about 6.8 GB) are too large for Git and are distributed separately:

**Download:** *[link to be added]*

After downloading, place the `CCTV-A/`, `CCTV-B/` and `Frames/` folders of each participant inside `S01/` and `S02/`, so that the layout matches [Folder structure](#folder-structure).

## Recording setup

| | CCTV-A | CCTV-B |
|---|---|---|
| View | Frontal (along the direction of the fall) | Lateral (fall seen side-on) |
| Horizontal distance to mat | ≈ 1.90 m | ≈ 2.10 m |
| Mounting height | ≈ 1.65 m | ≈ 1.65 m |
| Original resolution / frame rate | 1920×1080, 30 fps | 1920×1080, 30 fps |
| Released frame rate | 10 fps | 10 fps |

- Each activity started at a fixed start position at the far end of a mat, and falls were performed onto the mat.
- The cameras are not frame-synchronised, so each camera was annotated separately.
- The room was in normal use, and other people (not wearing the suit) are sometimes visible in the background.

**Participants and suit.**

| Participant | Age (years) | Height (m) | Body mass (kg) | Sex |
|---|---|---|---|---|
| S01 | 20 | 1.73 | 77 | Male |
| S02 | 23 | 1.70 | 62 | Male |

- Both participants wore the Sakamoto Aged Simulation Suit II (model M176, size L).
- A 0.5 kg weight was placed in every weight pocket of the suit.
- The goggles, ear plugs, gloves and belt covers were not worn.
- The suit is green with red straps; the baseline uses this colour to extract the participant's silhouette.

## Activities

The three trials of each activity were recorded in succession, in the order below. The activity set and durations follow UP-Fall, without its jumping activity.

| ID | Class | Activity | Duration |
|---|---|---|---|
| 1 | FALL | Forward fall using hands | 10 s |
| 2 | FALL | Forward fall using knees | 10 s |
| 3 | FALL | Falling backwards | 10 s |
| 4 | FALL | Falling sideward | 10 s |
| 5 | FALL | Falling while sitting in an empty chair | 10 s |
| 6 | ADL | Walking | 60 s |
| 7 | ADL | Standing | 60 s |
| 8 | ADL | Sitting in a chair | 60 s |
| 9 | ADL | Picking up an object from the floor | 10 s |
| 10 | ADL | Lying down | 60 s |

## Folder structure

```
UB-SuitFall/
├── S01/
│   ├── CCTV-A/                    videos, ActivityXTrialY.mp4          (download)
│   ├── CCTV-B/                                                          (download)
│   ├── Frames/
│   │   ├── CCTV-A/ActivityXTrialY/00001.jpg, 00002.jpg, ...  10 fps   (download)
│   │   └── CCTV-B/ActivityXTrialY/...                                   (download)
│   ├── frame_labels/              S01_A01_T1_CCTV-A.csv, ...  per-frame labels
│   └── fall_labels.csv            one row per recording
├── S02/                           same structure
├── MANIFEST.csv                   one row per released recording
└── code/
    ├── preprocessing/             frame extraction, renaming, annotation, release builder
    └── benchmark/                 RF / SVM / LSTM benchmark
```

## Annotations

**Fall annotation.** For every fall recording, three frames were annotated per camera, at 10 fps (0.1 s resolution):

| Event | Definition |
|---|---|
| Fall start | First frame of the loss of balance |
| Impact | First frame at which the pelvis or trunk is stopped by the ground (earlier contact of hands or knees belongs to the fall) |
| Fall end | First frame at which the body comes to rest, whatever the final posture |

- Candidate frames were proposed automatically from the vertical trajectory of the hip, estimated with an off-the-shelf pose model.
- One annotator then checked and corrected every candidate.
- The pose model was used only for these proposals and is not part of the benchmark.

**`fall_labels.csv`** (one row per recording):

| Column | Content |
|---|---|
| `subject_id` | Participant (S01, S02) |
| `trial_uid` | Trial identifier, e.g. `S02_A03_T1` (shared by both cameras) |
| `camera` | CCTV-A or CCTV-B |
| `activity_id` | Activity number (1–10) |
| `activity_name` | Activity name |
| `trial_id` | Trial number (1–3) |
| `category` | FALL or NON_FALL |
| `original_filename` | Name of the original video file |
| `n_frames` | Number of frames (0 if unavailable) |
| `fps` | Frame rate of the released frames (10) |
| `frame_numbering` | Frame numbering convention (1-based) |
| `fall_start_frame` | First frame of the fall (this camera; empty for ADLs) |
| `impact_frame` | Impact frame (this camera; empty for ADLs) |
| `fall_end_frame` | Frame at which the body comes to rest (this camera; empty for ADLs) |

**`frame_labels/<trial_uid>_<camera>.csv`** (one row per frame):

| Column | Content |
|---|---|
| `frame` | Frame number (1-based, matches the frame file name) |
| `filename` | Frame file name |
| `state` | PRE_FALL, FALL, POST_FALL or NON_FALL |
| `label` | 1 for FALL (fall start to fall end), 0 otherwise |
| `label_3class` | 0 normal, 1 falling, 2 after the fall |

All frames of ADL recordings are NON_FALL.

**`MANIFEST.csv`** lists every released recording with its paths: `subject_id`, `trial_uid`, `camera`, `activity_id`, `trial_id`, `category`, `video`, `frames_dir`, `n_frames` and `frame_labels`.

## Missing recordings

Of the 120 planned recordings (2 participants × 10 activities × 3 trials × 2 cameras), 108 are available. All 12 unavailable recordings are from S01, from six trials whose videos were missing or corrupted, and each affects both cameras:

| Activity | Trials | Recordings |
|---|---|---|
| 7 Standing | 2 | 2 |
| 9 Picking up an object | 2, 3 | 4 |
| 10 Lying down | 1, 2, 3 | 6 |

As a result, S01 has no lying-down recordings and only one picking-up trial. Their rows are kept in `fall_labels.csv` with `n_frames = 0`.

## Code

All scripts are run with Python 3.10 or newer. Install the dependencies with:

```bash
pip install -r code/benchmark/requirements.txt
```

Install the PyTorch build that matches your CUDA version if you want GPU timings. Frame extraction also needs `ffmpeg`.

### Preprocessing (`code/preprocessing/`)

These scripts produced the released data from the raw recordings. You do not need them to use the dataset; they document how it was made.

| Script | Purpose |
|---|---|
| `rename_trials.py` | Renames videos named `<activity>_<trial>.mp4` to `ActivityXTrialY.mp4` |
| `frame_extraction.py` | Extracts frames at 10 fps with ffmpeg into `Frames/<camera>/ActivityXTrialY/` |
| `fall_labels.py` | Proposes candidate fall frames, supports their review, and exports `fall_labels.csv` and `frame_labels/` |
| `build_release.py` | Assembles the released folder structure and `MANIFEST.csv` from the working folders |

```bash
python code/preprocessing/frame_extraction.py --root /path/to/S01 --skip Activity7Trial2.mp4
python code/preprocessing/frame_extraction.py --root /path/to/S02
```

Each script describes its usage at the top of the file.

### Benchmark (`code/benchmark/`)

How the baseline works:

1. **Silhouette.** It extracts the participant's silhouette from the colour of the suit. Static green objects in the room are removed with a background mask.
2. **Windows.** It describes the silhouette's shape and motion in 2 s windows (stride 0.5 s).
3. **Classifiers.** It compares RF and SVM on window summaries with an LSTM on the ordered per-frame signals.
4. **Evaluation.** Evaluation is leave-one-subject-out. The hyperparameters, decision thresholds, imputation, standardisation and background masks are all estimated from the training participant only.

`config_foldmask.yaml` reproduces the paper. Run from `code/benchmark/` after the frames are in place:

```bash
cd code/benchmark
python run.py extract    --config config_foldmask.yaml   # silhouettes and per-frame features
python run.py inspect    --config config_foldmask.yaml   # dataset check: 108 recordings, 5,890 windows
python run.py experiment --config config_foldmask.yaml   # tuning, training, predictions (longest step)
python run.py evaluate   --config config_foldmask.yaml   # metrics, confidence intervals, paired tests
python run.py benchmark  --config config_foldmask.yaml   # latency, throughput, memory
python run.py report     --config config_foldmask.yaml   # tables and figures
python run.py ablations  --config config_foldmask.yaml   # optional
```

- **Outputs** are written to `code/benchmark/runs/`.
- **`config.yaml`** holds every shared setting. `config_foldmask.yaml` inherits it and sets the training-participant-only background masks.
- **Documentation:** `code/benchmark/README.md` describes the modules, and `code/benchmark/METHODS.md` describes the design decisions.

## Baseline results

Leave-one-subject-out test results (mean ± SD over five seeds of the mean over both folds). FA/h is false alarms per hour of ADL recording.

| Metric | RF | SVM | LSTM |
|---|---|---|---|
| F1 (FALL) | **0.785 ± 0.005** | 0.623 | 0.756 ± 0.045 |
| PR-AUC | **0.935 ± 0.001** | 0.846 | 0.851 ± 0.032 |
| Balanced accuracy | **0.917 ± 0.002** | 0.840 | 0.883 ± 0.020 |
| Recall | **0.850 ± 0.004** | 0.703 | 0.780 ± 0.039 |
| Fall instances detected | **60/60** | 59/60 | **60/60** |
| Median detection delay (s) | **1.00 ± 0.06** | 1.20 | 1.19 ± 0.05 |
| FA/h | **14.3 ± 0.8** | 23.8 | 20.9 ± 3.1 |

With only two participants, each fold measures generalisation to one unseen person, so these results are an initial reference rather than a definitive comparison. See the paper for confidence intervals, paired tests, per-fold results and ablations.

## Citation

If you use UB-SuitFall, please cite:

```bibtex
@article{priguna_ubsuitfall,
  title   = {{UB-SuitFall}: A two-view {CCTV} fall detection dataset recorded in an age-simulation suit},
  author  = {Priguna, Jonathan Arya and Pradhana D R, Rakha Apta and Rohman, Zahra Auliaul and Fauzi, Muhammad Ali},
  note    = {Under review},
  year    = {2026}
}
```

## Licence

- **Data** (videos, frames, labels and manifest): [Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International (CC BY-NC-SA 4.0)](https://creativecommons.org/licenses/by-nc-sa/4.0/).
- **Code:** *[licence to be added]*

Both participants consented to the publication of their recordings, including identifiable images of their faces. Please respect their privacy and do not attempt to identify them or to link the recordings to other information about them.

## Contact

Corresponding author: **Muhammad Ali Fauzi** ([moch.ali.fauzi@ub.ac.id](mailto:moch.ali.fauzi@ub.ac.id)), Faculty of Computer Science (FILKOM), Universitas Brawijaya.

## Acknowledgements

We thank the participants and the Faculty of Computer Science (FILKOM) and the Faculty of Health Sciences (FIKES), Universitas Brawijaya, for their contribution to this dataset.
