"""Computational cost under one protocol for every model.

Same machine, same held-out windows (first fold, first seed), float32 input.
Each model is timed in a fresh process limited to `benchmark.threads` CPU
threads (the LSTM also on GPU if available): warm-up calls are discarded,
then single-window latency (median, p95) and batch throughput are measured
with time.perf_counter. Memory = increase of resident memory, sampled every
~1 ms, while loading the model and running inference.

Preprocessing is timed separately, per component:
  - frame read: reading a stored JPEG frame from disk, decoding it and
    resizing it (NOT decoding a live CCTV video stream);
  - silhouette extraction with the fold's static mask;
  - window features.
Static-mask generation is done offline before inference and is NOT included.
The ESTIMATED end-to-end time per window is the sum of separately measured
medians: stride x (frame read + silhouette) + window features + model
latency. It is not a directly measured end-to-end execution time.
Operation counts are labelled measured or theoretical.
"""
from __future__ import annotations

import multiprocessing as mp
import os
import platform
import sys
import threading
import time
from pathlib import Path

import numpy as np
import pandas as pd

from . import masking
from .config import out, save_json


def _rss_mb() -> float:
    return int(Path("/proc/self/statm").read_text().split()[1]) * os.sysconf("SC_PAGE_SIZE") / 2 ** 20


def _measure(job: dict, queue) -> None:
    from threadpoolctl import threadpool_limits
    import torch
    from .models import Model
    threadpool_limits(job["threads"])
    torch.set_num_threads(job["threads"])
    X = np.load(job["data"])[job["input"]]
    base = peak = _rss_mb()
    stop = threading.Event()

    def sample():
        nonlocal peak
        while not stop.is_set():
            peak = max(peak, _rss_mb())
            time.sleep(0.001)
    threading.Thread(target=sample, daemon=True).start()
    model = Model.load(job["path"])
    if hasattr(model, "pipe") and hasattr(model.pipe[-1], "n_jobs"):
        model.pipe[-1].n_jobs = job["threads"]
    if job["device"] == "cuda":
        model.to("cuda")
        torch.cuda.reset_peak_memory_stats()

    def call(x):
        r = model.score(x)
        if job["device"] == "cuda":
            torch.cuda.synchronize()
        return r
    for i in range(job["warmup"]):
        call(X[i % len(X):][:1])
    lat = []
    for i in range(job["repeats"]):
        x = X[i % len(X):][:1]
        t0 = time.perf_counter()
        call(x)
        lat.append(1000 * (time.perf_counter() - t0))
    batch = np.resize(X, (job["batch"],) + X.shape[1:])
    tput = []
    for _ in range(10):
        t0 = time.perf_counter()
        call(batch)
        tput.append(time.perf_counter() - t0)
    stop.set()
    ops, unit = model.ops(X[:2000])
    queue.put(dict(latency_ms=np.median(lat), latency_p95_ms=np.percentile(lat, 95),
                   throughput_per_s=job["batch"] / np.median(tput), memory_increase_mb=max(0.0, peak - base),
                   gpu_peak_mb=torch.cuda.max_memory_allocated() / 2 ** 20 if job["device"] == "cuda" else None,
                   model_file_mb=os.path.getsize(job["path"]) / 2 ** 20, ops_per_window=ops, ops_unit=unit,
                   **model.describe()))


def _isolated(job: dict) -> dict:
    ctx = mp.get_context("spawn")
    q = ctx.Queue()
    p = ctx.Process(target=_measure, args=(job, q))
    p.start()
    r = q.get()
    p.join()
    return r


def preprocessing_cost(cfg: dict, index: pd.DataFrame, subject: str) -> dict:
    import cv2
    from .data import list_frames, load_recording
    from .features import frame_signals, summarise
    from .silhouette import Tracker, read
    cv2.setNumThreads(1)
    s, n = cfg["silhouette"], cfg["benchmark"]["sample_frames"]
    recs = index[index.usable & (index.subject_id == subject)].sample(frac=1, random_state=cfg["seed"])
    decode, silhouette = [], []
    for rec in recs.itertuples():
        static = cv2.imread(str(masking.mask_png(cfg, rec.subject_id, rec.camera)), 0)
        track = Tracker(static, s)
        for f in list_frames(rec.frames_dir)[:30]:
            t0 = time.perf_counter()
            img = read(f, s["width"])
            t1 = time.perf_counter()
            track(img)
            decode.append(t1 - t0)
            silhouette.append(time.perf_counter() - t1)
        if len(decode) >= n:
            break
    df = load_recording(cfg, recs.iloc[0]).iloc[:cfg["windows"]["length"]]
    feat = []
    for _ in range(200):
        t0 = time.perf_counter()
        sig = frame_signals(df, cfg["fps"])
        summarise(sig.to_numpy(np.float32)[None], list(sig.columns), cfg["windows"]["features"])
        feat.append(time.perf_counter() - t0)
    ms = lambda v: 1000 * float(np.median(v))  # noqa: E731
    per_window = cfg["windows"]["stride"] * (ms(decode) + ms(silhouette)) + ms(feat)
    return dict(frames_timed=len(decode), frame_read_ms_per_frame=ms(decode), silhouette_ms_per_frame=ms(silhouette),
                window_features_ms=ms(feat), preprocessing_ms_per_window=per_window,
                notes=dict(frame_read="stored JPEG frame: disk read + decode + resize to silhouette.width; "
                                      "not live video-stream decoding",
                           static_mask="generated offline before inference; excluded from all timings",
                           mask_used=masking.mode(cfg) + (f" (from {cfg['mask_from']})" if "mask_from" in cfg else ""),
                           end_to_end="estimated: stride x (frame read + silhouette) + window features + "
                                      "model latency, each a separately measured median"))


def hardware() -> dict:
    import cv2
    import sklearn
    import torch
    cpu = next((l.split(":", 1)[1].strip() for l in Path("/proc/cpuinfo").read_text().splitlines()
                if l.startswith("model name")), platform.processor())
    return dict(platform=platform.platform(), cpu=cpu, logical_cpus=os.cpu_count(),
                ram_gb=round(os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / 2 ** 30, 1),
                gpu=torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
                cuda=torch.version.cuda, python=sys.version.split()[0], numpy=np.__version__,
                pandas=pd.__version__, scikit_learn=sklearn.__version__, torch=torch.__version__,
                opencv=cv2.__version__)


def run(cfg: dict) -> None:
    import torch
    from .data import build_index
    from .features import build_windows
    b = cfg["benchmark"]
    log = pd.read_csv(out(cfg, "training_log.csv"))
    fold, seed = sorted(log.fold.unique())[0], log.seed.min()
    subject = fold.removeprefix("test_")
    fcfg = masking.fold_cfg(cfg, subject)
    index = build_index(cfg)
    test = build_windows(fcfg, [subject], index)
    data = out(cfg, "benchmark_windows.npz")
    np.savez(data, sum=test["X_sum"], seq=test["X_seq"].astype(np.float32))
    rows = []
    for r in log[(log.fold == fold) & (log.seed == seed)].itertuples():
        devices = ["cpu", "cuda"] if r.model == "LSTM" and torch.cuda.is_available() else ["cpu"]
        for dev in devices:
            print(f"  timing {r.model} ({dev}) ...")
            job = dict(path=r.path, data=str(data), input="seq" if r.model == "LSTM" else "sum", device=dev,
                       threads=b["threads"], warmup=b["warmup"], repeats=b["repeats"], batch=b["batch"])
            rows.append(dict(model=r.model, device=dev, train_seconds=r.train_seconds, **_isolated(job)))
    pre = preprocessing_cost(fcfg, index, subject)
    bench = pd.DataFrame(rows).assign(preprocessing_ms=pre["preprocessing_ms_per_window"])
    bench["estimated_end_to_end_ms"] = bench["latency_ms"] + bench["preprocessing_ms"]
    bench.to_csv(out(cfg, "results", "benchmark.csv"), index=False)
    save_json(dict(hardware=hardware(), fold=fold, seed=seed, windows=len(test["meta"]), settings=b,
                   preprocessing=pre), out(cfg, "results", "benchmark_info.json"))
    print(f"benchmark -> {out(cfg, 'results', 'benchmark.csv')}")
