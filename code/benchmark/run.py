#!/usr/bin/env python3
"""Run the fall-detection benchmark pipeline."""
import argparse
import time
from pathlib import Path

from fallbench import benchmark, evaluate, experiment, report, silhouette
from fallbench.config import load_config

QUICK = {"tuning": {"seeds": [0]}, "evaluation": {"bootstrap": 50},
         "models": {"rf": {"n_iter": 2}, "svm": {"n_iter": 3}, "lstm": {"n_iter": 1, "max_epochs": 5}},
         "benchmark": {"repeats": 30, "warmup": 3, "sample_frames": 30},
         "ablations": {"seeds": [0], "search_scale": 0.2}}

STAGES = {
    "extract": lambda cfg, a: silhouette.extract(cfg, a.force, a.accept_camera_shift),
    "inspect": lambda cfg, a: report.inspect(cfg),
    "experiment": lambda cfg, a: experiment.run(cfg),
    "evaluate": lambda cfg, a: evaluate.run(cfg),
    "benchmark": lambda cfg, a: benchmark.run(cfg),
    "report": lambda cfg, a: report.run(cfg),
    "ablations": lambda cfg, a: (experiment.ablations(cfg), report.reports(cfg)),
}

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("stage", choices=[*STAGES, "all"])
    ap.add_argument("--config", default=Path(__file__).with_name("config.yaml"))
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--accept-camera-shift", action="store_true")
    a = ap.parse_args()
    cfg = load_config(a.config)
    if a.quick:
        cfg = load_config(a.config, QUICK, output_dir=str(Path(cfg["output_dir"]).parent / "quick"))
    print(f"config: {cfg['config_path']}\nmask source: {cfg.get('mask_source', 'participant')}\n"
          f"features: {cfg['features_dir']}\noutput: {cfg['output_dir']}")
    for stage in [s for s in STAGES if s != "ablations"] if a.stage == "all" else [a.stage]:
        t0 = time.time()
        print(f"\n######## {stage}")
        STAGES[stage](cfg, a)
        print(f"[{stage}] {time.time() - t0:.0f} s")
