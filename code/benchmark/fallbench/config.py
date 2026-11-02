"""Configuration and small shared helpers."""
from __future__ import annotations

import copy
import json
import random
from pathlib import Path

import numpy as np
import yaml


def merge(base: dict, override: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in (override or {}).items():
        out[k] = merge(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else copy.deepcopy(v)
    return out


def load_config(path, *overrides: dict, output_dir: str | None = None) -> dict:
    path = Path(path).resolve()
    cfg = yaml.safe_load(path.read_text())
    if "inherit" in cfg:   # e.g. config_foldmask.yaml: every setting of config.yaml, plus a few changes
        base = yaml.safe_load((path.parent / cfg.pop("inherit")).read_text())
        cfg = merge(base, cfg)
    for o in overrides:
        cfg = merge(cfg, o)
    if output_dir:
        cfg["output_dir"] = output_dir
    # relative paths are relative to the config file's folder
    for key in ("output_dir", "features_dir"):
        p = Path(cfg[key]).expanduser()
        cfg[key] = str(p if p.is_absolute() else (path.parent / p).resolve())
    cfg["config_path"] = str(path)
    cfg["overrides"] = list(overrides)
    return cfg


def out(cfg: dict, *parts: str) -> Path:
    """Path inside the run's output folder (parent folders are created)."""
    p = Path(cfg["output_dir"]).joinpath(*parts)
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except ImportError:
        pass


def save_json(obj, path: Path) -> None:
    def default(o):
        if isinstance(o, np.generic):
            return o.item()
        if isinstance(o, np.ndarray):
            return o.tolist()
        return str(o)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, default=default))


def md_table(df, fmt: str = "{:.3f}", index: bool = True) -> str:
    d = df.reset_index() if index else df
    cell = lambda v: ("" if v != v else fmt.format(v)) if isinstance(v, float) else str(v)  # noqa: E731
    rows = ["| " + " | ".join(map(str, d.columns)) + " |", "|" + "---|" * len(d.columns)]
    rows += ["| " + " | ".join(cell(v) for v in r) + " |" for r in d.itertuples(index=False)]
    return "\n".join(rows)
