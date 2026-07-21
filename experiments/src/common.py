# -*- coding: utf-8 -*-
"""Shared utilities: config, paths, deterministic data loading, small helpers.

Every script imports from here so the evaluation protocol (seed, columns, splits)
is defined exactly once. Run scripts from anywhere; paths resolve to the repo root.
"""
from __future__ import annotations
import os, sys, json, subprocess, random
import numpy as np
import pandas as pd
import yaml

# repo root = two levels up from this file (experiments/src/common.py -> repo root)
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))


def load_config() -> dict:
    with open(os.path.join(ROOT, "experiments", "config.yaml"), "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


CFG = load_config()
SEED = int(CFG["seed"])


def rp(*parts) -> str:
    """Resolve a path relative to the repo root."""
    return os.path.join(ROOT, *parts)


def seed_everything(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def git_hash() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"],
                                       cwd=ROOT, stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        return "nogit"


# ---------------------------------------------------------------- data loading
def load_task(task: str) -> pd.DataFrame:
    """Load a task-specific cut with the agreed prep applied.

    task: 'rating' or 'tags'. Returns a DataFrame with:
      - statement_proc filled (the 2 known NaN cells -> '')
      - a 'tokens' column (statement_proc.split(), for Word2Vec)
      - numeric feature columns coerced to float
      - target column intact ('rating' or 'tags_norm')
    """
    key = "rating_csv" if task == "rating" else "tags_csv"
    df = pd.read_csv(rp(CFG["paths"][key]), low_memory=False)

    proc = CFG["columns"]["text_proc"]
    df[proc] = df[proc].fillna("")
    df["tokens"] = df[proc].map(lambda s: s.split())

    for c in CFG["columns"]["numeric"]:
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0)

    if task == "rating":
        df["rating"] = pd.to_numeric(df["rating"], errors="coerce")
        df = df[df["rating"].notna()].reset_index(drop=True)
    else:
        df["tags_norm"] = df["tags_norm"].fillna("")
        df = df[df["tags_norm"].str.strip() != ""].reset_index(drop=True)
    return df


def parse_tags(series: pd.Series) -> list[list[str]]:
    """tags_norm string -> list of tag lists (split on ',', stripped)."""
    return [[t.strip() for t in s.split(",") if t.strip()] for s in series]


def fmt_mean_std(mean: float, std: float, nd: int = 4) -> str:
    return f"{round(mean, nd)}±{round(std, nd)}"


def dump_json(obj, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False, default=_json_default)


def _json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.ndarray,)):
        return o.tolist()
    return str(o)


if __name__ == "__main__":
    seed_everything()
    for t in ["rating", "tags"]:
        d = load_task(t)
        print(f"{t:7s}: {d.shape[0]} rows, cv_groups={d[CFG['split']['group_col']].nunique()}, "
              f"proc_nan={int(d[CFG['columns']['text_proc']].isna().sum())}")
