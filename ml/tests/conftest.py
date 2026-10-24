"""Shared fixtures: a small SYNTHETIC dataset, trained once per test session in tmp dirs.

Tests never touch ml/artifacts or data/processed.
"""
import shutil

import pandas as pd
import pytest

from ml import train
from ml.synthetic import make_synthetic

AS_OF = "2023-12-03T00:00:00+05:30"        # inside the synthetic 2023 event block
SMALL_BBOX = (13.00, 80.20, 13.09, 80.292)  # ~40 x 40 cells


@pytest.fixture(scope="session")
def env(tmp_path_factory):
    root = tmp_path_factory.mktemp("ml")
    data = root / "data"
    art = root / "artifacts"
    make_synthetic.generate(data, seed=42, bbox=SMALL_BBOX, now="2024-01-01T00:00:00+05:30")
    mp = pytest.MonkeyPatch()
    mp.setenv("ML_DATA_DIR", str(data))
    mp.setenv("ML_ARTIFACT_DIR", str(art))
    mp.setenv("ML_AS_OF", AS_OF)
    meta = train.train(data)
    yield {"root": root, "data": data, "art": art, "meta": meta}
    mp.undo()


def clone_data(env, name, rain_scale=None, zero_rain=False):
    """Copy the synthetic dir, optionally rewriting rainfall.parquet (for rainfall tests)."""
    dst = env["root"] / name
    shutil.copytree(env["data"], dst)
    if rain_scale is not None or zero_rain:
        df = pd.read_parquet(dst / "rainfall.parquet")
        df["precip_mm"] = 0.0 if zero_rain else df["precip_mm"] * rain_scale
        df.to_parquet(dst / "rainfall.parquet", index=False)
    return dst
