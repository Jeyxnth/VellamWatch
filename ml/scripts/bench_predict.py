"""Time predict_risk on a full-Chennai-size SYNTHETIC grid (~23k cells). Pipeline timing only.

Run: python -m ml.scripts.bench_predict [--repeats 7]
Generates data and a model in a temp folder; nothing in ml/artifacts or data/processed is touched.
"""
import argparse
import os
import statistics
import sys
import tempfile
import time
import warnings
from pathlib import Path

from ml import config


def main(argv=None):
    ap = argparse.ArgumentParser(description="Latency benchmark for predict_risk (synthetic data)")
    ap.add_argument("--repeats", type=int, default=7)
    a = ap.parse_args(argv)
    print(f"*** {config.SYNTHETIC_BANNER} (timing only) ***")
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        os.environ["ML_DATA_DIR"] = str(tmp / "data")
        os.environ["ML_ARTIFACT_DIR"] = str(tmp / "artifacts")
        os.environ["ML_AS_OF"] = "2023-12-03T00:00:00+05:30"
        from ml import train
        from ml.predict import predict_risk
        from ml.synthetic import make_synthetic as ms

        info = ms.generate(tmp / "data", bbox=ms.FULL_BBOX)
        print(f"grid: {info['cells']} cells")
        warnings.simplefilter("ignore")
        t = time.perf_counter()
        train.train(tmp / "data")
        print(f"train: {time.perf_counter() - t:.1f} s")

        t = time.perf_counter()
        predict_risk(*ms.FULL_BBOX, 24)
        print(f"warm-up (load model + features + first call): {(time.perf_counter() - t) * 1000:.0f} ms")
        views = {"whole grid (~23k cells)": ms.FULL_BBOX,
                 "viewport (~1k cells)": (13.00, 80.20, 13.06, 80.25)}
        for name, box in views.items():
            times = []
            for _ in range(a.repeats):
                t = time.perf_counter()
                n = len(predict_risk(*box, 24))
                times.append((time.perf_counter() - t) * 1000)
            print(f"{name}: {n} cells, median {statistics.median(times):.0f} ms, max {max(times):.0f} ms")
    return 0


if __name__ == "__main__":
    sys.exit(main())
