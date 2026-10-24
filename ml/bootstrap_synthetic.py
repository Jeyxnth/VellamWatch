"""One command to get a working SYNTHETIC model (for backend / frontend development).

    python -m ml.bootstrap_synthetic [--full] [--seed 42]

Runs ml.synthetic.make_synthetic (-> ml/synthetic_out/) and then ml.train
(-> ml/artifacts/model_synthetic/), warms the predictor up and prints the data source.
After it, `from ml.predict import predict_risk` works. Nothing it produces is a real result.
Tip: the synthetic forecast block starts at generation time and lasts 72 h, so re-run this to refresh it,
or set ML_AS_OF=2023-12-03T00:00:00+05:30 to replay the synthetic 2023 event at any time.
"""
import argparse
import sys
import warnings

from ml import config, train
from ml.features import data_source
from ml.predict import get_model_info, warm_up
from ml.synthetic import make_synthetic as ms


def main(argv=None):
    ap = argparse.ArgumentParser(description="Generate synthetic data and train a synthetic model")
    ap.add_argument("--full", action="store_true", help="whole contract bounding box (~23k cells) instead of the sub-area")
    ap.add_argument("--seed", type=int, default=config.SEED)
    ap.add_argument("--data-dir", default=str(config.SYNTHETIC_DIR))
    a = ap.parse_args(argv)

    print(f"*** {config.SYNTHETIC_BANNER} ***")
    info = ms.generate(a.data_dir, a.seed, ms.FULL_BBOX if a.full else ms.SUB_BBOX)
    print(f"1/3 data   : {a.data_dir} ({info['cells']} cells)")
    meta = train.train(a.data_dir)
    print(f"2/3 trained: {config.model_dir(meta['data_source'])}")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        wu = warm_up(a.data_dir)
    print(f"3/3 warm-up: {wu['warm_up_s']} s")
    assert data_source(a.data_dir) == "synthetic" == get_model_info(a.data_dir)["data_source"]
    print(f"data source: {wu['data_source'].upper()} (rain stage: {wu['rain_stage']})")
    print("Ready. In Python: from ml.predict import predict_risk, warm_up, get_model_info")
    print("Pin the clock for demos: ML_AS_OF=2023-12-03T00:00:00+05:30")
    return 0


if __name__ == "__main__":
    sys.exit(main())
