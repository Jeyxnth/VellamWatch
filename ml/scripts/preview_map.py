"""Write an HTML map (folium) of predict_risk output, coloured by band. Optional dependency.

Run: python -m ml.scripts.preview_map [--bbox MIN_LAT MIN_LON MAX_LAT MAX_LON] [--horizon 24]
                                      [--as-of ISO] [--data-dir DIR] [--out PATH]
Default bbox = extent of the data. Needs: pip install folium
"""
import argparse
import sys

from ml import config
from ml.features import data_source, find_data_dir
from ml.predict import get_state, predict_risk_at
from ml.rain import resolve_as_of

COLORS = {"low": "#2e9e44", "medium": "#f0a30a", "high": "#d62728"}


def main(argv=None):
    ap = argparse.ArgumentParser(description="HTML preview of flood-hotspot ranking")
    ap.add_argument("--bbox", nargs=4, type=float, metavar=("MIN_LAT", "MIN_LON", "MAX_LAT", "MAX_LON"))
    ap.add_argument("--horizon", type=int, default=24)
    ap.add_argument("--as-of")
    ap.add_argument("--data-dir")
    ap.add_argument("--out", default=str(config.artifact_root() / "preview.html"))
    a = ap.parse_args(argv)
    try:
        import folium
    except ImportError:
        print("folium is not installed. Run: pip install folium", file=sys.stderr)
        return 1

    d = find_data_dir(a.data_dir)
    st = get_state(d)
    bbox = a.bbox or (st.lat.min(), st.lon.min(), st.lat.max(), st.lon.max())
    cells = predict_risk_at(*bbox, a.horizon, as_of=a.as_of, data_dir=d)
    half_lat, half_lon = 125 / 111320.0, 125 / (111320.0 * 0.974)
    fmap = folium.Map(location=[(bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2], zoom_start=12, tiles="OpenStreetMap")
    for c in cells:
        tip = f"{c['cell_id']}: {c['risk_band']} ({c['risk_score']:.2f})<br>" + "<br>".join(c["drivers"])
        folium.Rectangle(
            bounds=[[c["lat"] - half_lat, c["lon"] - half_lon], [c["lat"] + half_lat, c["lon"] + half_lon]],
            color=COLORS[c["risk_band"]], weight=0, fill=True, fill_opacity=0.55, tooltip=tip).add_to(fmap)
    title = (f"{config.SYNTHETIC_BANNER}<br>" if data_source(d) == "synthetic" else "")
    title += f"Flood-hotspot RANKING (not depth) | as of {resolve_as_of(a.as_of).isoformat()} | +{a.horizon} h"
    fmap.get_root().html.add_child(folium.Element(
        f'<div style="position:fixed;top:8px;left:50px;z-index:9999;background:#fff;padding:6px 10px;'
        f'font:13px sans-serif;border:1px solid #888">{title}</div>'))
    fmap.save(a.out)
    print(f"wrote {a.out} ({len(cells)} cells)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
