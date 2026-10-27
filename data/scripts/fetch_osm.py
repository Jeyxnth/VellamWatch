from pathlib import Path
import time
import requests
import geopandas as gpd
from shapely.geometry import LineString

ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = ROOT / "data" / "raw"

SOUTH, WEST, NORTH, EAST = 12.85, 80.05, 13.25, 80.35

QUERY = f"""
[out:json][timeout:90];
(
  way["highway"]({SOUTH},{WEST},{NORTH},{EAST});
  way["waterway"]({SOUTH},{WEST},{NORTH},{EAST});
);
out body geom;
"""

ENDPOINTS = [
    "https://overpass.private.coffee/api/interpreter",
    "https://maps.mail.ru/osm/tools/overpass/api/interpreter",
    "https://overpass-api.de/api/interpreter",
]

HEADERS = {
    "User-Agent": "VellamWatchDataPipeline/1.0 (student flood-risk research project)",
    "Accept": "application/json",
    "Content-Type": "application/x-www-form-urlencoded",
}

def main():
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    data = None

    for url in ENDPOINTS:
        print(f"Trying: {url}")
        try:
            response = requests.post(
                url,
                data={"data": QUERY},
                headers=HEADERS,
                timeout=120,
            )

            if response.status_code == 406:
                print("Server rejected request (406); waiting before next endpoint.")
                time.sleep(30)
                continue

            response.raise_for_status()
            data = response.json()
            break

        except requests.RequestException as exc:
            print(f"Request failed: {exc}")

    if data is None:
        raise RuntimeError(
            "All Overpass endpoints failed. No output files were created."
        )

    records = []
    for element in data.get("elements", []):
        points = element.get("geometry", [])
        if len(points) < 2:
            continue

        tags = element.get("tags", {})
        records.append({
            "osm_id": element["id"],
            "feature_type": "road" if "highway" in tags else "waterway",
            "highway": tags.get("highway"),
            "waterway": tags.get("waterway"),
            "name": tags.get("name"),
            "geometry": LineString([
                (point["lon"], point["lat"]) for point in points
            ]),
        })

    if not records:
        raise RuntimeError("The server returned no usable road or waterway lines.")

    features = gpd.GeoDataFrame(
        records, geometry="geometry", crs="EPSG:4326"
    )

    roads = features[features["feature_type"] == "road"].copy()
    waterways = features[features["feature_type"] == "waterway"].copy()

    roads.to_file(RAW_DIR / "roads.geojson", driver="GeoJSON")
    waterways.to_file(RAW_DIR / "waterways.geojson", driver="GeoJSON")

    print("Road features:", len(roads))
    print("Waterway features:", len(waterways))
    print("Saved to:", RAW_DIR)

if __name__ == "__main__":
    main()
