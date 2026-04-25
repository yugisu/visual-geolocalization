# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "marimo",
#     "python-dotenv==1.2.2",
#     "numpy==2.2.6",
#     "matplotlib==3.10.9",
#     "pandas==2.3.3",
#     "rasterio==1.4.4",
#     "torch",
#     "torchvision",
# ]
# ///

import marimo

__generated_with = "0.23.3"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _():
    import sys
    import os
    from pathlib import Path

    import marimo as mo
    import numpy as np
    import matplotlib.pyplot as plt
    from dotenv import load_dotenv

    project_root = Path(__file__).parent.parent
    sys.path.insert(0, str(project_root))

    from lib.visloc import UAVDataset, SatChunkDataset

    load_dotenv(project_root / ".env")
    data_root = Path(os.environ["DATA_ROOT"])
    return SatChunkDataset, UAVDataset, data_root, mo, plt, project_root


@app.cell(hide_code=True)
def _(mo):
    VALIDATION_FLIGHT = "03"

    flight_dropdown = mo.ui.dropdown(
        options=["01", "02", "03", "04", "05", "06", "08", "09", "10", "11"],
        value=VALIDATION_FLIGHT,
        label="Flight ID",
    )
    map_scale_slider = mo.ui.slider(start=0.05, stop=0.5, step=0.05, value=0.15, label="Map scale")
    mo.hstack([flight_dropdown, map_scale_slider])
    return flight_dropdown, map_scale_slider


@app.cell
def _(
    SatChunkDataset,
    UAVDataset,
    data_root,
    flight_dropdown,
    map_scale_slider,
):
    flight_id = flight_dropdown.value
    visloc_root = data_root / "visloc"

    uav_ds = UAVDataset(visloc_root, flight_id)
    sat_ds = SatChunkDataset(visloc_root, flight_id, scale_factor=map_scale_slider.value)

    sat_img = sat_ds._img
    h, w = sat_img.shape[:2]
    lat_min, lon_min, lat_max, lon_max = sat_ds._bounds

    drone_df = uav_ds.records
    xs = ((drone_df["lon"] - lon_min) / (lon_max - lon_min) * w).astype(int)
    ys = ((lat_max - drone_df["lat"]) / (lat_max - lat_min) * h).astype(int)
    return sat_img, xs, ys


@app.cell
def _(mo, plt, project_root, sat_img, xs, ys):
    fig, ax = plt.subplots(figsize=(12, 8))

    ax.imshow(sat_img)
    ax.plot(xs, ys, color="#22cc22", linewidth=1.5, alpha=0.85)
    ax.scatter(xs, ys, color="#22cc22", s=8, zorder=4, alpha=0.9)
    ax.plot(xs.iloc[0], ys.iloc[0], "o", color="#ee4444", markersize=8, label="Start", zorder=5)
    ax.plot(xs.iloc[-1], ys.iloc[-1], "o", color="#4444ee", markersize=8, label="End", zorder=5)
    ax.axis("off")
    fig.tight_layout(pad=0)

    if mo.app_meta().mode == "script":
        out_path = project_root / "out" / "fig-visloc-val-flight-map.png"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        print(f"Saved: {out_path}")

    fig
    return


if __name__ == "__main__":
    app.run()
