# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "marimo",
#     "jedi<0.20.0",
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
    from matplotlib.patches import Rectangle
    from dotenv import load_dotenv

    project_root = Path(__file__).parent.parent
    sys.path.insert(0, str(project_root))

    from lib.visloc import UAVDataset, SatChunkDataset

    load_dotenv(project_root / ".env")
    data_root = Path(os.environ["DATA_ROOT"])
    return (
        Rectangle,
        SatChunkDataset,
        UAVDataset,
        data_root,
        mo,
        np,
        plt,
        project_root,
    )


@app.cell(hide_code=True)
def _(mo):
    VALIDATION_FLIGHT = "03"

    flight_dropdown = mo.ui.dropdown(
        options=["01", "02", "03", "04", "05", "06", "08", "09", "10", "11"],
        value=VALIDATION_FLIGHT,
        label="Flight ID",
    )
    map_scale_slider = mo.ui.number(start=0.05, stop=0.5, step=0.05, value=0.25, label="Map scale")
    chunk_pixels_slider = mo.ui.number(start=128, stop=1024, step=32, value=512, label="Chunk size (px)")
    chunk_stride_slider = mo.ui.number(start=16, stop=512, step=16, value=128, label="Chunk stride (px)")

    mo.vstack([
        mo.hstack([flight_dropdown, map_scale_slider]),
        mo.hstack([chunk_pixels_slider, chunk_stride_slider]),
    ])
    return (
        chunk_pixels_slider,
        chunk_stride_slider,
        flight_dropdown,
        map_scale_slider,
    )


@app.cell(hide_code=True)
def _(
    SatChunkDataset,
    UAVDataset,
    chunk_pixels_slider,
    chunk_stride_slider,
    data_root,
    flight_dropdown,
    map_scale_slider,
    np,
):
    flight_id = flight_dropdown.value
    visloc_root = data_root / "visloc"
    chunk_pixels = chunk_pixels_slider.value
    chunk_stride = chunk_stride_slider.value

    uav_ds = UAVDataset(visloc_root, flight_id)
    sat_ds = SatChunkDataset(
        visloc_root,
        flight_id,
        chunk_pixels=chunk_pixels,
        stride_pixels=chunk_stride,
        scale_factor=map_scale_slider.value,
    )

    sat_img = sat_ds._img
    h, w = sat_img.shape[:2]
    lat_min, lon_min, lat_max, lon_max = sat_ds._bounds

    chunk_origins = np.array([(x, y) for x, y, _, _ in sat_ds._chunks], dtype=int)
    if len(chunk_origins) > 0:
        x_edges = np.unique(np.concatenate([chunk_origins[:, 0], chunk_origins[:, 0] + chunk_pixels]))
        y_edges = np.unique(np.concatenate([chunk_origins[:, 1], chunk_origins[:, 1] + chunk_pixels]))
        first_chunk_rect = (int(chunk_origins[0, 0]), int(chunk_origins[0, 1]), int(chunk_pixels), int(chunk_pixels))
    else:
        x_edges = np.array([], dtype=int)
        y_edges = np.array([], dtype=int)
        first_chunk_rect = None

    drone_df = uav_ds.records
    xs = ((drone_df["lon"] - lon_min) / (lon_max - lon_min) * w).to_numpy(dtype=float)
    ys = ((lat_max - drone_df["lat"]) / (lat_max - lat_min) * h).to_numpy(dtype=float)
    return first_chunk_rect, sat_img, x_edges, xs, y_edges, ys


@app.cell(hide_code=True)
def _(
    Rectangle,
    first_chunk_rect,
    mo,
    plt,
    project_root,
    sat_img,
    x_edges,
    xs,
    y_edges,
    ys,
):
    fig, ax = plt.subplots(figsize=(12, 8))

    ax.imshow(sat_img)
    if len(x_edges) > 0 and len(y_edges) > 0:
        ax.vlines(
            x_edges,
            ymin=int(y_edges.min()),
            ymax=int(y_edges.max()),
            colors="#ffffff",
            linewidth=0.25,
            alpha=0.4,
            zorder=2,
        )
        ax.hlines(
            y_edges,
            xmin=int(x_edges.min()),
            xmax=int(x_edges.max()),
            colors="#ffffff",
            linewidth=0.25,
            alpha=0.4,
            zorder=2,
        )

    if first_chunk_rect is not None:
        chunk_x, chunk_y, chunk_w, chunk_h = first_chunk_rect
        ax.add_patch(
            Rectangle(
                (chunk_x, chunk_y),
                chunk_w,
                chunk_h,
                fill=False,
                edgecolor="#ff0000",
                linewidth=1.5,
                zorder=3,
            )
        )

    ax.plot(xs, ys, color="#ffdd00", linewidth=1, alpha=0.85, label="Flight trajectory")
    ax.scatter(xs, ys, color="#ffdd00", s=8, zorder=4, alpha=0.9)
    ax.plot(xs[0], ys[0], "o", color="#ee4444", markersize=8, label="Start", zorder=5)
    ax.plot(xs[-1], ys[-1], "o", color="#4444ee", markersize=8, label="End", zorder=5)
    ax.axis("off")
    ax.legend(loc="upper right")
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
