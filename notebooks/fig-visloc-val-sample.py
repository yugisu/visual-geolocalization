# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "marimo",
#     "python-dotenv==1.2.2",
#     "pillow==12.2.0",
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
    import random
    from pathlib import Path

    import marimo as mo
    import numpy as np
    import matplotlib.pyplot as plt
    from dotenv import load_dotenv
    from PIL import Image, ImageOps
    from PIL.Image import Resampling

    project_root = Path(__file__).parent.parent
    sys.path.insert(0, str(project_root))

    from lib.visloc import UAVDataset, SatChunkDataset, SAT_SCALES

    load_dotenv(project_root / ".env")
    data_root = Path(os.environ["DATA_ROOT"])
    return (
        ImageOps,
        SAT_SCALES,
        SatChunkDataset,
        UAVDataset,
        data_root,
        mo,
        np,
        plt,
        project_root,
        random,
    )


@app.cell(hide_code=True)
def _(mo):
    flight_dropdown = mo.ui.dropdown(
        options=["01", "02", "03", "04", "05", "06", "08", "09", "10", "11"],
        value="03",
        label="Flight ID",
    )
    seed_input = mo.ui.number(value=42, start=0, stop=9999, step=1, label="Random seed")
    mo.hstack([flight_dropdown, seed_input])
    return flight_dropdown, seed_input


@app.cell(hide_code=True)
def _(
    SAT_SCALES,
    SatChunkDataset,
    UAVDataset,
    data_root,
    flight_dropdown,
    np,
    random,
    seed_input,
):
    flight_id = flight_dropdown.value
    visloc_root = data_root / "visloc"
    random.seed(seed_input.value)


    uav_ds = UAVDataset(
        visloc_root,
        flight_id,
    )
    sat_ds = SatChunkDataset(
        visloc_root,
        flight_id,
        chunk_pixels=512,
        stride_pixels=128,
        scale_factor=SAT_SCALES[flight_id],
    )

    indices = random.sample(range(len(uav_ds)), 4)
    uav_samples = [uav_ds[i] for i in indices]
    sat_coords = np.array(sat_ds.chunk_coords)

    pairs = []
    for uav_img, lat, lon in uav_samples:
        dists = np.hypot(sat_coords[:, 0] - lat, sat_coords[:, 1] - lon)
        closest = int(np.argmin(dists))
        sat_chunk, _, _ = sat_ds[closest]
        pairs.append((uav_img, sat_chunk))
    return (pairs,)


@app.cell(hide_code=True)
def _(ImageOps, mo, pairs, plt, project_root):
    DISPLAY_PX = 512

    fig, axes = plt.subplots(2, 4, figsize=(12, 6), gridspec_kw={"hspace": 0.04, "wspace": 0.04})

    row_labels = ["Drone", "Satellite"]
    for col, (drone_img, sat_img) in enumerate(pairs):
        for ri, img in enumerate([drone_img, sat_img]):
            if col == 0:
                print(f"{row_labels[ri]} image size:", img.size)
        
            ax = axes[ri, col]
            ax.imshow(ImageOps.fit(img, (DISPLAY_PX, DISPLAY_PX)))
            ax.set_xticks([])
            ax.set_yticks([])
            if col == 0:
                ax.set_ylabel(row_labels[ri], fontsize=9, labelpad=4)
            for spine in ax.spines.values():
                spine.set_visible(False)

    if mo.app_meta().mode == "script":
        out_path = project_root / "out" / "fig-visloc-val-sample.png"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=250, bbox_inches="tight")
        print(f"Saved: {out_path}")

    fig
    return


if __name__ == "__main__":
    app.run()
