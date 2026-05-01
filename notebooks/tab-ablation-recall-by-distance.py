# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "marimo",
#     "numpy==2.2.6",
#     "pandas==2.3.3",
#     "python-dotenv==1.2.2",
#     "torch==2.3.1",
#     "torchvision==0.18.1",
#     "rasterio==1.4.4",
# ]
# ///

import marimo

__generated_with = "0.23.4"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _():
    import os
    import sys
    from pathlib import Path

    import marimo as mo
    import numpy as np
    import pandas as pd
    from dotenv import load_dotenv

    project_root = Path(__file__).parent.parent
    sys.path.insert(0, str(project_root))

    from lib.visloc import SatChunkDataset, UAVDataset
    from lib.evaluation import calculate_metrics

    load_dotenv(project_root / ".env")
    data_root = Path(os.environ["DATA_ROOT"])
    visloc_root = data_root / "visloc"
    return (
        SatChunkDataset,
        UAVDataset,
        calculate_metrics,
        mo,
        np,
        pd,
        project_root,
        visloc_root,
    )


@app.cell(hide_code=True)
def _(mo):
    flight_id_ui = mo.ui.dropdown(
        options=["03"],
        value="03",
        label="Flight ID",
    )
    chunk_pixels_ui = mo.ui.number(
        start=128,
        stop=1024,
        step=32,
        value=512,
        label="Chunk pixels",
        disabled=True,
    )
    chunk_stride_ui = mo.ui.number(
        start=16,
        stop=512,
        step=16,
        value=128,
        label="Chunk stride",
        disabled=True,
    )
    map_scale_ui = mo.ui.number(
        start=0.05,
        stop=0.5,
        step=0.025,
        value=0.25,
        label="Map scale",
        disabled=True,
    )

    mo.vstack(
        [
            mo.hstack([flight_id_ui, map_scale_ui]),
            mo.hstack([chunk_pixels_ui, chunk_stride_ui]),
        ]
    )
    return chunk_pixels_ui, chunk_stride_ui, flight_id_ui, map_scale_ui


@app.cell(hide_code=True)
def _(mo, project_root):
    emb_dir = project_root / "embeddings"

    gallery_embs = list(emb_dir.glob("*-emb-gallery.npy"))
    query_embs = list(emb_dir.glob("*-emb-query.npy"))

    emb_names = [f.name.split("-emb-")[0] for f in gallery_embs]

    ok = len(emb_names) > 0
    mo.stop(
        not ok,
        mo.md(
            "Embedding files not found. Run model notebooks first to generate embeddings."
        ),
    )

    emb_names_ui = mo.ui.dropdown(
        options=emb_names,
        value="ft-dinov3-vitb-ssl4eo_ch-visloc" if "ft-dinov3-vitb-ssl4eo_ch-visloc" in emb_names else (emb_names[0] if emb_names else None),
        label="Embeddings name",
    )

    emb_names_ui
    return emb_dir, emb_names_ui


@app.cell(hide_code=True)
def _(emb_dir, emb_names_ui, mo, np):
    gallery_emb_path = emb_dir / f"{emb_names_ui.value}-emb-gallery.npy"
    query_emb_path = emb_dir / f"{emb_names_ui.value}-emb-query.npy"

    paths_ok = gallery_emb_path.exists() and query_emb_path.exists()
    mo.stop(
        not paths_ok,
        mo.md(f"Embedding files not found: {gallery_emb_path}, {query_emb_path}."),
    )

    gallery_embeddings = np.load(gallery_emb_path)
    query_embeddings = np.load(query_emb_path)

    print(f"Loaded gallery embeddings: {gallery_embeddings.shape} from {gallery_emb_path}")
    print(f"Loaded query embeddings:   {query_embeddings.shape} from {query_emb_path}")
    return gallery_embeddings, query_embeddings


@app.cell(hide_code=True)
def _(
    SatChunkDataset,
    UAVDataset,
    chunk_pixels_ui,
    chunk_stride_ui,
    flight_id_ui,
    map_scale_ui,
    visloc_root,
):
    flight_id = flight_id_ui.value
    gallery_dataset = SatChunkDataset(
        visloc_root,
        flight_id,
        chunk_pixels=chunk_pixels_ui.value,
        stride_pixels=chunk_stride_ui.value,
        scale_factor=map_scale_ui.value,
    )
    uav_dataset = UAVDataset(visloc_root, flight_id)
    return gallery_dataset, uav_dataset


@app.cell(hide_code=True)
def _(
    calculate_metrics,
    gallery_dataset,
    gallery_embeddings,
    np,
    pd,
    query_embeddings,
    uav_dataset,
):
    n_gallery = min(gallery_embeddings.shape[0], len(gallery_dataset))
    n_query = min(query_embeddings.shape[0], len(uav_dataset))

    gallery_trim = gallery_embeddings[:n_gallery]
    query_trim = query_embeddings[:n_query]

    # Compute similarity and predictions
    # query_trim: (n_query, embed_dim)
    # gallery_trim: (n_gallery, embed_dim)
    sims = query_trim @ gallery_trim.T  # shape: (n_query, n_gallery)
    preds = np.argsort(-sims, axis=1)

    uav_coords = uav_dataset.records[["lat", "lon"]].to_numpy(dtype=float)[:n_query]
    chunk_bboxes = gallery_dataset.chunk_bboxes[:n_gallery]

    thresholds = [None, 400.0, 300.0, 200.0, 150.0, 100.0, 50.0, 10.0]

    results = []
    for th in thresholds:
        metrics = calculate_metrics(
            preds=preds,
            uav_coords=uav_coords,
            chunk_bboxes=chunk_bboxes,
            dist_threshold=th
        )

        definition = "bbox" if th is None else f"{int(th)}m"
        results.append({
            "Positive chunk definition": definition,
            "R@1": metrics.get("Recall@1", 0.0),
        })

    results_df = pd.DataFrame(results)

    print(results_df)
    return (results_df,)


@app.cell(hide_code=True)
def _(mo, results_df):
    mo.md(f"""
    ### Ablation Study: Impact of Positive Radius on R@1

    {mo.as_html(results_df)}
    """)
    return


if __name__ == "__main__":
    app.run()
