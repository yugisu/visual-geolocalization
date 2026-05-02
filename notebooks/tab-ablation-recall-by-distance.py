# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "marimo",
#     "jedi<0.20.0",
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
        value="ft-dinov3-vitb-ssl4eo_ch-visloc-tta" if "ft-dinov3-vitb-ssl4eo_ch-visloc-tta" in emb_names else (emb_names[0] if emb_names else None),
        label="Embeddings name",
    )

    emb_names_ui
    return emb_dir, emb_names_ui


@app.cell(hide_code=True)
def _(emb_dir, emb_names_ui, mo, np):
    gallery_emb_path = emb_dir / f"{emb_names_ui.value}-emb-gallery.npy"
    query_emb_path = emb_dir / f"{emb_names_ui.value}-emb-query.npy"
    gallery_patch_path = emb_dir / f"{emb_names_ui.value}-patch-emb-gallery.npy"
    query_patch_path = emb_dir / f"{emb_names_ui.value}-patch-emb-query.npy"

    paths_ok = (
        gallery_emb_path.exists() and 
        query_emb_path.exists() and 
        gallery_patch_path.exists() and 
        query_patch_path.exists()
    )
    mo.stop(
        not paths_ok,
        mo.md(f"Embedding files not found: {gallery_emb_path}, {query_emb_path}, {gallery_patch_path}, {query_patch_path}."),
    )

    gallery_embeddings = np.load(gallery_emb_path)
    query_embeddings = np.load(query_emb_path)
    gallery_patch_embeddings = np.load(gallery_patch_path, mmap_mode='r')
    query_patch_embeddings = np.load(query_patch_path, mmap_mode='r')

    print(f"Loaded gallery embeddings: {gallery_embeddings.shape} from {gallery_emb_path}")
    print(f"Loaded query embeddings:   {query_embeddings.shape} from {query_emb_path}")
    print(f"Loaded gallery patch embeddings: {gallery_patch_embeddings.shape} from {gallery_patch_path}")
    print(f"Loaded query patch embeddings:   {query_patch_embeddings.shape} from {query_patch_path}")
    return (
        gallery_embeddings,
        gallery_patch_embeddings,
        query_embeddings,
        query_patch_embeddings,
    )


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
    gallery_patch_embeddings,
    np,
    pd,
    project_root,
    query_embeddings,
    query_patch_embeddings,
    uav_dataset,
):
    import torch

    def chamfer_rerank(
        sims: np.ndarray,
        q_patches: torch.Tensor,
        g_patches: torch.Tensor,
        K: int,
        alpha: float,
        batch_size: int = 32,
        device: str | torch.device | None = None,
    ) -> np.ndarray:
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            device = torch.device(device)

        N_q = len(sims)
        preds = np.argsort(-sims, axis=1)[:, :K]
        new_preds = np.zeros_like(preds)

        top_k_tensor = torch.from_numpy(preds)

        for i in range(0, N_q, batch_size):
            end = min(i + batch_size, N_q)
            q_p = q_patches[i:end].to(device).unsqueeze(1)  # [B, 1, P, D]
            g_p_chunk = g_patches[top_k_tensor[i:end]].to(device) # [B, K, P, D]

            sim_mat = q_p @ g_p_chunk.transpose(-1, -2)  # [B, K, P_q, P_g]
            patch_sims = sim_mat.max(dim=-1).values.mean(dim=-1)  # [B, K]

            idx_b = np.arange(i, end)[:, None]
            k_idx_b = preds[i:end]
            global_sims = sims[idx_b, k_idx_b]

            blended_sims = alpha * global_sims + (1 - alpha) * patch_sims.cpu().numpy()

            sort_idx = np.argsort(-blended_sims, axis=1)  # [B, K]
            new_preds[i:end] = np.take_along_axis(preds[i:end], sort_idx, axis=1)

        return new_preds

    n_gallery = min(gallery_embeddings.shape[0], len(gallery_dataset))
    n_query = min(query_embeddings.shape[0], len(uav_dataset))

    gallery_trim = gallery_embeddings[:n_gallery]
    query_trim = query_embeddings[:n_query]
    gallery_patch_trim = gallery_patch_embeddings[:n_gallery]
    query_patch_trim = query_patch_embeddings[:n_query]

    # Compute similarity and predictions
    # query_trim: (n_query, embed_dim)
    # gallery_trim: (n_gallery, embed_dim)
    sims = query_trim @ gallery_trim.T  # shape: (n_query, n_gallery)
    preds = np.argsort(-sims, axis=1)

    q_patches_tensor = torch.from_numpy(query_patch_trim).float()
    g_patches_tensor = torch.from_numpy(gallery_patch_trim).float()

    reranked_topk_preds = chamfer_rerank(
        sims,
        q_patches_tensor,
        g_patches_tensor,
        K=50,
        alpha=0.5
    )

    preds_reranked = preds.copy()
    preds_reranked[:, :50] = reranked_topk_preds

    uav_coords = uav_dataset.records[["lat", "lon"]].to_numpy(dtype=float)[:n_query]
    chunk_bboxes = gallery_dataset.chunk_bboxes[:n_gallery]

    thresholds = [None, 400.0, 300.0, 200.0, 150.0, 100.0, 50.0, 10.0]

    results = []
    for th in thresholds:
        metrics_baseline = calculate_metrics(
            preds=preds,
            uav_coords=uav_coords,
            chunk_bboxes=chunk_bboxes,
            dist_threshold=th
        )
        metrics_reranked = calculate_metrics(
            preds=preds_reranked,
            uav_coords=uav_coords,
            chunk_bboxes=chunk_bboxes,
            dist_threshold=th
        )

        definition = "bbox" if th is None else f"{int(th)}m"
        results.append({
            "Positive chunk definition": definition,
            "Baseline R@1": metrics_baseline.get("Recall@1", 0.0),
            "Reranked R@1": metrics_reranked.get("Recall@1", 0.0),
        })

    results_df = pd.DataFrame(results)

    out_dir = project_root / "out"
    out_dir.mkdir(exist_ok=True, parents=True)
    out_csv = out_dir / "tab-ablation-recall-by-distance.csv"
    results_df.to_csv(out_csv, index=False)
    print(f"Saved results to {out_csv}")

    print(results_df)
    return chamfer_rerank, results_df


@app.cell(hide_code=True)
def _(mo, results_df):
    mo.md(f"""
    ### Ablation Study: Impact of Positive Radius on R@1

    {mo.as_html(results_df)}
    """)
    return


if __name__ == "__main__":
    app.run()
