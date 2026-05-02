# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "marimo",
#     "jedi<0.20.0",
#     "altair==6.1.0",
#     "matplotlib==3.10.9",
#     "scikit-learn==1.7.2",
#     "umap-learn==0.5.12",
#     "pyarrow>=16.0.0",
#     "python-dotenv==1.2.2",
#     "numpy==2.2.6",
#     "pandas==2.3.3",
#     "torch==2.3.1",
#     "torchvision==0.18.1",
#     "rasterio==1.4.4",
# ]
# ///


# FIXME: NOT WORKING

import marimo

__generated_with = "0.23.4"
app = marimo.App(width="medium")


@app.cell
def _():
    import os
    import sys
    from pathlib import Path

    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np
    from dotenv import load_dotenv

    project_root = Path(__file__).parent.parent
    sys.path.insert(0, str(project_root))

    from lib.visloc import SatChunkDataset, UAVDataset
    from lib.evaluation import build_ground_truth

    load_dotenv(project_root / ".env")
    data_root = Path(os.environ["DATA_ROOT"])
    visloc_root = data_root / "visloc"
    return (
        SatChunkDataset,
        UAVDataset,
        build_ground_truth,
        mo,
        np,
        plt,
        project_root,
        visloc_root,
    )


@app.cell(hide_code=True)
def _(mo, project_root):
    emb_dir = project_root / "embeddings"

    gallery_cls_path = emb_dir / "ft-dinov3-vitb-ssl4eo_ch-visloc-emb-gallery.npy"
    query_cls_path = emb_dir / "ft-dinov3-vitb-ssl4eo_ch-visloc-emb-query.npy"
    gallery_patch_path = emb_dir / "ft-dinov3-vitb-ssl4eo_ch-visloc-patch-emb-gallery.npy"
    query_patch_path = emb_dir / "ft-dinov3-vitb-ssl4eo_ch-visloc-patch-emb-query.npy"

    paths_ok = all(p.exists() for p in [gallery_cls_path, query_cls_path, gallery_patch_path, query_patch_path])

    mo.stop(
        not paths_ok,
        mo.md("Embedding files not found. Make sure ft-dinov3-vitb-ssl4eo_ch-visloc.py finishes running to save patch embeddings.")
    )
    return (
        gallery_cls_path,
        gallery_patch_path,
        query_cls_path,
        query_patch_path,
    )


@app.cell(hide_code=True)
def _(
    gallery_cls_path,
    gallery_patch_path,
    np,
    query_cls_path,
    query_patch_path,
):
    gallery_cls = np.load(gallery_cls_path)
    query_cls = np.load(query_cls_path)

    gallery_patches = np.load(gallery_patch_path, mmap_mode='r')
    query_patches = np.load(query_patch_path, mmap_mode='r')
    return gallery_cls, gallery_patches, query_cls, query_patches


@app.cell(hide_code=True)
def _(SatChunkDataset, UAVDataset, visloc_root):
    flight_id = "03"
    gallery_dataset = SatChunkDataset(
        visloc_root,
        flight_id,
        chunk_pixels=512,
        stride_pixels=128,
        scale_factor=0.25,
    )
    uav_dataset = UAVDataset(visloc_root, flight_id)
    return gallery_dataset, uav_dataset


@app.cell(hide_code=True)
def _(build_ground_truth, gallery_dataset, uav_dataset):
    n_gallery = len(gallery_dataset)
    n_query = len(uav_dataset)

    uav_coords = uav_dataset.records[["lat", "lon"]].to_numpy(dtype=float)[:n_query]
    ground_truth = build_ground_truth(uav_coords, gallery_dataset.chunk_bboxes[:n_gallery])
    return (ground_truth,)


@app.cell(hide_code=True)
def _(
    gallery_cls,
    gallery_patches,
    ground_truth,
    np,
    query_cls,
    query_patches,
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
        """
        Vectorized and batched Chamfer similarity reranking over top-K candidates.

        Args:
            sims: [N_q, N_g] global similarity matrix.
            q_patches: [N_q, P, D] patch embeddings for queries.
            g_patches: [N_g, P, D] patch embeddings for gallery.
            K: number of top candidates to rerank.
            alpha: weight for blending global and patch similarity.
            batch_size: query batch size to prevent OOM during [N_q, K, P, P] computation.
            device: device to run the batched matrix multiplications on. 
                    Defaults to "cuda" if available, else CPU.

        Returns:
            [N_q, K] array of reranked gallery indices.
        """
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            device = torch.device(device)

        N_q = len(sims)
        preds = np.argsort(-sims, axis=1)[:, :K]
        preds = preds.cpu().numpy()
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


    _sims = gallery_cls @ query_cls.T
    _sims_t = torch.from_numpy(_sims).T # [N_q, N_g]

    _q_patches = torch.from_numpy(query_patches).float()
    _g_patches = torch.from_numpy(gallery_patches).float()

    # We use ALPHA=0.2 and K=50 to match the plotting
    _new_preds = chamfer_rerank(_sims_t, _q_patches, _g_patches, K=50, alpha=0.2)
    _new_top1 = _new_preds[:, 0].numpy()

    _old_preds = torch.argsort(_sims_t, dim=-1, descending=True)[:, 0].numpy()

    changed_indices = []
    for _q_idx in range(len(query_cls)):
        _gt_for_q = ground_truth[_q_idx]
        _old_correct = _old_preds[_q_idx] in _gt_for_q
        _new_correct = _new_top1[_q_idx] in _gt_for_q

        if _old_correct != _new_correct:
            changed_indices.append(int(_q_idx))
    return (changed_indices,)


@app.cell(hide_code=True)
def _(changed_indices, mo):
    options = {f"Query #{idx}": idx for idx in changed_indices}
    query_slider = mo.ui.dropdown(
        options=options, 
        value=list(options.keys())[0] if options else None, 
        label="Select Query (R@1 changed)"
    )
    query_slider
    return (query_slider,)


@app.cell(hide_code=True)
def _(gallery_cls, np, query_cls, query_slider):
    q_idx = query_slider.value
    q_emb = query_cls[q_idx]
    sims = gallery_cls @ q_emb
    top_50_idx = np.argsort(-sims)[:50]
    return q_idx, sims, top_50_idx


@app.cell(hide_code=True)
def _(
    gallery_dataset,
    gallery_patches,
    ground_truth,
    np,
    plt,
    q_idx,
    query_patches,
    sims,
    top_50_idx,
    uav_dataset,
):
    ALPHA = 0.2

    gt_for_q = ground_truth[q_idx]

    uav_img_pil, _, _ = uav_dataset[q_idx]
    q_p = query_patches[q_idx]

    P = q_p.shape[0]
    grid_size = int(np.sqrt(P))
    num_registers = P - (grid_size * grid_size)

    spatial_q_p = q_p[num_registers:]

    candidates = []
    for g_idx in top_50_idx:
        g_p = gallery_patches[g_idx]
        spatial_g_p = g_p[num_registers:]
        sim_mat = spatial_q_p @ spatial_g_p.T

        patch_sim = sim_mat.max(axis=1).mean()
        blended_sim = (1 - ALPHA) * sims[g_idx] + ALPHA * patch_sim
        sat_heatmap = sim_mat.max(axis=0).reshape((grid_size, grid_size))

        candidates.append({
            'g_idx': g_idx,
            'global_sim': sims[g_idx],
            'blended_sim': blended_sim,
            'heatmap': sat_heatmap
        })

    reranked = sorted(candidates, key=lambda x: x['blended_sim'], reverse=True)

    fig, axes = plt.subplots(5, 5, figsize=(20, 22))

    for i in range(5):
        axes[0, i].axis('off')

    axes[0, 2].imshow(np.array(uav_img_pil))
    axes[0, 2].set_title(f"UAV Query #{q_idx}", fontsize=16)

    def plot_cand(ax, cand, title_prefix):
        g_idx = cand['g_idx']
        sat_img_pil, _, _ = gallery_dataset[g_idx]
        sat_img = np.array(sat_img_pil)

        ax.imshow(sat_img)
        h, w, _ = sat_img.shape
        ax.imshow(cand['heatmap'], cmap='jet', alpha=0.4, extent=[0, w, h, 0])

        is_gt = "★ GT" if g_idx in gt_for_q else ""
        color = "green" if is_gt else "black"
        ax.set_title(f"{title_prefix}: Chunk {g_idx} {is_gt}", color=color, fontsize=12)
        ax.axis('off')

    for i in range(10):
        row = 1 + (i // 5)
        col = i % 5
        plot_cand(axes[row, col], candidates[i], title_prefix=f"Reg Rank {i+1}")

    for i in range(10):
        row = 3 + (i // 5)
        col = i % 5
        plot_cand(axes[row, col], reranked[i], title_prefix=f"Re-rank {i+1}")

    fig.tight_layout()
    return (fig,)


@app.cell
def _(fig, mo):
    mo.md(f"""
    # Chamfer Similarity Heatmaps

    {mo.as_html(fig)}
    """)
    return


if __name__ == "__main__":
    app.run()
