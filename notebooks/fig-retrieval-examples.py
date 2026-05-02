# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "marimo",
#     "matplotlib==3.10.9",
#     "python-dotenv==1.2.2",
#     "numpy==2.2.6",
#     "pandas==2.3.3",
#     "torch==2.3.1",
#     "torchvision==0.18.1",
#     "rasterio==1.4.4",
# ]
# ///

import marimo

__generated_with = "0.23.4"
app = marimo.App(width="medium")

@app.cell
def _():
    import os
    import sys
    from pathlib import Path

    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
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
        np,
        pd,
        plt,
        project_root,
        visloc_root,
    )

@app.cell
def _(
    SatChunkDataset,
    UAVDataset,
    build_ground_truth,
    np,
    plt,
    project_root,
    visloc_root,
):
    emb_dir = project_root / "embeddings"
    emb_name = "ft-dinov3-vitb-ssl4eo_ch-visloc"
    gallery_emb_path = emb_dir / f"{emb_name}-emb-gallery.npy"
    query_emb_path = emb_dir / f"{emb_name}-emb-query.npy"

    gallery_embeddings = np.load(gallery_emb_path)
    query_embeddings = np.load(query_emb_path)

    flight_id = "03"
    chunk_pixels = 512
    stride_pixels = 128
    map_scale = 0.25

    gallery_dataset = SatChunkDataset(
        visloc_root,
        flight_id,
        chunk_pixels=chunk_pixels,
        stride_pixels=stride_pixels,
        scale_factor=map_scale,
    )
    uav_dataset = UAVDataset(visloc_root, flight_id)

    n_gallery = min(gallery_embeddings.shape[0], len(gallery_dataset))
    n_query = min(query_embeddings.shape[0], len(uav_dataset))

    gallery_trim = gallery_embeddings[:n_gallery]
    query_trim = query_embeddings[:n_query]

    uav_coords = uav_dataset.records[["lat", "lon"]].to_numpy(dtype=float)[:n_query]
    ground_truth = build_ground_truth(uav_coords, gallery_dataset.chunk_bboxes[:n_gallery])

    sims_matrix = gallery_trim @ query_trim.T
    
    # We want top 5 chunks for each query
    # argsort is ascending, so we take the last 5 and reverse
    query_topk_idx = np.argsort(-sims_matrix, axis=0)[:5, :] # shape (5, n_query)
    
    # Let's find an "easy" query and a "hard" query
    easy_query_idx = None
    hard_query_idx = None
    
    for q in range(n_query):
        gt_set = set(int(idx) for idx in ground_truth[q])
        if not gt_set:
            continue
            
        retrieved_top5 = [int(query_topk_idx[r, q]) for r in range(5)]
        correct_count = sum(1 for idx in retrieved_top5 if idx in gt_set)
        
        top1_correct = retrieved_top5[0] in gt_set
        
        if correct_count >= 4 and top1_correct and easy_query_idx is None:
            easy_query_idx = q
            
        if correct_count <= 1 and not top1_correct and hard_query_idx is None:
            hard_query_idx = q
            
        if easy_query_idx is not None and hard_query_idx is not None:
            break
            
    # fallback if not found
    if easy_query_idx is None:
        easy_query_idx = 0
    if hard_query_idx is None:
        hard_query_idx = 1
        
    print(f"Easy query: {easy_query_idx}, Hard query: {hard_query_idx}")

    # Now create the figure
    fig, axes = plt.subplots(2, 6, figsize=(18, 6.3))
    
    queries = [easy_query_idx, hard_query_idx]
    
    for row_idx, q_idx in enumerate(queries):
        gt_set = set(int(idx) for idx in ground_truth[q_idx])
        retrieved_top5 = [int(query_topk_idx[r, q_idx]) for r in range(5)]
        
        # Plot query image
        q_img, _, _ = uav_dataset[q_idx]
        axes[row_idx, 0].imshow(np.asarray(q_img))
        
        # Adding title to the row instead of every image might look better,
        # but let's stick to the query image having the main title.
        title_prefix = "Great Match" if row_idx == 0 else "Failure"
        axes[row_idx, 0].set_title(f"Query #{q_idx}\n{title_prefix}")
        axes[row_idx, 0].axis("off")
        
        # Plot top 5 retrieved chunks
        for col_idx in range(5):
            sat_idx = retrieved_top5[col_idx]
            sat_img, _, _ = gallery_dataset[sat_idx]
            ax = axes[row_idx, col_idx + 1]
            ax.imshow(np.asarray(sat_img))
            
            is_correct = sat_idx in gt_set
            border_color = "#2ECC71" if is_correct else "#E63946"
            
            # ax.set_title(f"Top-{col_idx + 1}")
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_visible(True)
                spine.set_linewidth(4.0)
                spine.set_color(border_color)
                
    fig.tight_layout()
    output_path = project_root / "out/fig-retrieval-examples"
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"Saved figure to {output_path}")

    return (fig,)

if __name__ == "__main__":
    app.run()
