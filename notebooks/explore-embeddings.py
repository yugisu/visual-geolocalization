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

import marimo

__generated_with = "0.23.4"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _():
    import os
    import sys
    from pathlib import Path

    import altair as alt
    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    from dotenv import load_dotenv
    from sklearn.manifold import TSNE
    import umap

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
        alt,
        build_ground_truth,
        mo,
        np,
        pd,
        plt,
        project_root,
        umap,
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
    retrieval_k_ui = mo.ui.slider(start=1, stop=50, step=1, value=10, label="Top-k retrieval for query context")
    max_images_ui = mo.ui.slider(start=1, stop=20, step=1, value=16, label="Max selected previews")

    mo.vstack(
        [
            mo.hstack([flight_id_ui, map_scale_ui]),
            mo.hstack([chunk_pixels_ui, chunk_stride_ui, retrieval_k_ui, max_images_ui]),
        ]
    )
    return (
        chunk_pixels_ui,
        chunk_stride_ui,
        flight_id_ui,
        map_scale_ui,
        retrieval_k_ui,
    )


@app.cell(hide_code=True)
def _(mo, project_root):
    emb_dir = project_root / "embeddings"

    gallery_embs = emb_dir.glob("*-emb-gallery.npy")
    query_embs = emb_dir.glob("*-emb-query.npy")

    emb_names = [f.name.split("-emb-")[0] for f in gallery_embs]

    ok = len(emb_names) > 0
    mo.stop(
        not ok,
        mo.md(
            "Embedding files not found. Run model notebooks (like notebooks/zeroshot-dinov3-vitb.py) first to generate: "
            "embeddings/*-emb-gallery.npy and "
            "embeddings/*-emb-query.npy."
        ),
    )

    emb_names_ui = mo.ui.dropdown(
        options=emb_names,
        value="ft-dinov3-vitb-ssl4eo_ch-visloc",
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
def _(gallery_dataset, gallery_embeddings, mo, query_embeddings, uav_dataset):
    status_lines = [
        f"Gallery dataset size: {len(gallery_dataset)} | embedding rows: {gallery_embeddings.shape[0]}",
        f"Query dataset size: {len(uav_dataset)} | embedding rows: {query_embeddings.shape[0]}",
    ]
    mo.md("\n".join(status_lines))
    return


@app.cell(hide_code=True)
def _(
    build_ground_truth,
    gallery_dataset,
    gallery_embeddings,
    np,
    pd,
    query_embeddings,
    uav_dataset,
    umap,
):
    n_gallery = min(gallery_embeddings.shape[0], len(gallery_dataset))
    n_query = min(query_embeddings.shape[0], len(uav_dataset))

    gallery_trim = gallery_embeddings[:n_gallery]
    query_trim = query_embeddings[:n_query]
    all_embs = np.concatenate([gallery_trim, query_trim], axis=0)

    n_total = all_embs.shape[0]
    if n_total < 4:
        raise ValueError("Need at least 4 embeddings total to run UMAP.")

    n_neighbors = min(30, max(2, n_total - 1))
    coords = umap.UMAP(
        metric="cosine",
        n_neighbors=100,
        min_dist=0.1,
        random_state=42,
    ).fit_transform(all_embs)

    uav_coords = uav_dataset.records[["lat", "lon"]].to_numpy(dtype=float)[:n_query]
    ground_truth = build_ground_truth(uav_coords, gallery_dataset.chunk_bboxes[:n_gallery])

    sims_matrix = gallery_trim @ query_trim.T
    query_top1_idx = np.argmax(sims_matrix, axis=0)
    query_top1_correct = np.array(
        [int(query_top1_idx[q]) in set(ground_truth[q]) for q in range(n_query)],
        dtype=bool,
    )

    gallery_colors = np.full(n_gallery, "#4ECDC4", dtype=object)
    query_colors = np.where(query_top1_correct, "#2ECC71", "#E63946")

    embedding_df = pd.DataFrame(
        {
            "x": coords[:, 0],
            "y": coords[:, 1],
            "split": ["gallery"] * n_gallery + ["query"] * n_query,
            "dataset_index": np.concatenate([np.arange(n_gallery), np.arange(n_query)]),
            "display_color": np.concatenate([gallery_colors, query_colors], axis=0),
            "query_top1_correct": np.concatenate(
                [np.full(n_gallery, np.nan, dtype=object), query_top1_correct.astype(object)],
                axis=0,
            ),
        }
    ).reset_index(names="embedding_index")
    return (embedding_df,)


@app.cell(hide_code=True)
def _(alt):
    def embedding_scatter(df):
        return (
            alt.Chart(df)
            .mark_point(filled=True, size=60, opacity=0.75)
            .encode(
                x=alt.X("x:Q", title="UMAP Dimension 1"),
                y=alt.Y("y:Q", title="UMAP Dimension 2"),
                color=alt.Color("display_color:N", scale=None, legend=None),
                shape=alt.Shape(
                    "split:N",
                    scale=alt.Scale(domain=["gallery", "query"], range=["circle", "triangle-up"]),
                    legend=None,
                ),
                tooltip=[
                    alt.Tooltip("split:N", title="Set"),
                    alt.Tooltip("dataset_index:Q", title="Dataset index"),
                    alt.Tooltip("embedding_index:Q", title="Embedding index"),
                    alt.Tooltip("query_top1_correct:N", title="Top-1 in GT"),
                ],
            )
            .properties(width=900, height=400, title="Interactive UMAP Embeddings (Query: green=top-1 hit, red=miss)")
        )

    return (embedding_scatter,)


@app.cell(hide_code=True)
def _(embedding_df, embedding_scatter, mo):
    embedding_chart = mo.ui.altair_chart(embedding_scatter(embedding_df))
    embedding_chart
    return (embedding_chart,)


@app.cell(hide_code=True)
def _(
    build_ground_truth,
    embedding_chart,
    gallery_dataset,
    gallery_embeddings,
    np,
    pd,
    query_embeddings,
    retrieval_k_ui,
    uav_dataset,
):
    focus_query_idx = None
    gt_list = None
    retrieved_topk = None
    correct = None
    incorrect = None
    retrieval_k = int(retrieval_k_ui.value)

    selected_df = pd.DataFrame(embedding_chart.value) if len(embedding_chart.value) else pd.DataFrame(embedding_chart.value)

    if not selected_df.empty:
        query_rows = selected_df[selected_df["split"] == "query"]
        query_candidates = query_rows["dataset_index"].astype(int).unique().tolist()

        if len(query_candidates) == 1:
            _focus_query_idx = int(query_candidates[0])
            _n_gallery = min(gallery_embeddings.shape[0], len(gallery_dataset))
            _n_query = min(query_embeddings.shape[0], len(uav_dataset))

            if 0 <= _focus_query_idx < _n_query:
                focus_query_idx = _focus_query_idx
                _uav_coords = uav_dataset.records[["lat", "lon"]].to_numpy(dtype=float)[:_n_query]
                _ground_truth = build_ground_truth(_uav_coords, gallery_dataset.chunk_bboxes[:_n_gallery])
                gt_list = [int(idx) for idx in _ground_truth[focus_query_idx]]

                sims = gallery_embeddings[:_n_gallery] @ query_embeddings[focus_query_idx]
                ranked = np.argsort(-sims)
                retrieved_topk = [int(idx) for idx in ranked[:retrieval_k]]

                _gt_set = set(gt_list)
                correct = [idx for idx in gt_list if idx in set(retrieved_topk)]
                incorrect = [idx for idx in retrieved_topk if idx not in _gt_set]
    return (
        correct,
        focus_query_idx,
        gt_list,
        incorrect,
        retrieval_k,
        retrieved_topk,
    )


@app.cell(hide_code=True)
def _(
    build_ground_truth,
    chunk_pixels_ui,
    focus_query_idx,
    gallery_dataset,
    gallery_embeddings,
    mo,
    np,
    plt,
    query_embeddings,
    uav_dataset,
):
    _n_gallery = min(gallery_embeddings.shape[0], len(gallery_dataset))
    _n_query = min(query_embeddings.shape[0], len(uav_dataset))

    mo.stop(_n_gallery == 0 or _n_query == 0, mo.md("No embeddings available for flight trajectory visualization."))

    _gallery_trim = gallery_embeddings[:_n_gallery]
    _query_trim = query_embeddings[:_n_query]

    _sims = _gallery_trim @ _query_trim.T
    _pred_top1_idx = np.argmax(_sims, axis=0).astype(int)

    _uav_coords = uav_dataset.records[["lat", "lon"]].to_numpy(dtype=float)[:_n_query]
    _ground_truth = build_ground_truth(_uav_coords, gallery_dataset.chunk_bboxes[:_n_gallery])
    _pred_is_correct = np.array(
        [int(_pred_top1_idx[q]) in set(_ground_truth[q]) for q in range(_n_query)],
        dtype=bool,
    )
    _pred_point_colors = np.where(_pred_is_correct, "#2ECC71", "#E63946")

    _chunks = np.array(gallery_dataset._chunks[:_n_gallery], dtype=float)
    _chunk_cx = _chunks[:, 0] + 0.5 * _chunks[:, 2]
    _chunk_cy = _chunks[:, 1] + 0.5 * _chunks[:, 3]
    _pred_x = _chunk_cx[_pred_top1_idx]
    _pred_y = _chunk_cy[_pred_top1_idx]

    _sat_img = gallery_dataset._img
    _h, _w = _sat_img.shape[:2]
    _lat_min, _lon_min, _lat_max, _lon_max = gallery_dataset._bounds

    _uav_df = uav_dataset.records.iloc[:_n_query]
    _gt_x = ((_uav_df["lon"] - _lon_min) / (_lon_max - _lon_min) * _w).to_numpy(dtype=float)
    _gt_y = ((_lat_max - _uav_df["lat"]) / (_lat_max - _lat_min) * _h).to_numpy(dtype=float)

    _fig, _ax = plt.subplots(figsize=(12, 8))
    _ax.imshow(_sat_img)

    chunk_origins = np.array([(x, y) for x, y, _, _ in gallery_dataset._chunks], dtype=int)
    if len(chunk_origins) > 0:
        x_edges = np.unique(np.concatenate([chunk_origins[:, 0], chunk_origins[:, 0] + chunk_pixels_ui.value]))
        y_edges = np.unique(np.concatenate([chunk_origins[:, 1], chunk_origins[:, 1] + chunk_pixels_ui.value]))
        first_chunk_rect = (int(chunk_origins[0, 0]), int(chunk_origins[0, 1]), int(chunk_pixels_ui.value), int(chunk_pixels_ui.value))
    else:
        x_edges = np.array([], dtype=int)
        y_edges = np.array([], dtype=int)
        first_chunk_rect = None

    if len(x_edges) > 0 and len(y_edges) > 0:
        _ax.vlines(
            x_edges,
            ymin=int(y_edges.min()),
            ymax=int(y_edges.max()),
            colors="#ffffff",
            linewidth=0.25,
            alpha=0.4,
            zorder=2,
        )
        _ax.hlines(
            y_edges,
            xmin=int(x_edges.min()),
            xmax=int(x_edges.max()),
            colors="#ffffff",
            linewidth=0.25,
            alpha=0.4,
            zorder=2,
        )

    _has_selection = focus_query_idx is not None and 0 <= int(focus_query_idx) < _n_query
    # _bg_line_alpha = 0.1 if _has_selection else 0.9
    # _bg_point_alpha = 0.1 if _has_selection else 0.55
    # _pred_line_alpha = 0.1 if _has_selection else 0.55
    # _pred_point_alpha = 0.1 if _has_selection else 0.85
    # _endpoint_alpha = 0.1 if _has_selection else 1.0
    _bg_line_alpha = 0.0 if _has_selection else 0.9
    _bg_point_alpha = 0.0 if _has_selection else 0.55
    _pred_line_alpha = 0.0 if _has_selection else 0.55
    _pred_point_alpha = 0.0 if _has_selection else 0.85
    _endpoint_alpha = 0.0 if _has_selection else 1.0

    _ax.plot(_gt_x, _gt_y, color="#2ECC71", linewidth=2.0, alpha=_bg_line_alpha, label="Ground truth flight")
    _ax.scatter(_gt_x, _gt_y, color="#2ECC71", s=7, alpha=_bg_point_alpha, zorder=3)

    _ax.plot(_pred_x, _pred_y, color="#E63946", linewidth=2.0, alpha=_pred_line_alpha, label="Predicted flight (top-1)")
    _ax.scatter(_pred_x, _pred_y, c=_pred_point_colors, s=12, alpha=_pred_point_alpha, zorder=3)

    _ax.plot(_gt_x[0], _gt_y[0], "o", color="#22A85A", markersize=8, zorder=4, alpha=_endpoint_alpha)
    _ax.plot(_gt_x[-1], _gt_y[-1], "o", color="#0B6E3D", markersize=8, zorder=4, alpha=_endpoint_alpha)
    _ax.plot(
        _pred_x[0],
        _pred_y[0],
        "o",
        color=("#2ECC71" if _pred_is_correct[0] else "#EE5A65"),
        markersize=8,
        zorder=4,
        alpha=_endpoint_alpha,
    )
    _ax.plot(
        _pred_x[-1],
        _pred_y[-1],
        "o",
        color=("#2ECC71" if _pred_is_correct[-1] else "#B3252F"),
        markersize=8,
        zorder=4,
        alpha=_endpoint_alpha,
    )

    pred_chunk_idx = None

    if _has_selection:
        _q_idx = int(focus_query_idx)
        _q_correct = bool(_pred_is_correct[_q_idx])
        _q_color = "#2ECC71" if _q_correct else "#E63946"

        # Highlight the selected UAV point using correctness color.
        _ax.plot(_gt_x[_q_idx], _gt_y[_q_idx], "o", color=_q_color, markersize=10, zorder=6)

        # Draw the selected query's top-1 predicted chunk with correctness-colored border.
        pred_chunk_idx = int(_pred_top1_idx[_q_idx])
        _chunk_x, _chunk_y, _chunk_w, _chunk_h = gallery_dataset._chunks[pred_chunk_idx]
        from matplotlib.patches import Rectangle as _Rectangle

        _ax.add_patch(
            _Rectangle(
                (float(_chunk_x), float(_chunk_y)),
                float(chunk_pixels_ui.value),
                float(chunk_pixels_ui.value),
                fill=False,
                edgecolor=_q_color,
                linewidth=2.5,
                zorder=7,
            )
        )

        _ax.text(
            float(_chunk_x),
            max(0.0, float(_chunk_y) - 6.0),
            f"top-1 chunk q={_q_idx} ({'hit' if _q_correct else 'miss'})",
            color=_q_color,
            fontsize=8,
            weight="bold",
            zorder=8,
            bbox={"facecolor": "black", "alpha": 0.4, "pad": 1.5, "edgecolor": "none"},
        )

    _ax.legend(loc="upper right", framealpha=0.9)
    _ax.axis("off")
    _fig.tight_layout(pad=0)

    _fig.savefig("fig.png", dpi=250)

    #    **Full-flight overlay: predicted vs ground truth**

    #    Green path: ground-truth UAV flight path.
    #    Predicted path points: green when top-1 chunk is in GT for that query, red otherwise.
    #    If a single query is selected in the chart, its UAV point and top-1 chunk border are highlighted in green/red by correctness.

    mo.md(
        f"""
        {mo.as_html(_fig)}
        """
    )
    return (pred_chunk_idx,)


@app.cell(hide_code=True)
def _(
    alt,
    correct,
    embedding_df,
    focus_query_idx,
    gt_list,
    incorrect,
    mo,
    np,
    pred_chunk_idx,
):
    highlighted_df = embedding_df.copy()
    highlighted_df["fill_color"] = np.where(highlighted_df["split"] == "gallery", "#4ECDC422", "#FF6B6B22")
    highlighted_df["stroke_color"] = "#ffffff22"
    highlighted_df["stroke_width"] = 0.1
    highlighted_df["point_size"] = 50
    highlighted_df["opacity"] = 0.1

    if focus_query_idx is not None:
        gallery_mask = highlighted_df["split"] == "gallery"
        gt_set = set(gt_list or [])
        _correct_set = set(correct or [])
        _incorrect_set = set(incorrect or [])

        gt = gallery_mask & highlighted_df["dataset_index"].isin(gt_set)
        c_mask = gallery_mask & highlighted_df["dataset_index"].isin(_correct_set)
        ic_mask = gallery_mask & highlighted_df["dataset_index"].isin(_incorrect_set)
        q_mask = (highlighted_df["split"] == "query") & (highlighted_df["dataset_index"] == focus_query_idx)

        guessed_correctly = pred_chunk_idx in gt_set

        highlighted_df.loc[gt | c_mask | ic_mask | q_mask, "opacity"] = 0.7
        highlighted_df.loc[gt | c_mask | ic_mask | q_mask, "stroke_width"] = 1.5

        highlighted_df.loc[c_mask | ic_mask, "fill_color"] = "#4ECDC488"
        highlighted_df.loc[q_mask, "fill_color"] = "#FF6B6B"

        highlighted_df.loc[
            gt,
            "fill_color",
        ] = "#FFD166"
        highlighted_df.loc[
            gt,
            "point_size",
        ] = 200
        highlighted_df.loc[
            gt,
            "stroke_color",
        ] = "#FFFFFF88"

        highlighted_df.loc[
            c_mask,
            "stroke_color",
        ] = "#2ECC71"
        highlighted_df.loc[
            c_mask,
            "point_size",
        ] = 200
        highlighted_df.loc[
            c_mask & highlighted_df["dataset_index"] == pred_chunk_idx,
            "fill_color",
        ] = "#E63946"

        highlighted_df.loc[
            ic_mask,
            "stroke_color",
        ] = "#E63946"
        highlighted_df.loc[
            ic_mask,
            "point_size",
        ] = 200
        highlighted_df.loc[
            ic_mask & highlighted_df["dataset_index"] == pred_chunk_idx,
            "fill_color",
        ] = "#E63946"

        highlighted_df.loc[
            q_mask,
            "stroke_color",
        ] = "#000000"
        highlighted_df.loc[
            q_mask,
            "stroke_color",
        ] = "#2ECC71" if guessed_correctly else "#000000"

        highlighted_df.loc[
            q_mask,
            "point_size",
        ] = 200

        # FIXME: not working
        highlighted_df.loc[
            gallery_mask & highlighted_df["dataset_index"] == pred_chunk_idx,
            "point_size",
        ] = 300


    def context_scatter(df):
        return (
            alt.Chart(df)
            .mark_point(filled=True)
            .encode(
                x=alt.X("x:Q", title="UMAP Dimension 1"),
                y=alt.Y("y:Q", title="UMAP Dimension 2"),
                color=alt.Color("fill_color:N", scale=None, legend=None),
                stroke=alt.Color("stroke_color:N", scale=None, legend=None),
                strokeWidth=alt.StrokeWidth("stroke_width:Q", legend=None),
                shape=alt.Shape(
                    "split:N",
                    scale=alt.Scale(domain=["gallery", "query"], range=["circle", "triangle-up"]),
                    legend=alt.Legend(title="Embedding set"),
                ),
                size=alt.Size("point_size:Q", legend=None),
                opacity=alt.Opacity("opacity:Q", legend=None),
                tooltip=[
                    alt.Tooltip("split:N", title="Set"),
                    alt.Tooltip("dataset_index:Q", title="Dataset index"),
                    alt.Tooltip("embedding_index:Q", title="Embedding index"),
                ],
            )
            .properties(width=900, height=460, title="Interactive UMAP with GT/Correct/Incorrect Highlights")
        )


    context_chart = mo.ui.altair_chart(context_scatter(highlighted_df))
    mo.md(
        f"""
        **Query context chart**

        Yellow fill: ground-truth gallery chunks for the selected query.
        Green border: correctly retrieved gallery chunks within top-k.
        Red border: incorrectly retrieved gallery chunks within top-k.

        {context_chart}
        """
    )
    return


@app.cell(hide_code=True)
def _(
    correct,
    focus_query_idx,
    gallery_dataset,
    gt_list,
    mo,
    np,
    plt,
    retrieval_k,
    retrieved_topk,
    uav_dataset,
):
    mo.stop(focus_query_idx is None, mo.md("Select exactly one query point to view query-vs-ground-truth chunk details."))

    q_img, q_lat, q_lon = uav_dataset[focus_query_idx]
    q_fig, q_ax = plt.subplots(1, 1, figsize=(4.5, 4.5))
    q_ax.imshow(np.asarray(q_img))
    q_ax.set_title(f"query[{focus_query_idx}]\\n({q_lat:.5f}, {q_lon:.5f})", fontsize=10)
    q_ax.set_xticks([])
    q_ax.set_yticks([])
    q_fig.tight_layout()

    gt_ordered = list(gt_list or [])
    mo.stop(not gt_ordered, mo.md("Selected query has no ground-truth chunks under the current config."))

    _gt_set = set(int(idx) for idx in gt_ordered)
    _correct_gt_set = set(correct or [])
    retrieved_rank = {int(idx): pos + 1 for pos, idx in enumerate(retrieved_topk or [])}

    n = len(gt_ordered)
    ncols = min(6, n)
    nrows = int(np.ceil(n / ncols))
    gt_fig, axes = plt.subplots(nrows, ncols, figsize=(3.1 * ncols, 3.0 * nrows))
    axes = np.atleast_1d(axes).ravel()

    for ax in axes[n:]:
        ax.axis("off")

    for pos, (_ax, gt_idx) in enumerate(zip(axes, gt_ordered), start=1):
        sat_img, lat, lon = gallery_dataset[int(gt_idx)]
        _ax.imshow(np.asarray(sat_img))

        is_correct = int(gt_idx) in _correct_gt_set
        border_color = "#2ECC71" if is_correct else "#666666"
        rank_txt = f"R@{retrieved_rank[int(gt_idx)]}" if int(gt_idx) in retrieved_rank else "not in top-k"
        _ax.set_title(f"GT#{pos} idx={int(gt_idx)}\\n{rank_txt}", fontsize=8)
        _ax.set_xticks([])
        _ax.set_yticks([])
        for spine in _ax.spines.values():
            spine.set_visible(True)
            spine.set_linewidth(4 if is_correct else 1.0)
            spine.set_color(border_color)

    gt_fig.tight_layout()

    incorrect_retrieved = [int(idx) for idx in (retrieved_topk or []) if int(idx) not in _gt_set]
    incorrect_block = "None in top-k."
    if incorrect_retrieved:
        n_bad = len(incorrect_retrieved)
        bad_cols = min(6, n_bad)
        bad_rows = int(np.ceil(n_bad / bad_cols))
        bad_fig, bad_axes = plt.subplots(bad_rows, bad_cols, figsize=(3.1 * bad_cols, 3.0 * bad_rows))
        bad_axes = np.atleast_1d(bad_axes).ravel()

        for _ax in bad_axes[n_bad:]:
            _ax.axis("off")

        for _ax, bad_idx in zip(bad_axes, incorrect_retrieved):
            sat_img, _, _ = gallery_dataset[int(bad_idx)]
            _ax.imshow(np.asarray(sat_img))
            _ax.set_title(f"R@{retrieved_rank.get(int(bad_idx), '?')} idx={int(bad_idx)}", fontsize=8)
            _ax.set_xticks([])
            _ax.set_yticks([])
            for spine in _ax.spines.values():
                spine.set_visible(True)
                spine.set_linewidth(3.0)
                spine.set_color("#E63946")

        bad_fig.tight_layout()
        incorrect_block = mo.as_html(bad_fig)

    mo.md(
        f"""
        **Single-query drilldown**

        Top-k used for retrieval context: **{retrieval_k}**

        **Selected query image**

        {mo.as_html(q_fig)}

        **Ground-truth gallery chunks (original GT ordering; green border = correctly retrieved in top-k)**

        {mo.as_html(gt_fig)}

        {incorrect_block}
        """
    )
    return


if __name__ == "__main__":
    app.run()
