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
#     "torchvision==0.23.0",
#     "rasterio==1.4.4",
#     "lomatch",
#     "opencv-python==4.13.0.92",
#     "torch==2.8.0",
#     "pillow==12.2.0",
# ]
#
# [tool.uv.sources]
# lomatch = { git = "https://github.com/davnords/LoMa.git" }
# ///

import marimo

__generated_with = "0.23.4"
app = marimo.App(width="medium")

with app.setup:
    import os
    import sys
    import torch
    from pathlib import Path

    import altair as alt
    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    from dotenv import load_dotenv
    from loma import LoMa, LoMaR
    import cv2

    project_root = Path(__file__).parent.parent
    sys.path.insert(0, str(project_root))

    from lib.visloc import SatChunkDataset, UAVDataset
    from lib.evaluation import build_ground_truth, calculate_metrics

    load_dotenv(project_root / ".env")
    data_root = Path(os.environ["DATA_ROOT"])
    visloc_root = data_root / "visloc"


@app.cell(hide_code=True)
def _():
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
    )

    mo.vstack(
        [
            mo.hstack([flight_id_ui, map_scale_ui]),
            mo.hstack([chunk_pixels_ui, chunk_stride_ui]),
        ]
    )
    return chunk_pixels_ui, chunk_stride_ui, flight_id_ui, map_scale_ui


@app.cell(hide_code=True)
def _():
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


@app.cell
def _(emb_dir, emb_names_ui):
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


@app.cell
def _(chunk_pixels_ui, chunk_stride_ui, flight_id_ui, map_scale_ui):
    flight_id = flight_id_ui.value
    sat_dataset = SatChunkDataset(
        visloc_root,
        flight_id,
        chunk_pixels=chunk_pixels_ui.value,
        stride_pixels=chunk_stride_ui.value,
        scale_factor=map_scale_ui.value,
    )
    uav_dataset = UAVDataset(visloc_root, flight_id)
    return sat_dataset, uav_dataset


@app.cell
def _(gallery_embeddings, query_embeddings, sat_dataset, uav_dataset):
    status_lines = [
        f"Gallery dataset size: {len(sat_dataset)} | embedding rows: {gallery_embeddings.shape[0]}",
        f"Query dataset size: {len(uav_dataset)} | embedding rows: {query_embeddings.shape[0]}",
    ]
    mo.md("\n".join(status_lines))
    return


@app.cell
def _(gallery_embeddings, query_embeddings, sat_dataset, uav_dataset):
    uav_coords = uav_dataset.records[["lat", "lon"]].to_numpy(dtype=float)
    ground_truth = build_ground_truth(uav_coords, sat_dataset.chunk_bboxes)

    sims = query_embeddings @ gallery_embeddings.T
    preds = np.argsort(-sims, axis=1)

    metrics_bare = calculate_metrics(preds, uav_coords, sat_dataset.chunk_bboxes)

    metrics_bare
    return ground_truth, preds, sims, uav_coords


@app.cell
def _(sat_dataset, sims, uav_coords):
    # Filter available gallery chunks within the radius of N meters, narrowing search field.
    # Q: Does this make sense? We know the approximate location from other sensors, and we need to correct this approximate location. I assume we can add random coordinate noise when evaluating to bring this closer to "real-world"?
    _filter_radius = 2000  # m

    _gt = build_ground_truth(uav_coords, sat_dataset.chunk_bboxes, _filter_radius)

    _mask = np.zeros((768, 2860))

    for _idx, _s_gt in enumerate(_gt):
        _mask[_idx][_s_gt] = 1

    sims_r = sims * _mask

    preds_r = np.argsort(-sims_r, axis=1)

    metrics_r = calculate_metrics(preds_r, uav_coords, sat_dataset.chunk_bboxes)

    metrics_r
    return metrics_r, preds_r


@app.cell
def _(metrics_r, preds_r, sat_dataset, uav_coords):
    # Now let's try to determine the precise location based on averaged top-3 chunk locations

    from lib.evaluation import flat_earth_dist_m


    def avg_pred():
        K = 3

        bboxes = np.array(sat_dataset.chunk_bboxes)
        lat_mins, lon_mins = bboxes[:, 0], bboxes[:, 1]
        lat_maxs, lon_maxs = bboxes[:, 2], bboxes[:, 3]
        center_lats = (lat_mins + lat_maxs) / 2
        center_lons = (lon_mins + lon_maxs) / 2

        topkpreds = preds_r[:, :K]

        avg_lats = []
        avg_lons = []

        for s_preds in topkpreds:
            topklats = center_lats[s_preds]
            avg_lats.append(topklats.mean())
            topklons = center_lons[s_preds]
            avg_lons.append(topklons.mean())

        avg_coords = np.stack([avg_lats, avg_lons], axis=1)

        u_lats = uav_coords[:, 0]
        u_lons = uav_coords[:, 1]

        dists = []
        for (lat, lon), (slat, slon) in zip(uav_coords, avg_coords):
            dists.append(flat_earth_dist_m(lat, lon, np.array([slat]), np.array([slon])))

        return dists


    avg_dist_error = np.mean(avg_pred())

    {"dist error": avg_dist_error, "diff": avg_dist_error - metrics_r["Dis@1"]}
    return


@app.cell
def _(ground_truth, preds, sat_dataset, uav_dataset):
    uav_idx = 5

    uav_img = uav_dataset[uav_idx][0]
    top1_sat_idx = preds[uav_idx][0]
    sat_img = sat_dataset[top1_sat_idx][0]

    print(top1_sat_idx in ground_truth[uav_idx])

    new_height = 512
    scale = new_height / uav_img.height
    new_w = int(uav_img.width * scale)  # Compute height based on aspect ratio

    uav_img = cv2.resize(np.array(uav_img), (new_w, new_height))


    mo.hstack([mo.image(uav_img, height=300), mo.image(sat_img, height=300)], justify="start")
    return sat_img, uav_img


@app.cell
def _(sat_img, uav_img):
    MAX_FEATURES = 500
    GOOD_MATCH_PERCENT = 0.01

    im1 = np.array(uav_img)
    im2 = np.array(sat_img)

    im1Gray = cv2.cvtColor(np.array(im1), cv2.COLOR_BGR2GRAY)
    im2Gray = cv2.cvtColor(np.array(im2), cv2.COLOR_BGR2GRAY)

    orb = cv2.ORB_create(MAX_FEATURES)
    keypoints1, descriptors1 = orb.detectAndCompute(im1Gray, None)
    keypoints2, descriptors2 = orb.detectAndCompute(im2Gray, None)

    # Match features.
    matcher = cv2.DescriptorMatcher_create(cv2.DESCRIPTOR_MATCHER_BRUTEFORCE_HAMMING)
    matches = matcher.match(descriptors1, descriptors2, None)
    matches = list(matches)
    matches.sort(key=lambda x: x.distance, reverse=False)

    numGoodMatches = int(len(matches) * GOOD_MATCH_PERCENT)
    matches = matches[:numGoodMatches]

    # Draw top matches
    imMatches = cv2.drawMatches(im1, keypoints1, im2, keypoints2, matches, None)

    # Extract location of good matches
    points1 = np.zeros((len(matches), 2), dtype=np.float32)
    points2 = np.zeros((len(matches), 2), dtype=np.float32)

    for i, match in enumerate(matches):
        points1[i, :] = keypoints1[match.queryIdx].pt
        points2[i, :] = keypoints2[match.trainIdx].pt

    # Find homography
    h, mask = cv2.findHomography(points1, points2, cv2.RANSAC)

    # Use homography
    height, width, channels = im2.shape
    im1Reg = cv2.warpPerspective(im1, h, (width, height))

    mo.vstack([mo.image(imMatches), mo.image(im1Reg)])
    return


@app.cell
def _():
    # import torch
    # from PIL import Image
    # from torchvision import transforms

    # model = LoMa(LoMaR())
    # model.to('mps');

    # t = transforms.Compose([
    #     transforms.Resize((336, 336)),
    #     transforms.ToTensor(),
    # ])

    # _im1 = t(Image.fromarray(im1)).to('mps').unsqueeze(0)
    # _im2 = t(Image.fromarray(im2)).to('mps').unsqueeze(0)


    # with torch.inference_mode():
    #     kptsA, kptsB = model.match(_im1, _im2, num_keypoints=500)

    # print(kptsA, kptsB)

    # imm = cv2.drawMatches(im1, kptsA, im2, kptsB, [], None)

    # print(_im1.squeeze(0).transpose(0,2).shape)

    # mo.vstack([
    #     mo.hstack([
    #         mo.image(_im1.cpu().squeeze(0).transpose(0,2)),
    #         mo.image(_im2.cpu().squeeze(0).transpose(0,2)),
    #     ]),
    #     mo.image(imm)
    # ])
    return


if __name__ == "__main__":
    app.run()
