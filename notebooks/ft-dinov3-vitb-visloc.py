# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "marimo",
#     "python-dotenv==1.2.2",
#     "numpy==2.2.6",
#     "pandas==2.3.3",
#     "torch==2.3.1",
#     "torchvision==0.18.1",
#     "transformers==4.56.0",
#     "faiss-cpu>=1.7.4",
#     "tqdm==4.67.3",
#     "rasterio==1.4.4",
# ]
# ///

import marimo

__generated_with = "0.23.4"
app = marimo.App(width="medium")


@app.cell
def _():
    import sys
    import os
    import time
    import warnings
    from pathlib import Path

    import numpy as np
    import pandas as pd
    import torch
    import torch.nn.functional as F
    from torchvision import transforms
    from torch.utils.data import DataLoader
    from tqdm import tqdm
    from dotenv import load_dotenv

    project_root = Path(__file__).parent.parent
    sys.path.insert(0, str(project_root))

    from lib.visloc import SatChunkDataset, UAVDataset
    from lib.evaluation import calculate_metrics
    from lib.full_dinov3_ft_backbone import DINOv3Retriever, DINO_MODEL, DEFAULT_SUPERVISED_CHECKPOINT, chamfer_rerank

    load_dotenv(project_root / ".env")

    data_root = Path(os.environ["DATA_ROOT"])
    visloc_root = data_root / "visloc"
    assert visloc_root.exists(), visloc_root

    ckpt_root = Path(os.environ["CHECKPOINTS_ROOT"])
    ckpt_path = ckpt_root / DEFAULT_SUPERVISED_CHECKPOINT
    assert ckpt_path.exists(), ckpt_path

    warnings.filterwarnings("ignore", message=".*invalid escape sequence.*")

    NUM_WORKERS = 8
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return (
        DEVICE,
        DINO_MODEL,
        DINOv3Retriever,
        DataLoader,
        F,
        NUM_WORKERS,
        SatChunkDataset,
        UAVDataset,
        calculate_metrics,
        chamfer_rerank,
        ckpt_path,
        np,
        pd,
        project_root,
        time,
        torch,
        tqdm,
        transforms,
        visloc_root,
    )


@app.cell
def _(DEVICE, DINO_MODEL, DINOv3Retriever, ckpt_path):
    from transformers import AutoImageProcessor

    processor = AutoImageProcessor.from_pretrained(DINO_MODEL)
    model = DINOv3Retriever(ckpt_path=ckpt_path, model_name=DINO_MODEL)

    model = model.to(DEVICE).eval()

    embedder = model
    preprocess = processor
    return embedder, processor


@app.cell
def _(DEVICE, DataLoader, F, embedder, time, torch, tqdm):
    @torch.inference_mode()
    def extract_embeddings(
        loader: DataLoader,
        tta: bool = False,
        with_patches: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor | None, list[float], list[float], float]:
        embeddings = []
        patch_tokens = []
        all_lats = []
        all_lons = []

        t0 = time.perf_counter()
        for imgs, lats, lons in tqdm(loader, desc="Building embeddings"):
            imgs = imgs.to(DEVICE)
            if with_patches:
                embs, patches = embedder(imgs, tta=tta, with_patches=True)
                patch_tokens.append(patches.cpu())
            else:
                embs = embedder(imgs, tta=tta)
            embeddings.append(embs.cpu())
            all_lats.extend(lats)
            all_lons.extend(lons)
        elapsed = time.perf_counter() - t0

        embeddings = torch.cat(embeddings, dim=0)
        embeddings = F.normalize(embeddings, p=2, dim=1)
        patches = torch.cat(patch_tokens, dim=0) if with_patches else None

        return embeddings, patches, all_lats, all_lons, elapsed

    return (extract_embeddings,)


@app.cell
def _(
    DataLoader,
    NUM_WORKERS,
    SatChunkDataset,
    UAVDataset,
    processor,
    transforms,
    visloc_root,
):
    TTA = False
    PATCH_RERANK = True

    FLIGHT_ID = "03"

    BATCH_SIZE = 128

    CHUNK_PIXELS = 512
    CHUNK_STRIDE = CHUNK_PIXELS // 4
    MAP_SCALE_FACTOR = 0.25
    RERANK_TOPK = 50
    RERANK_ALPHA = 0.5

    mean = processor.image_mean
    std = processor.image_std

    inference_transforms = transforms.Compose([
        transforms.Resize((336, 336)),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])

    gallery_dataset = SatChunkDataset(
        visloc_root,
        FLIGHT_ID,
        chunk_pixels=CHUNK_PIXELS,
        stride_pixels=CHUNK_STRIDE,
        scale_factor=MAP_SCALE_FACTOR,
        transform=inference_transforms,
    )
    gallery_loader = DataLoader(gallery_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)

    uav_dataset = UAVDataset(visloc_root, FLIGHT_ID, transform=inference_transforms)
    uav_loader = DataLoader(uav_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)

    print(f"Gallery: {len(gallery_dataset)} satellite chunks")
    print(f"Query:   {len(uav_dataset)} UAV images")
    print(f"TTA: {TTA} | Patch re-rank: {PATCH_RERANK} | K={RERANK_TOPK} | alpha={RERANK_ALPHA}")
    return (
        CHUNK_PIXELS,
        CHUNK_STRIDE,
        FLIGHT_ID,
        MAP_SCALE_FACTOR,
        PATCH_RERANK,
        RERANK_ALPHA,
        RERANK_TOPK,
        TTA,
        gallery_dataset,
        gallery_loader,
        uav_dataset,
        uav_loader,
    )


@app.cell
def _(
    PATCH_RERANK,
    RERANK_ALPHA,
    RERANK_TOPK,
    TTA,
    chamfer_rerank,
    extract_embeddings,
    gallery_loader,
    np,
    time,
    uav_loader,
):
    gallery_embeddings, gallery_patches, _, _, t_gallery_embed = extract_embeddings(
        gallery_loader,
        tta=TTA,
        with_patches=PATCH_RERANK,
    )
    query_embeddings, query_patches, uav_lats, uav_lons, _ = extract_embeddings(
        uav_loader,
        tta=TTA,
        with_patches=PATCH_RERANK,
    )

    t0_retrieve = time.perf_counter()
    sims = (query_embeddings @ gallery_embeddings.T).cpu().numpy().astype(np.float32)
    preds = np.argsort(-sims, axis=1)

    if PATCH_RERANK:
        preds = chamfer_rerank(
            sims,
            query_patches,
            gallery_patches,
            K=RERANK_TOPK,
            alpha=RERANK_ALPHA,
        )
    t_retrieve = time.perf_counter() - t0_retrieve

    preds = preds[:, :10]
    t_gallery_s = t_gallery_embed
    retriever_type = "ip+tta+patch-rerank" if PATCH_RERANK else "ip+tta"

    print(f"Gallery build: {t_gallery_s:.1f} s")
    print(f"Retrieval compute: {t_retrieve:.2f} s")
    return (
        gallery_embeddings,
        gallery_patches,
        preds,
        query_embeddings,
        query_patches,
        retriever_type,
        t_gallery_s,
        uav_lats,
        uav_lons,
    )


@app.cell
def _(
    DEVICE,
    DataLoader,
    TTA,
    extract_embeddings,
    t_gallery_s,
    torch,
    uav_dataset,
):
    N_BENCH = 20
    _bench_loader = DataLoader(torch.utils.data.Subset(uav_dataset, range(N_BENCH)), batch_size=1, shuffle=False, num_workers=0)

    # warm up
    extract_embeddings(_bench_loader, tta=TTA, with_patches=False)

    if DEVICE.type == "cuda":
        torch.cuda.reset_peak_memory_stats(DEVICE)

    _, _, _, _, _elapsed = extract_embeddings(_bench_loader, tta=TTA, with_patches=False)

    _vram_mb = torch.cuda.max_memory_allocated(DEVICE) / 1024**2 if DEVICE.type == "cuda" else float("nan")
    _ms_per_sample = _elapsed / N_BENCH * 1000

    print(f"Inference: {_ms_per_sample:.2f} ms/sample  |  VRAM peak: {_vram_mb:.1f} MB")

    perf_metrics = {
        "gallery_build_s": t_gallery_s,
        "inference_ms_per_sample": _ms_per_sample,
        "vram_mb_peak_1sample": _vram_mb,
    }

    print(perf_metrics)
    return (perf_metrics,)


@app.cell
def _(calculate_metrics, gallery_dataset, np, preds, uav_lats, uav_lons):
    uav_coords = np.stack([uav_lats, uav_lons], axis=1)

    metrics = calculate_metrics(preds, uav_coords, gallery_dataset.chunk_bboxes)

    print(metrics)
    return (metrics,)


@app.cell
def _(
    CHUNK_PIXELS,
    CHUNK_STRIDE,
    FLIGHT_ID,
    MAP_SCALE_FACTOR,
    PATCH_RERANK,
    TTA,
    gallery_dataset,
    gallery_embeddings,
    metrics,
    perf_metrics,
    retriever_type,
    uav_dataset,
):
    entry = {
        "model": "ft-dinov3-vitb-visloc",
        "model_extra": {
            "TTA": TTA,
            "patch_rerank": PATCH_RERANK,
        },
        "dataset": "visloc",
        "dataset_extra": {
            "flight_id": FLIGHT_ID,
            "chunk_pixels": CHUNK_PIXELS,
            "chunk_stride": CHUNK_STRIDE,
            "map_scale_factor": MAP_SCALE_FACTOR,
        },
        "emb_dim": gallery_embeddings.shape[1],
        "n_gallery": len(gallery_dataset),
        "n_query": len(uav_dataset),
        "retriever_type": retriever_type,
        **metrics,
        **perf_metrics,
    }

    entry
    return (entry,)


@app.cell
def _(entry, pd, project_root):
    results_path = project_root / "out/ft-dinov3-comparison.tsv"
    results = pd.read_csv(results_path, sep="\t").to_dict(orient="records") if results_path.exists() else []

    key_cols = ["model", "model_extra", "dataset", "dataset_extra"]
    results = [r for r in results if not all(str(r.get(k)) == str(entry.get(k)) for k in key_cols)]
    results.append(entry)

    pd.DataFrame(results).sort_values("Recall@1", ascending=False).to_csv(results_path, sep="\t", index=False)
    return


@app.cell
def _(
    TTA,
    gallery_embeddings,
    gallery_patches,
    np,
    project_root,
    query_embeddings,
    query_patches,
):
    emb_dir = project_root / "embeddings"
    emb_dir.mkdir(parents=True, exist_ok=True)

    base_name = "ft-dinov3-vitb-visloc"

    if TTA:
        base_name += "-tta"

    gallery_path = emb_dir / f"{base_name}-emb-gallery.npy"
    query_path = emb_dir / f"{base_name}-emb-query.npy"
    gallery_patch_path = emb_dir / f"{base_name}-patch-emb-gallery.npy"
    query_patch_path = emb_dir / f"{base_name}-patch-emb-query.npy"

    np.save(gallery_path, gallery_embeddings.detach().cpu().numpy())
    np.save(query_path, query_embeddings.detach().cpu().numpy())

    if gallery_patches is not None:
        np.save(gallery_patch_path, gallery_patches.detach().cpu().numpy())
        print(f"Saved gallery patches:    {gallery_patch_path}")
    if query_patches is not None:
        np.save(query_patch_path, query_patches.detach().cpu().numpy())
        print(f"Saved query patches:      {query_patch_path}")

    print(f"Saved gallery embeddings: {gallery_path}")
    print(f"Saved query embeddings:   {query_path}")
    return


if __name__ == "__main__":
    app.run()
