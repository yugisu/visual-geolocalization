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

__generated_with = "0.23.3"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
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
    from torch.utils.data import DataLoader
    from tqdm import tqdm
    from dotenv import load_dotenv

    project_root = Path(__file__).parent.parent
    sys.path.insert(0, str(project_root))

    from lib.visloc import SatChunkDataset, UAVDataset
    from lib.evaluation import calculate_metrics

    load_dotenv(project_root / ".env")
    data_root = Path(os.environ["DATA_ROOT"])
    visloc_root = data_root / "visloc"

    warnings.filterwarnings("ignore", message=".*invalid escape sequence.*")

    NUM_WORKERS = 8
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return (
        DEVICE,
        DataLoader,
        F,
        NUM_WORKERS,
        SatChunkDataset,
        UAVDataset,
        calculate_metrics,
        np,
        pd,
        project_root,
        time,
        torch,
        tqdm,
        visloc_root,
    )


@app.cell
def _(DEVICE):
    from transformers import AutoImageProcessor, AutoModel

    processor = AutoImageProcessor.from_pretrained("facebook/dinov3-vitl16-pretrain-sat493m")
    model = AutoModel.from_pretrained("facebook/dinov3-vitl16-pretrain-sat493m")

    model = model.to(DEVICE).eval()

    embedder = model
    preprocess = processor
    return embedder, preprocess


@app.cell
def _():
    import faiss
    import numpy as _np
    import torch as _torch
    from typing import Literal

    class FAISSRetriever:
        """Cosine similarity retriever; assumes L2-normalised embeddings."""

        def __init__(self, database_embeddings: _torch.Tensor, type: Literal["l2", "ip"] = "ip"):
            self.database_embeddings = database_embeddings
            self.type = type

            if type == "ip":
                self.index = faiss.IndexFlatIP(database_embeddings.shape[1])
            elif type == "l2":
                self.index = faiss.IndexFlatL2(database_embeddings.shape[1])

            g_np = _np.ascontiguousarray(database_embeddings.detach().cpu().numpy().astype(_np.float32))
            self.index.add(g_np)

        def search(self, query_embeddings: _torch.Tensor, k: int = 10):
            q_np = _np.ascontiguousarray(query_embeddings.detach().cpu().numpy().astype(_np.float32))
            return self.index.search(q_np, k)

    return (FAISSRetriever,)


@app.cell
def _(DEVICE, DataLoader, F, embedder, preprocess, time, torch, tqdm):
    @torch.inference_mode()
    def extract_embeddings(loader: DataLoader) -> tuple[torch.Tensor, list[float], list[float], float]:
        embeddings = []
        all_lats = []
        all_lons = []

        t0 = time.perf_counter()
        for imgs, lats, lons in tqdm(loader, desc="Building embeddings"):
            imgs = imgs.to(DEVICE)
            inputs = preprocess(imgs, return_tensors="pt").to(DEVICE)
            embs = embedder(**inputs).last_hidden_state[:, 0]
            embeddings.append(embs.cpu())
            all_lats.extend(lats)
            all_lons.extend(lons)
        elapsed = time.perf_counter() - t0

        embeddings = torch.cat(embeddings, dim=0)
        embeddings = F.normalize(embeddings, p=2, dim=1)

        return embeddings, all_lats, all_lons, elapsed

    return (extract_embeddings,)


@app.cell
def _(DataLoader, NUM_WORKERS, SatChunkDataset, UAVDataset, np, visloc_root):
    FLIGHT_ID = "03"

    BATCH_SIZE = 128

    CHUNK_PIXELS = 512
    CHUNK_STRIDE = 128
    MAP_SCALE_FACTOR = 0.25

    def inference_transforms(img):
        return np.array(img)

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
    return (
        BATCH_SIZE,
        CHUNK_PIXELS,
        CHUNK_STRIDE,
        FLIGHT_ID,
        MAP_SCALE_FACTOR,
        gallery_dataset,
        gallery_loader,
        uav_dataset,
        uav_loader,
    )


@app.cell
def _(FAISSRetriever, extract_embeddings, gallery_loader, time, uav_loader):
    gallery_embeddings, _, _, t_gallery_embed = extract_embeddings(gallery_loader)
    query_embeddings, uav_lats, uav_lons, _ = extract_embeddings(uav_loader)

    t0_idx = time.perf_counter()
    retriever = FAISSRetriever(gallery_embeddings)
    t_gallery_s = t_gallery_embed + (time.perf_counter() - t0_idx)

    print(f"Gallery build: {t_gallery_s:.1f} s")
    return gallery_embeddings, query_embeddings, retriever, t_gallery_s, uav_lats, uav_lons


@app.cell
def _(DEVICE, DataLoader, extract_embeddings, t_gallery_s, torch, uav_dataset):
    N_BENCH = 20
    _bench_loader = DataLoader(torch.utils.data.Subset(uav_dataset, range(N_BENCH)), batch_size=1, shuffle=False, num_workers=0)

    # warm up
    extract_embeddings(_bench_loader)

    if DEVICE.type == "cuda":
        torch.cuda.reset_peak_memory_stats(DEVICE)

    _, _, _, _elapsed = extract_embeddings(_bench_loader)

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
def _(
    calculate_metrics,
    gallery_dataset,
    np,
    query_embeddings,
    retriever,
    uav_lats,
    uav_lons,
):
    uav_coords = np.stack([uav_lats, uav_lons], axis=1)

    _distances, preds = retriever.search(query_embeddings, k=10)

    metrics = calculate_metrics(preds, uav_coords, gallery_dataset.chunk_bboxes)

    print(metrics)

    return (metrics,)


@app.cell
def _(
    CHUNK_PIXELS,
    CHUNK_STRIDE,
    FLIGHT_ID,
    MAP_SCALE_FACTOR,
    gallery_dataset,
    gallery_embeddings,
    metrics,
    perf_metrics,
    retriever,
    uav_dataset,
):
    entry = {
        "model": "facebook/dinov3-vitl16-pretrain-sat493m",
        "model_extra": {},
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
        "retriever_type": retriever.type,
        **metrics,
        **perf_metrics,
    }

    entry
    return (entry,)


@app.cell
def _(entry, pd, project_root):
    results_path = project_root / "out/zeroshot-comparison.tsv"
    results = pd.read_csv(results_path, sep="\t").to_dict(orient="records") if results_path.exists() else []

    results.append(entry)

    pd.DataFrame(results).sort_values("Recall@1", ascending=False).to_csv(results_path, sep="\t", index=False)
    return


if __name__ == "__main__":
    app.run()
