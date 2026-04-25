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
    from lib.evaluation import build_ground_truth, calculate_metrics

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
        build_ground_truth,
        calculate_metrics,
        np,
        pd,
        project_root,
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
def _(DEVICE, DataLoader, F, embedder, preprocess, torch, tqdm):
    @torch.inference_mode()
    def extract_embeddings(loader: DataLoader) -> tuple[torch.Tensor, list[float], list[float]]:
        embeddings = []
        all_lats = []
        all_lons = []

        for imgs, lats, lons in tqdm(loader, desc="Building embeddings"):
            imgs = imgs.to(DEVICE)
            inputs = preprocess(imgs, return_tensors="pt").to(DEVICE)

            embs = embedder(**inputs).pooler_output

            embeddings.append(embs.cpu())
            all_lats.extend(lats)
            all_lons.extend(lons)

        embeddings = torch.cat(embeddings, dim=0)
        embeddings = F.normalize(embeddings, p=2, dim=1)

        return embeddings, all_lats, all_lons

    return (extract_embeddings,)


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
def _(extract_embeddings, gallery_loader, uav_loader):
    gallery_embeddings, _, _ = extract_embeddings(gallery_loader)
    query_embeddings, uav_lats, uav_lons = extract_embeddings(uav_loader)
    return gallery_embeddings, query_embeddings, uav_lats, uav_lons


@app.cell
def _(
    FAISSRetriever,
    build_ground_truth,
    calculate_metrics,
    gallery_dataset,
    gallery_embeddings,
    np,
    query_embeddings,
    uav_lats,
    uav_lons,
):
    uav_coords = np.stack([uav_lats, uav_lons], axis=1)
    ground_truth = build_ground_truth(uav_coords, gallery_dataset.chunk_bboxes)

    print(f"{len(ground_truth)} UAV queries, avg {np.mean([len(gt) for gt in ground_truth]):.1f} matching chunks each")

    retriever = FAISSRetriever(gallery_embeddings)
    _distances, preds = retriever.search(query_embeddings, k=10)

    metrics = calculate_metrics(preds, ground_truth)

    print(metrics)
    return metrics, retriever


@app.cell
def _(
    CHUNK_PIXELS,
    CHUNK_STRIDE,
    FLIGHT_ID,
    MAP_SCALE_FACTOR,
    gallery_dataset,
    gallery_embeddings,
    metrics,
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
    }

    entry
    return (entry,)


@app.cell
def _(entry, pd, project_root):
    results_path = project_root / "out/1-baseline-comparison.csv"
    results = pd.read_csv(results_path).to_dict(orient="records") if results_path.exists() else []

    results.append(entry)

    pd.DataFrame(results).sort_values("Recall@1", ascending=False).to_csv(results_path, index=False)
    return


if __name__ == "__main__":
    app.run()
