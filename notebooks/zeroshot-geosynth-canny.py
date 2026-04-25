# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "marimo",
#     "python-dotenv==1.2.2",
#     "numpy==2.2.6",
#     "pandas==2.3.3",
#     "torch==2.3.1",
#     "torchvision==0.18.1",
#     "diffusers>=0.27.0",
#     "accelerate>=0.29.0",
#     "transformers>=4.40.0",
#     "opencv-python>=4.9.0",
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
    from torchvision import transforms
    from tqdm import tqdm
    from dotenv import load_dotenv

    project_root = Path(__file__).parent.parent
    sys.path.insert(0, str(project_root))

    diffusion_vpr_root = project_root.parent / "diffusion-vpr"
    sys.path.insert(0, str(diffusion_vpr_root))

    from lib.visloc import SatChunkDataset, UAVDataset
    from lib.evaluation import calculate_metrics

    load_dotenv(project_root / ".env")
    data_root = Path(os.environ["DATA_ROOT"])
    visloc_root = data_root / "visloc"

    warnings.filterwarnings("ignore", message=".*invalid escape sequence.*")

    NUM_WORKERS = 8
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    DTYPE = torch.bfloat16
    return (
        DEVICE,
        DTYPE,
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
        transforms,
        tqdm,
        visloc_root,
    )


@app.cell
def _(DEVICE, DTYPE):
    import torch as _torch
    from diffusers import ControlNetModel, StableDiffusionControlNetPipeline, UniPCMultistepScheduler

    from src.embedders import PoolConcatEmbedder
    from src.ldm_extractor import LDMExtractor, LDMExtractorCfg

    class GeoSynthCannyBackbone(_torch.nn.Module):
        def __init__(self, device: _torch.device, dtype: _torch.dtype):
            super().__init__()
            self.device = device
            self.dtype = dtype

            controlnet = ControlNetModel.from_pretrained("MVRL/GeoSynth-Canny")
            scheduler = UniPCMultistepScheduler.from_pretrained("sd2-community/stable-diffusion-2-1", subfolder="scheduler")

            self.pipe = StableDiffusionControlNetPipeline.from_pretrained(
                "sd2-community/stable-diffusion-2-1",
                controlnet=controlnet,
                scheduler=scheduler,
                torch_dtype=self.dtype,
                low_cpu_mem_usage=False,
            )
            self.pipe = self.pipe.to(device)
            self.vae = self.pipe.vae
            self.ldm_extractor: LDMExtractor | None = None

        def set_ldm_extractor_cfg(self, cfg: LDMExtractorCfg):
            self.ldm_extractor_cfg = cfg
            with _torch.autocast(str(self.device), dtype=self.dtype):
                self.ldm_extractor = LDMExtractor(cfg, self.pipe)

        @_torch.inference_mode()
        def forward(self, imgs: _torch.Tensor) -> dict:
            if self.ldm_extractor is None:
                raise ValueError("LDM extractor not configured. Please call set_ldm_extractor_cfg() first.")
            latents = self.vae.encode(imgs.to(dtype=self.dtype)).latent_dist.sample() * 0.18215
            feats, _ = self.ldm_extractor.forward(latents)
            return feats

    BATCH_SIZE = 128

    SAVE_TIMESTEPS = [8, 7]
    NUM_TIMESTEPS = 10
    LAYER_IDXS = {"up_blocks": {"attn1": "all"}}

    backbone = GeoSynthCannyBackbone(DEVICE, DTYPE)
    cfg = LDMExtractorCfg(
        img_size=512,
        save_timesteps=SAVE_TIMESTEPS,
        num_timesteps=NUM_TIMESTEPS,
        layer_idxs=LAYER_IDXS,
        batch_size=BATCH_SIZE,
    )
    backbone.set_ldm_extractor_cfg(cfg)

    embedder = PoolConcatEmbedder(
        feature_dims=backbone.ldm_extractor.collected_dims,
        save_timesteps=SAVE_TIMESTEPS,
    )
    return BATCH_SIZE, backbone, cfg, embedder


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
def _(DEVICE, DTYPE, DataLoader, F, backbone, embedder, time, torch, tqdm):
    @torch.inference_mode()
    def extract_embeddings(loader: DataLoader) -> tuple[torch.Tensor, list[float], list[float], float]:
        embeddings = []
        all_lats = []
        all_lons = []

        t0 = time.perf_counter()
        for imgs, lats, lons in tqdm(loader, desc="Building embeddings"):
            imgs = imgs.to(DEVICE, dtype=DTYPE)
            feats = backbone(imgs)
            embs = embedder(feats)
            embeddings.append(embs.cpu())
            all_lats.extend(lats)
            all_lons.extend(lons)
        elapsed = time.perf_counter() - t0

        embeddings = torch.cat(embeddings, dim=0)
        embeddings = F.normalize(embeddings, p=2, dim=1)

        return embeddings, all_lats, all_lons, elapsed

    return (extract_embeddings,)


@app.cell
def _(BATCH_SIZE, DataLoader, NUM_WORKERS, SatChunkDataset, UAVDataset, transforms, visloc_root):
    FLIGHT_ID = "03"

    CHUNK_PIXELS = 512
    CHUNK_STRIDE = CHUNK_PIXELS // 4
    MAP_SCALE_FACTOR = 0.25

    inference_sat_transforms = transforms.Compose(
        [
            transforms.Resize((512, 512)),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        ]
    )
    inference_uav_transforms = transforms.Compose(
        [
            transforms.Resize(512),
            transforms.CenterCrop((512, 512)),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        ]
    )

    gallery_dataset = SatChunkDataset(
        visloc_root,
        FLIGHT_ID,
        chunk_pixels=CHUNK_PIXELS,
        stride_pixels=CHUNK_STRIDE,
        scale_factor=MAP_SCALE_FACTOR,
        transform=inference_sat_transforms,
    )
    gallery_loader = DataLoader(gallery_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)

    uav_dataset = UAVDataset(visloc_root, FLIGHT_ID, transform=inference_uav_transforms)
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
    cfg,
    gallery_dataset,
    gallery_embeddings,
    metrics,
    perf_metrics,
    retriever,
    uav_dataset,
):
    entry = {
        "model": "geosynth-canny-SatDiFuser-extraction",
        "model_extra": {
            "embedder": "PoolConcatEmbedder",
            **dict(cfg.__dict__.items()),
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
