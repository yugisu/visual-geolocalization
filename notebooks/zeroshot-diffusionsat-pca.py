# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "marimo",
#     "python-dotenv==1.2.2",
#     "torch>=2.2.0",
#     "torchvision",
#     "numpy>=1.26.3,<2",
#     "pillow>=12.1.1",
#     "rasterio>=1.4.4",
#     "pandas>=2.3.3",
#     "tqdm>=4.67.3",
#     "matplotlib>=3.10.0",
#     "scikit-learn>=1.5.0",
#     "diffusers==0.17.0",
#     "transformers==4.40.0",
#     "huggingface-hub==0.19.3",
#     "accelerate==0.18.0",
#     "xformers==0.0.25.post1",
#     "setuptools<81",
#     "faiss-cpu>=1.7.4",
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

    from lib.visloc import SatChunkDataset, UAVDataset
    from lib.evaluation import calculate_metrics

    load_dotenv(project_root / ".env")
    data_root = Path(os.environ["DATA_ROOT"])
    visloc_root = data_root / "visloc"
    diffusionsat_ckpt = Path(os.environ["CHECKPOINTS_ROOT"]) / "finetune_sd21_256_sn-satlas-fmow_snr5_md7norm_bs64_trimmed"

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
        diffusionsat_ckpt,
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
def _(DEVICE, DTYPE, diffusionsat_ckpt, torch):
    from diffusers import AutoencoderKL, DDIMScheduler
    from transformers import CLIPTextModel, CLIPTokenizer
    from lib.diffusionsat import SatUNet

    BATCH_SIZE = 32
    IMG_SIZE = 256
    DDIM_STEPS = 10
    COLLECT = {0, 1, 2}  # low-noise inversion steps: t≈1, 101, 201
    PROMPT = "A satellite image"

    print("Loading DiffusionSat SatUNet, VAE, CLIP text encoder...")
    unet = SatUNet.from_pretrained(
        str(diffusionsat_ckpt),
        subfolder="checkpoint-150000/unet",
        num_metadata=0,
        use_metadata=False,
        low_cpu_mem_usage=False,
        torch_dtype=DTYPE,
    ).to(DEVICE)

    vae = AutoencoderKL.from_pretrained(str(diffusionsat_ckpt), subfolder="vae", torch_dtype=DTYPE).to(DEVICE)

    tokenizer = CLIPTokenizer.from_pretrained(str(diffusionsat_ckpt), subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained(str(diffusionsat_ckpt), subfolder="text_encoder", torch_dtype=DTYPE).to(DEVICE)

    scheduler = DDIMScheduler.from_pretrained(str(diffusionsat_ckpt), subfolder="scheduler")
    scheduler.set_timesteps(DDIM_STEPS)
    inv_timesteps = list(reversed(scheduler.timesteps.tolist()))
    alphas_cumprod = scheduler.alphas_cumprod
    print(f"DDIM inversion timesteps (clean→noisy): {inv_timesteps}")

    unet.eval().requires_grad_(False)
    vae.eval().requires_grad_(False)
    text_encoder.eval().requires_grad_(False)

    try:
        unet.enable_xformers_memory_efficient_attention()
        print("xformers enabled.")
    except Exception:
        pass

    _text_inputs = tokenizer(
        PROMPT,
        return_tensors="pt",
        padding="max_length",
        max_length=tokenizer.model_max_length,
        truncation=True,
    )
    with torch.inference_mode():
        prompt_embeds = text_encoder(_text_inputs.input_ids.to(DEVICE))[0]

    # Hook attn1 (self-attention) inside each Transformer2DModel in down_blocks.
    # Matches train.py: layer_idxs={'down_blocks': {'attn1': 'all'}}
    features_dict = {}

    def _make_hook(name):
        def hook(_module, _input, output):
            out = output[0] if isinstance(output, tuple) else output
            if hasattr(out, "sample"):
                out = out.sample
            out = out.detach().float()
            if out.dim() == 3:
                B, L, C = out.shape
                H = W = int(L**0.5)
                out = out.reshape(B, H, W, C).permute(0, 3, 1, 2)
            features_dict[name] = out

        return hook

    _hooks = []
    for _i, _block in enumerate(unet.down_blocks):
        if hasattr(_block, "attentions"):
            for _j, _transformer in enumerate(_block.attentions):
                for _k, _tblock in enumerate(_transformer.transformer_blocks):
                    _hooks.append(_tblock.attn1.register_forward_hook(_make_hook(f"d{_i}_{_j}_{_k}")))
    print(f"Registered {len(_hooks)} attn1 hooks on down_blocks.")
    return (
        BATCH_SIZE,
        COLLECT,
        DDIM_STEPS,
        IMG_SIZE,
        PROMPT,
        alphas_cumprod,
        features_dict,
        inv_timesteps,
        prompt_embeds,
        unet,
        vae,
    )


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
def _(
    COLLECT,
    DEVICE,
    DTYPE,
    F,
    alphas_cumprod,
    features_dict,
    inv_timesteps,
    prompt_embeds,
    time,
    torch,
    tqdm,
    unet,
    vae,
):
    def gem_pool(x: torch.Tensor, p: float = 3.0, eps: float = 1e-6) -> torch.Tensor:
        """GeM pooling: (B, C, H, W) → (B, C). Matches PoolConcatEmbedder."""
        return F.avg_pool2d(x.clamp(min=eps).pow(p), x.shape[-2:]).pow(1.0 / p).flatten(1)

    @torch.inference_mode()
    def extract_embeddings(loader):
        all_embs = []
        all_lats = []
        all_lons = []
        max_step = max(COLLECT)

        t0 = time.perf_counter()
        pe_cache = None
        for imgs, lats, lons in tqdm(loader, desc="Extracting"):
            imgs = imgs.to(DEVICE, dtype=DTYPE)
            B = imgs.shape[0]

            z = vae.encode(imgs).latent_dist.mode() * 0.18215
            if pe_cache is None or pe_cache.shape[0] != B:
                pe_cache = prompt_embeds.expand(B, -1, -1)

            collected = []
            for step_idx, t_curr in enumerate(inv_timesteps):
                t_tensor = torch.tensor([t_curr] * B, device=DEVICE, dtype=torch.long)
                features_dict.clear()
                noise_pred = unet(z, t_tensor, encoder_hidden_states=pe_cache).sample

                if step_idx in COLLECT:
                    vecs = [gem_pool(features_dict[k]) for k in sorted(features_dict)]
                    collected.append(torch.cat(vecs, dim=1).float())

                if step_idx >= max_step:
                    break

                t_next = inv_timesteps[step_idx + 1]
                a_t = alphas_cumprod[t_curr].to(z.dtype)
                a_next = alphas_cumprod[t_next].to(z.dtype)
                x0_pred = (z - (1 - a_t).sqrt() * noise_pred) / a_t.sqrt()
                z = a_next.sqrt() * x0_pred + (1 - a_next).sqrt() * noise_pred

            emb = F.normalize(torch.cat(collected, dim=1), dim=1)
            all_embs.append(emb.cpu())
            all_lats.extend(lats)
            all_lons.extend(lons)

        elapsed = time.perf_counter() - t0
        return torch.cat(all_embs, dim=0), all_lats, all_lons, elapsed

    return (extract_embeddings,)


@app.cell
def _(
    BATCH_SIZE,
    DataLoader,
    IMG_SIZE,
    NUM_WORKERS,
    SatChunkDataset,
    UAVDataset,
    transforms,
    visloc_root,
):
    FLIGHT_ID = "03"
    CHUNK_PIXELS = 512
    CHUNK_STRIDE = 128
    MAP_SCALE_FACTOR = 0.25

    inference_sat_transforms = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
    ])
    inference_uav_transforms = transforms.Compose([
        transforms.Resize(IMG_SIZE),
        transforms.CenterCrop(IMG_SIZE),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
    ])

    gallery_dataset = SatChunkDataset(
        visloc_root,
        FLIGHT_ID,
        chunk_pixels=CHUNK_PIXELS,
        stride_pixels=CHUNK_STRIDE,
        scale_factor=MAP_SCALE_FACTOR,
        transform=inference_sat_transforms,
    )
    gallery_loader = DataLoader(
        gallery_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True,
    )

    uav_dataset = UAVDataset(visloc_root, FLIGHT_ID, transform=inference_uav_transforms)
    uav_loader = DataLoader(
        uav_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True,
    )

    uav_loader_flip = DataLoader(
        UAVDataset(
            visloc_root,
            FLIGHT_ID,
            transform=transforms.Compose([
                transforms.Resize(IMG_SIZE),
                transforms.CenterCrop(IMG_SIZE),
                transforms.RandomHorizontalFlip(p=1.0),
                transforms.ToTensor(),
                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
            ]),
        ),
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True,
    )

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
        uav_loader_flip,
    )


@app.cell
def _(
    F,
    extract_embeddings,
    gallery_loader,
    time,
    uav_loader,
    uav_loader_flip,
):
    _t0 = time.perf_counter()
    gallery_embeddings_raw, _, _, _ = extract_embeddings(gallery_loader)
    t_gallery_embed = time.perf_counter() - _t0

    print("Extracting UAV embeddings (original)...")
    _query_orig, uav_lats, uav_lons, _ = extract_embeddings(uav_loader)
    print("Extracting UAV embeddings (h-flip TTA)...")
    _query_flip, _, _, _ = extract_embeddings(uav_loader_flip)
    query_embeddings_raw = F.normalize(_query_orig + _query_flip, p=2, dim=1)
    return (
        gallery_embeddings_raw,
        query_embeddings_raw,
        t_gallery_embed,
        uav_lats,
        uav_lons,
    )


@app.cell
def _(F, gallery_embeddings_raw, np, query_embeddings_raw, torch):
    from sklearn.decomposition import PCA as _PCA

    PCA_REMOVE = 16
    PCA_KEEP = 1024

    _all_np = np.concatenate(
        [
            query_embeddings_raw.numpy(),
            gallery_embeddings_raw.numpy(),
        ],
        axis=0,
    ).astype(np.float32)

    _pca = _PCA(n_components=PCA_REMOVE + PCA_KEEP, whiten=False)
    _pca.fit(_all_np)

    query_embeddings = F.normalize(
        torch.tensor(_pca.transform(query_embeddings_raw.numpy())[:, PCA_REMOVE:], dtype=torch.float32),
        p=2,
        dim=1,
    )
    gallery_embeddings = F.normalize(
        torch.tensor(_pca.transform(gallery_embeddings_raw.numpy())[:, PCA_REMOVE:], dtype=torch.float32),
        p=2,
        dim=1,
    )
    print(f"PCA whitening: removed top {PCA_REMOVE}, kept {PCA_KEEP} → {query_embeddings.shape[1]} dims")
    return PCA_KEEP, PCA_REMOVE, gallery_embeddings, query_embeddings


@app.cell
def _(FAISSRetriever, gallery_embeddings, t_gallery_embed, time):
    _t0_idx = time.perf_counter()
    retriever = FAISSRetriever(gallery_embeddings)
    t_gallery_s = t_gallery_embed + (time.perf_counter() - _t0_idx)
    print(f"Gallery build: {t_gallery_s:.1f} s")
    return retriever, t_gallery_s


@app.cell
def _(DEVICE, DataLoader, extract_embeddings, t_gallery_s, torch, uav_dataset):
    N_BENCH = 20
    _bench_loader = DataLoader(
        torch.utils.data.Subset(uav_dataset, range(N_BENCH)),
        batch_size=1,
        shuffle=False,
        num_workers=0,
    )

    extract_embeddings(_bench_loader)  # warm up

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
    BATCH_SIZE,
    CHUNK_PIXELS,
    CHUNK_STRIDE,
    COLLECT,
    DDIM_STEPS,
    FLIGHT_ID,
    IMG_SIZE,
    MAP_SCALE_FACTOR,
    PCA_KEEP,
    PCA_REMOVE,
    PROMPT,
    gallery_dataset,
    gallery_embeddings,
    metrics,
    perf_metrics,
    retriever,
    uav_dataset,
):
    entry = {
        "model": "diffusionsat-direct-extraction",
        "model_extra": {
            "img_size": IMG_SIZE,
            "ddim_steps": DDIM_STEPS,
            "collect": sorted(COLLECT),
            "prompt": PROMPT,
            "tta_hflip": True,
            "pca_remove": PCA_REMOVE,
            "pca_keep": PCA_KEEP,
            "batch_size": BATCH_SIZE,
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
    key_cols = ["model", "model_extra", "dataset", "dataset_extra"]
    results = [r for r in results if not all(str(r.get(k)) == str(entry.get(k)) for k in key_cols)]
    results.append(entry)
    pd.DataFrame(results).sort_values("Recall@1", ascending=False).to_csv(results_path, sep="\t", index=False)
    return


if __name__ == "__main__":
    app.run()
