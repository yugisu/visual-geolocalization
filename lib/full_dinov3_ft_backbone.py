"""
Standalone inference wrapper for a DINOv3 st2 retrieval checkpoint.

No dependency on backbone.py or prepare.py — drop this file into any repo
alongside a torch + transformers environment.

The checkpoint must come from train-st2.py (LoRA already merged into weights).

Usage:
    model = DINOv3Retriever("checkpoints/E2-s42-ssl-model-st2-routine/epoch=12-step=1116.ckpt")

    # Encode a batch of images (preprocessed to [N, 3, 336, 336])
    embs = model(imgs)                               # [N, 512], L2-normed
    embs = model(imgs, tta=True)                     # 4-rotation averaged

    # Encode + return patch tokens for re-ranking
    embs, patches = model(imgs, tta=True, with_patches=True)

    # Full retrieval pipeline: encode both sides + optional patch re-rank
    sims = model.retrieve(query_imgs, gallery_imgs, tta=True, patch_rerank=True)
    ranked = np.argsort(-sims, axis=1)               # [N_q, N_g] ranked indices
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel

DINO_MODEL = "facebook/dinov3-vitb16-pretrain-lvd1689m"
DEFAULT_CHECKPOINT = "dinov3-ssl4eos12-visloc-smoothap-r@1=0.85-1c95460.ckpt"
DEFAULT_SUPERVISED_CHECKPOINT = "dinov3-visloc-smoothap-r@1=0.80-1c95460.ckpt"


class DINOv3Retriever(nn.Module):
    """
    Args:
        ckpt_path:      path to a .ckpt saved by train-st2.py
        model_name:     HuggingFace model id (must match training)
        embedding_dim:  projection head output size (512 in all published runs)
        device:         "cuda" / "cpu" / torch.device
    """

    def __init__(
        self,
        ckpt_path: str,
        model_name: str = DINO_MODEL,
        embedding_dim: int = 512,
        device: str | torch.device = "cuda",
    ):
        super().__init__()
        self.device = torch.device(device)

        self.backbone = AutoModel.from_pretrained(model_name, trust_remote_code=True)
        hidden = self.backbone.config.hidden_size  # 768 for ViT-B/16

        self.proj = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden, embedding_dim),
        )

        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        # logit_scale is vestigial in SmoothAP runs; drop it to allow strict=False
        state = {k: v for k, v in ckpt["state_dict"].items() if k != "logit_scale"}
        missing, _ = self.load_state_dict(state, strict=False)
        if missing:
            raise RuntimeError(f"Missing keys when loading checkpoint: {missing}")

        self.to(self.device)
        self.eval()

    # ------------------------------------------------------------------
    # Internal primitives
    # ------------------------------------------------------------------

    def _cls(self, x: torch.Tensor) -> torch.Tensor:
        h = self.backbone(pixel_values=x).last_hidden_state
        return F.normalize(self.proj(h[:, 0]), dim=-1)

    def _cls_and_patches(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.backbone(pixel_values=x).last_hidden_state
        cls = F.normalize(self.proj(h[:, 0]), dim=-1)
        patches = F.normalize(h[:, 1:], dim=-1)
        return cls, patches

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @torch.no_grad()
    def forward(
        self,
        imgs: torch.Tensor,
        tta: bool = False,
        with_patches: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """
        Encode a batch of images.

        Args:
            imgs:         [N, 3, H, W] pre-processed tensor (any device)
            tta:          average CLS embedding over 4 cardinal rotations;
                          patch tokens are always taken from the 0° rotation
            with_patches: also return patch tokens [N, P, D] (needed for re-ranking)

        Returns:
            embs [N, D]  — or  (embs [N, D], patches [N, P, D])  if with_patches=True
        """
        x = imgs.to(self.device)
        cls, patches = self._cls_and_patches(x)

        if tta:
            r1 = self._cls(torch.rot90(x, 1, [2, 3]))
            r2 = self._cls(torch.rot90(x, 2, [2, 3]))
            r3 = self._cls(torch.rot90(x, 3, [2, 3]))
            cls = F.normalize((cls + r1 + r2 + r3) / 4.0, dim=-1)

        return (cls, patches) if with_patches else cls

    @torch.no_grad()
    def retrieve(
        self,
        query_imgs: torch.Tensor,
        gallery_imgs: torch.Tensor,
        tta: bool = False,
        patch_rerank: bool = False,
        K: int = 50,
        alpha: float = 0.5,
        batch_size: int = 64,
    ) -> np.ndarray:
        """
        Full retrieval pipeline: encode → (optionally) chamfer patch re-rank.

        Args:
            query_imgs:   [N_q, 3, H, W]
            gallery_imgs: [N_g, 3, H, W]
            tta:          4-rotation TTA on both sides
            patch_rerank: re-rank top-K CLS candidates with chamfer patch similarity
            K:            number of CLS candidates to re-rank
            alpha:        CLS/patch blend weight  (final = α·CLS + (1-α)·chamfer)
            batch_size:   encoding chunk size to avoid OOM on large galleries

        Returns:
            preds [N_q, N_g] — ranked indices: np.argsort(-sims, axis=1)
        """
        q_embs, q_patches = self._encode_batched(query_imgs, tta=tta, batch_size=batch_size)
        g_embs, g_patches = self._encode_batched(gallery_imgs, tta=tta, batch_size=batch_size)

        sims = (q_embs @ g_embs.t()).cpu().numpy().astype(np.float32)
        preds = np.argsort(-sims, axis=1)

        if patch_rerank:
            preds[:, :K] = chamfer_rerank(sims, q_patches, g_patches, K=K, alpha=alpha)

        return preds

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _encode_batched(
        self,
        imgs: torch.Tensor,
        tta: bool,
        batch_size: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        embs, patches = [], []
        for i in range(0, len(imgs), batch_size):
            e, p = self(imgs[i : i + batch_size], tta=tta, with_patches=True)
            embs.append(e)
            patches.append(p)
        return torch.cat(embs), torch.cat(patches)



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
