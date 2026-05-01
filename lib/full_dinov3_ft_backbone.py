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
            sims [N_q, N_g] — higher = better match
            ranked indices: np.argsort(-sims, axis=1)
        """
        q_embs, q_patches = self._encode_batched(query_imgs, tta=tta, batch_size=batch_size)
        g_embs, g_patches = self._encode_batched(gallery_imgs, tta=tta, batch_size=batch_size)

        sims = (q_embs @ g_embs.t()).cpu().numpy().astype(np.float32)

        if patch_rerank:
            sims = self._chamfer_rerank(sims, q_patches, g_patches, K=K, alpha=alpha)

        return sims

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

    def _chamfer_rerank(
        self,
        sims: np.ndarray,
        q_patches: torch.Tensor,
        g_patches: torch.Tensor,
        K: int,
        alpha: float,
    ) -> np.ndarray:
        """Per-query chamfer similarity between patch tokens over top-K candidates."""
        sims = sims.copy()
        for i in range(len(sims)):
            top_k = np.argsort(-sims[i])[:K]
            uav_p = q_patches[i]  # [P, D]
            sat_k = g_patches[top_k]  # [K, P, D]
            sim_mat = uav_p.unsqueeze(0) @ sat_k.transpose(-1, -2)  # [K, P, P]
            patch_sims = sim_mat.max(dim=2).values.mean(dim=1).cpu().numpy()  # [K]
            sims[i, top_k] = alpha * sims[i, top_k] + (1 - alpha) * patch_sims
        return sims
