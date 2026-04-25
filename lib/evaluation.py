from dataclasses import dataclass
from pathlib import Path

import numpy as np
from torch.utils.data import DataLoader

from lib.visloc import SatChunkDataset, UAVDataset


def flat_earth_dist_m(lat1: float, lon1: float, lats: np.ndarray, lons: np.ndarray) -> np.ndarray:
    """Approximate flat-earth distance in metres from one point to an array of points."""
    dlat = (lats - lat1) * 111_111
    dlon = (lons - lon1) * 111_111 * np.cos(np.radians(lat1))
    return np.sqrt(dlat**2 + dlon**2)


def build_ground_truth(
    uav_coords: np.ndarray,
    chunk_bboxes: list[tuple[float, float, float, float]],
) -> list[list[int]]:
    """For each UAV query, return indices of satellite chunks whose bbox contains the GPS point.

    Multiple overlapping chunks are sorted by distance from the UAV point to the chunk centre.
    Falls back to the single nearest chunk when the GPS point falls outside all bboxes.

    Args:
        uav_coords:   (N, 2) float array of (lat, lon) per UAV image.
        chunk_bboxes: List of (lat_min, lon_min, lat_max, lon_max) per gallery chunk.

    Returns:
        List of length N; each entry is a sorted list of matching chunk indices.
    """
    bboxes = np.array(chunk_bboxes)
    lat_mins, lon_mins = bboxes[:, 0], bboxes[:, 1]
    lat_maxs, lon_maxs = bboxes[:, 2], bboxes[:, 3]
    center_lats = (lat_mins + lat_maxs) / 2
    center_lons = (lon_mins + lon_maxs) / 2

    ground_truth = []
    for lat, lon in uav_coords:
        mask = (lat_mins <= lat) & (lat <= lat_maxs) & (lon_mins <= lon) & (lon <= lon_maxs)
        indices = np.where(mask)[0]
        if len(indices) == 0:
            dists = flat_earth_dist_m(lat, lon, center_lats, center_lons)
            indices = np.array([np.argmin(dists)])
        else:
            dists = flat_earth_dist_m(lat, lon, center_lats[indices], center_lons[indices])
            indices = indices[np.argsort(dists)]
        ground_truth.append(indices.tolist())
    return ground_truth


def distance_at_k(
    preds: np.ndarray,
    uav_coords: np.ndarray,
    chunk_bboxes: list[tuple[float, float, float, float]],
    k: int,
) -> float:
    """Mean minimum flat-earth distance (metres) from each query to its closest top-k chunk centre."""
    bboxes = np.array(chunk_bboxes)
    center_lats = (bboxes[:, 0] + bboxes[:, 2]) / 2
    center_lons = (bboxes[:, 1] + bboxes[:, 3]) / 2
    min_dists = [
        float(np.min(flat_earth_dist_m(lat, lon, center_lats[preds[i, :k]], center_lons[preds[i, :k]])))
        for i, (lat, lon) in enumerate(uav_coords)
    ]
    return float(np.mean(min_dists))


def recall_at_k(preds: np.ndarray, ground_truth: list[list[int]], k: int) -> float:
    """Fraction of queries where any ground-truth chunk appears in the top-k predictions."""
    hits = sum(any(p in gt for p in preds[i, :k]) for i, gt in enumerate(ground_truth))
    return hits / len(ground_truth)


def calculate_metrics(
    preds: np.ndarray,
    uav_coords: np.ndarray,
    chunk_bboxes: list[tuple[float, float, float, float]],
) -> dict[str, float]:
    """Compute Recall@1/5/10 (bbox-based) and Dis@1/5/10 (min distance among top-k)."""
    ground_truth = build_ground_truth(uav_coords, chunk_bboxes)
    return {
        "Recall@1": recall_at_k(preds, ground_truth, k=1),
        "Recall@5": recall_at_k(preds, ground_truth, k=5),
        "Recall@10": recall_at_k(preds, ground_truth, k=10),
        "Dis@1": distance_at_k(preds, uav_coords, chunk_bboxes, k=1),
        "Dis@5": distance_at_k(preds, uav_coords, chunk_bboxes, k=5),
        "Dis@10": distance_at_k(preds, uav_coords, chunk_bboxes, k=10),
    }


@dataclass
class VisLocEvalConfig:
    flight_id: str = "03"
    chunk_pixels: int = 512
    chunk_stride: int = 128
    scale_factor: float = 0.25
    batch_size: int = 256
    num_workers: int = 8


def build_visloc_eval(
    root: Path,
    cfg: VisLocEvalConfig,
    uav_transform=None,
    sat_transform=None,
) -> tuple[DataLoader, DataLoader, UAVDataset, SatChunkDataset]:
    """Build query and gallery dataloaders for a VisLoc flight evaluation.

    Returns:
        uav_loader:      DataLoader over UAV drone images (queries).
        gallery_loader:  DataLoader over satellite chunks (gallery).
        uav_dataset:     Underlying UAVDataset (for coordinate access).
        gallery_dataset: Underlying SatChunkDataset (for bbox access).
    """
    uav_dataset = UAVDataset(root, cfg.flight_id, transform=uav_transform)
    gallery_dataset = SatChunkDataset(
        root,
        cfg.flight_id,
        chunk_pixels=cfg.chunk_pixels,
        stride_pixels=cfg.chunk_stride,
        scale_factor=cfg.scale_factor,
        transform=sat_transform,
    )
    uav_loader = DataLoader(
        uav_dataset,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=True,
    )
    gallery_loader = DataLoader(
        gallery_dataset,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=True,
    )
    return uav_loader, gallery_loader, uav_dataset, gallery_dataset
