"""
telescope.data
==============
PyTorch Dataset for Argoverse 2 Sensor — used as a substitute for TruckDrive
(which is not yet publicly available).

Argoverse 2 covers objects up to ~250m; TruckDrive goes to 1km.
The dataset format and class mapping are adapted to match the Telescope
training protocol as closely as possible.

Installation:
    pip install av2

Download (~1 TB full, ~30 GB mini split):
    python -m av2.datasets.sensor.download --target_dir ./data/argoverse2

Reference: Wilson et al., "Argoverse 2: Next Generation Datasets for
Self-Driving Perception and Forecasting", 2023.
"""

import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
from torch import Tensor
from torch.utils.data import Dataset
import numpy as np

__all__ = ["Argoverse2Dataset", "collate_fn", "CLASS_NAMES", "NUM_CLASSES"]

# Argoverse 2 → Telescope class mapping (background = last index)
CLASS_NAMES = [
    "REGULAR_VEHICLE",   # Car
    "LARGE_VEHICLE",     # Truck
    "SIGN",              # Sign
    "BICYCLIST",         # Bike
    "CONSTRUCTION_CONE", # Debris proxy
    "PEDESTRIAN",        # Person
    "__background__",
]
NUM_CLASSES = len(CLASS_NAMES)

_AV2_TO_IDX = {name: i for i, name in enumerate(CLASS_NAMES[:-1])}
_FRONT_CAM   = "ring_front_center"


class Argoverse2Dataset(Dataset):
    """PyTorch Dataset wrapper for Argoverse 2 Sensor split.

    Returns one sample per camera frame (front camera only by default).
    Bounding boxes are projected from 3D LiDAR annotations to 2D image
    coordinates using the pinhole camera model and normalised to [-1, 1]².

    Args:
        root_dir   : path to the downloaded Argoverse 2 sensor data
                     (e.g. './data/argoverse2/sensor/train')
        split      : 'train', 'val', or 'test'
        image_size : (H, W) to resize images to — default (1024, 1024)
        camera     : which ring camera to use — default front centre
        max_dist   : filter annotations beyond this distance in metres
    """

    def __init__(
        self,
        root_dir:   str,
        split:      str = "train",
        image_size: Tuple[int, int] = (1024, 1024),
        camera:     str = _FRONT_CAM,
        max_dist:   float = 300.0,
    ) -> None:
        try:
            from av2.datasets.sensor.av2_sensor_dataloader import AV2SensorDataLoader
            from av2.map.map_api import ArgoverseStaticMap
            self._av2 = AV2SensorDataLoader(Path(root_dir), Path(root_dir))
        except ImportError:
            raise ImportError(
                "av2 package required.\n"
                "  pip install av2\n"
                "  python -m av2.datasets.sensor.download --target_dir ./data/argoverse2"
            )

        self.root_dir   = Path(root_dir)
        self.split      = split
        self.image_size = image_size
        self.camera     = camera
        self.max_dist   = max_dist

        # Build flat index: list of (log_id, timestamp_ns) pairs
        self._index: List[Tuple[str, int]] = []
        for log_id in self._av2.get_log_ids():
            for ts in self._av2.get_ordered_log_lidar_timestamps(log_id):
                self._index.append((log_id, ts))

    def __len__(self) -> int:
        return len(self._index)

    def __getitem__(self, idx: int) -> Tuple[Tensor, Dict]:
        log_id, timestamp_ns = self._index[idx]

        # ── Load image ────────────────────────────────────────────────────────
        img_path = self._av2.get_closest_img_fpath(log_id, self.camera, timestamp_ns)
        image    = self._load_image(img_path)   # (3, H, W) in [0, 1]

        H_orig, W_orig = image.shape[-2:]
        image = F.interpolate(
            image.unsqueeze(0),
            size=self.image_size,
            mode="bilinear",
            align_corners=True,
        ).squeeze(0)   # (3, H_new, W_new)

        # ── Load 3D annotations and project to 2D ────────────────────────────
        try:
            cuboids   = self._av2.get_labels_at_lidar_timestamp(log_id, timestamp_ns)
            cam_params = self._av2.get_camera_params(log_id, self.camera)
            boxes_2d, labels = self._project_cuboids(
                cuboids, cam_params, W_orig, H_orig
            )
        except Exception:
            boxes_2d = torch.zeros(0, 4)
            labels   = torch.zeros(0, dtype=torch.long)

        # ── Normalise boxes to [-1, 1]² ───────────────────────────────────────
        if len(boxes_2d) > 0:
            H_new, W_new = self.image_size
            boxes_2d[:, 0] = boxes_2d[:, 0] / W_orig * 2 - 1   # cx
            boxes_2d[:, 1] = boxes_2d[:, 1] / H_orig * 2 - 1   # cy
            boxes_2d[:, 2] = boxes_2d[:, 2] / W_orig * 2        # w
            boxes_2d[:, 3] = boxes_2d[:, 3] / H_orig * 2        # h

        return image, {
            "boxes":    boxes_2d,
            "labels":   labels,
            "image_id": idx,
            "log_id":   log_id,
            "timestamp_ns": timestamp_ns,
        }

    def _load_image(self, img_path: Path) -> Tensor:
        from PIL import Image
        import torchvision.transforms.functional as TF
        img = Image.open(img_path).convert("RGB")
        return TF.to_tensor(img)   # (3, H, W) in [0, 1]

    def _project_cuboids(self, cuboids, cam_params, W_orig, H_orig):
        """Project 3D cuboids to 2D [cx, cy, w, h] boxes in pixel coords."""
        boxes, labels = [], []

        for cuboid in cuboids:
            # Filter by class
            cat = cuboid.category
            if cat not in _AV2_TO_IDX:
                continue

            # Filter by distance
            dist = float(np.linalg.norm(cuboid.dst_SE3_object.translation[:2]))
            if dist > self.max_dist:
                continue

            # Project 8 corners to image plane
            corners_3d = cuboid.vertices_m                    # (8, 3)
            uvz = cam_params.project_ego_to_img(corners_3d)  # (8, 3) [u, v, z]
            # Only keep corners in front of camera
            in_front = uvz[:, 2] > 0
            if in_front.sum() < 4:
                continue
            uv = uvz[in_front, :2]

            # Axis-aligned bounding box over projected corners
            x1, y1 = uv[:, 0].min(), uv[:, 1].min()
            x2, y2 = uv[:, 0].max(), uv[:, 1].max()

            # Clip to image
            x1 = max(0.0, float(x1)); y1 = max(0.0, float(y1))
            x2 = min(W_orig, float(x2)); y2 = min(H_orig, float(y2))

            if x2 <= x1 or y2 <= y1:
                continue

            cx = (x1 + x2) / 2; cy = (y1 + y2) / 2
            w  = x2 - x1;       h  = y2 - y1

            boxes.append([cx, cy, w, h])
            labels.append(_AV2_TO_IDX[cat])

        if boxes:
            return torch.tensor(boxes, dtype=torch.float32), \
                   torch.tensor(labels, dtype=torch.long)
        return torch.zeros(0, 4), torch.zeros(0, dtype=torch.long)


def collate_fn(batch: list) -> Tuple[Tensor, List[Dict]]:
    """Custom collate: images to a stacked tensor, targets kept as list.

    DETR dataloaders keep targets as a list of dicts (variable M per image).
    """
    images  = torch.stack([item[0] for item in batch], dim=0)
    targets = [item[1] for item in batch]
    return images, targets
