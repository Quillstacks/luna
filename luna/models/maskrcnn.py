"""Minimal torchvision Mask R-CNN wiring for lunar pit segmentation.

Two classes:
    0 = background
    1 = pit
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import torch
from PIL import Image
from pycocotools.coco import COCO
from pycocotools import mask as mask_utils
from torch.utils.data import Dataset
from torchvision.models import ResNet50_Weights
from torchvision.models.detection import maskrcnn_resnet50_fpn_v2
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor


NUM_CLASSES_DEFAULT = 2  # background + pit
ESSA_NUM_CLASSES = 3     # background + skylight + pit (Le Corre et al. 2025)


def build_maskrcnn(num_classes: int = NUM_CLASSES_DEFAULT, pretrained: bool = True) -> torch.nn.Module:
    """Torchvision Mask R-CNN with pretrained COCO backbone, swapped heads."""
    weights = "DEFAULT" if pretrained else None
    model = maskrcnn_resnet50_fpn_v2(weights=weights)

    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    hidden = 256
    model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask, hidden, num_classes)
    return model


def build_essa_model(checkpoint: Optional[str | Path] = None,
                     map_location: str = "cpu") -> torch.nn.Module:
    """Mask R-CNN wired to match Le Corre et al. (2025) ESSA exactly.

    Reproduces the construction in dlecorre387/Entrances-to-Sub-Surface-Areas
    so the published checkpoint loads without key mismatches:
        - ImageNet1K_V2 ResNet50 backbone, all 5 stages trainable
        - box_predictor swapped to 3 classes (bg + skylight + pit)
        - mask_predictor left at torchvision default (91-class output);
          only the channels for class 1/2 are ever indexed at inference

    The Zenodo checkpoint is a dict ``{"model_state_dict": ..., ...}``.
    """
    model = maskrcnn_resnet50_fpn_v2(
        weights_backbone=ResNet50_Weights.IMAGENET1K_V2,
        trainable_backbone_layers=5,
    )
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, ESSA_NUM_CLASSES)

    if checkpoint is not None:
        state = torch.load(str(checkpoint), map_location=map_location)
        sd = state.get("model_state_dict", state) if isinstance(state, dict) else state
        model.load_state_dict(sd)
    return model


class CocoMaskDataset(Dataset):
    """COCO dataset returning torchvision-detection-style targets."""

    def __init__(self, coco_json: str | Path, image_dir: str | Path, transforms=None):
        self.coco = COCO(str(coco_json))
        self.image_dir = Path(image_dir)
        self.ids = list(self.coco.imgs.keys())
        self.transforms = transforms

    def __len__(self) -> int:
        return len(self.ids)

    def __getitem__(self, idx: int):
        image_id = self.ids[idx]
        info = self.coco.loadImgs(image_id)[0]
        img = Image.open(self.image_dir / info["file_name"]).convert("RGB")

        ann_ids = self.coco.getAnnIds(imgIds=image_id)
        anns = self.coco.loadAnns(ann_ids)

        boxes, labels, masks, areas, iscrowd = [], [], [], [], []
        for a in anns:
            x, y, w, h = a["bbox"]
            if w <= 0 or h <= 0:
                continue
            boxes.append([x, y, x + w, y + h])
            labels.append(int(a["category_id"]))
            areas.append(float(a.get("area", w * h)))
            iscrowd.append(int(a.get("iscrowd", 0)))
            seg = a.get("segmentation")
            if seg is None:
                masks.append(np.zeros((info["height"], info["width"]), dtype=np.uint8))
            else:
                rle = seg if isinstance(seg, dict) else mask_utils.frPyObjects(seg, info["height"], info["width"])
                m = mask_utils.decode(rle)
                if m.ndim == 3:
                    m = m.max(axis=-1)
                masks.append(m)

        target = {
            "boxes": torch.as_tensor(boxes, dtype=torch.float32).reshape(-1, 4),
            "labels": torch.as_tensor(labels, dtype=torch.int64),
            "masks": torch.as_tensor(np.stack(masks) if masks else np.zeros((0, info["height"], info["width"]), dtype=np.uint8), dtype=torch.uint8),
            "image_id": torch.tensor([image_id]),
            "area": torch.as_tensor(areas, dtype=torch.float32),
            "iscrowd": torch.as_tensor(iscrowd, dtype=torch.int64),
        }

        if self.transforms is not None:
            img, target = self.transforms(img, target)
        else:
            img = torch.as_tensor(np.array(img).transpose(2, 0, 1), dtype=torch.float32) / 255.0

        return img, target


def collate_fn(batch):
    return tuple(zip(*batch))
