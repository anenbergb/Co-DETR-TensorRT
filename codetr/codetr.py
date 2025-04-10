import copy
from typing import Tuple, Union, Optional, List
import warnings

import torch
import torch.nn as nn
from torch import Tensor

from mmdet.models.detectors.base import BaseDetector
from mmdet.registry import MODELS
from mmdet.structures import OptSampleList, SampleList
from mmdet.utils import InstanceList, OptConfigType, OptMultiConfig
from mmdet.evaluation import get_classes

from mmengine.config import Config
from mmengine.runner.checkpoint import _load_checkpoint, _load_checkpoint_to_model

from codetr.swin import SwinTransformer
from codetr.co_dino_head import CoDINOHead


class CoDETR(nn.Module):

    def __init__(
        self,
        backbone,
        neck=None,
        query_head=None,  # detr head
        train_cfg=[None, None],
        test_cfg=[None, None],
        **kwargs,
    ):
        super().__init__()
        # eval_module is detr

        assert backbone.pop("type") == "SwinTransformer"
        self.backbone = SwinTransformer(**backbone)
        if neck is not None:
            self.neck = MODELS.build(neck)

        assert query_head is not None
        head_idx = 0
        query_head.update(
            train_cfg=train_cfg[head_idx] if (train_cfg is not None and train_cfg[head_idx] is not None) else None
        )
        query_head.update(test_cfg=test_cfg[head_idx])
        assert query_head.pop("type") == "CoDINOHead"
        self.query_head = CoDINOHead(**query_head)
        self.query_head.init_weights()

    def forward(self, batch_inputs: Tensor, img_masks: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        """Predict results from a batch of inputs and data samples with post-
        processing.

        Args:
            batch_inputs (Tensor): Inputs, has shape (bs, 3, H, W).
            img_masks (Tensor): Masks for the input images, has shape (bs, H, W).

        Returns:
            Tuple[Tensor, Tensor, Tensor]:
                detected_boxes: (bs,num_boxes,4) where num_boxes is typicaly 300
                scores: (bs,num_boxes)
                labels: (bs,num_boxes)
        """
        # (bs,dim,H,W) -> List[ (bs,dim,H,W), ...]
        image_feats = self.backbone(batch_inputs)
        image_feats = self.neck(image_feats)
        predictions = self.query_head(image_feats, img_masks)
        return predictions


def get_dataset_meta(checkpoint):
    checkpoint_meta = checkpoint.get("meta", {})
    # save the dataset_meta in the model for convenience
    if "dataset_meta" in checkpoint_meta:
        # mmdet 3.x, all keys should be lowercase
        dataset_meta = {k.lower(): v for k, v in checkpoint_meta["dataset_meta"].items()}
    elif "CLASSES" in checkpoint_meta:
        # < mmdet 3.x
        classes = checkpoint_meta["CLASSES"]
        dataset_meta = {"classes": classes}
    else:
        warnings.warn(
            "dataset_meta or class names are not saved in the " "checkpoint's meta data, use COCO classes by default."
        )
        dataset_meta = {"classes": get_classes("coco")}
    dataset_meta["palette"] = "coco"
    return dataset_meta


def build_CoDETR(model_file: str, weights_file: Optional[str] = None, device: str = "cuda") -> CoDETR:
    """Build CoDETR model from model file and weights file."""
    cfg = Config.fromfile(model_file)
    # Delete the `pretrained` field to prevent model from loading the
    # the pretrained weights unnecessarily.
    if cfg.model.get("pretrained") is not None:
        del cfg.model.pretrained
    assert cfg.model.pop("type") == "CoDETR"
    model = CoDETR(**cfg.model)
    model.cfg = cfg
    if weights_file is None:
        model.to(device)
        model.eval()
        return model
    checkpoint = _load_checkpoint(weights_file, map_location="cpu")
    _load_checkpoint_to_model(model, checkpoint)
    model.to(device)
    model.eval()
    dataset_meta = get_dataset_meta(checkpoint)
    return model, dataset_meta
