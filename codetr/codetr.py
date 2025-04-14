import warnings
from typing import Optional, Tuple

import torch.nn as nn
from mmdet.evaluation import get_classes
from mmdet.registry import MODELS
from mmengine.config import Config
from mmengine.runner.checkpoint import _load_checkpoint, _load_checkpoint_to_model
from torch import Tensor

from codetr.co_dino_head import CoDINOHead
from codetr.swin import SwinTransformer


class CoDETR(nn.Module):
    """CoDETR: A model for object detection using a Swin Transformer backbone.

    This class implements the CoDETR model, which consists of a Swin Transformer backbone,
    an optional neck, and a CoDINO head for object detection. It supports training and
    inference configurations.

    Args:
        backbone (dict): Configuration for the Swin Transformer backbone.
        neck (dict, optional): Configuration for the neck module. Default: None.
        query_head (dict): Configuration for the CoDINO head.
        train_cfg (list[dict | None], optional): Training configuration for the model.
            Default: [None, None].
        test_cfg (list[dict | None], optional): Testing configuration for the model.
            Default: [None, None].
        **kwargs: Additional keyword arguments.
    """

    def __init__(
        self,
        backbone,
        neck=None,
        query_head=None,  # detr head
        train_cfg=[None, None],
        test_cfg=[None, None],
        **kwargs,
    ):
        """Initializes the CoDETR model.

        Raises:
            AssertionError: If the backbone type is not "SwinTransformer".
            AssertionError: If the query head type is not "CoDINOHead".
        """
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
        """Predict results from a batch of inputs and data samples

        Args:
            batch_inputs (Tensor): Input images of shape `(bs, 3, H, W)`, where:
                - `bs`: Batch size.
                - `3`: Number of input channels (RGB).
                - `H, W`: Height and width of the input images.
            img_masks (Tensor): Masks for the input images of shape `(bs, H, W)`, where:
                - `bs`: Batch size.
                - `H, W`: Height and width of the input images.

        Returns:
            Tuple[Tensor, Tensor, Tensor]:
                - detected_boxes (Tensor): Bounding boxes of shape `(bs, num_boxes, 4)`, where:
                    - `num_boxes`: Number of detected boxes (typically 300).
                    - `4`: Coordinates of each box `(x1, y1, x2, y2)`.
                - scores (Tensor): Confidence scores of shape `(bs, num_boxes)`.
                - labels (Tensor): Class labels of shape `(bs, num_boxes)`.
        """
        # (bs,dim,H,W) -> List[ (bs,dim,H,W), ...]
        image_feats = self.backbone(batch_inputs)
        image_feats = self.neck(image_feats)
        predictions = self.query_head(image_feats, img_masks)
        return predictions


def get_dataset_meta(checkpoint):
    """Extract dataset metadata from a checkpoint.

    This function retrieves the dataset metadata (e.g., class names and palette)
    from the checkpoint's metadata. If the metadata is not available, it defaults
    to COCO classes.

    Args:
        checkpoint (dict): The checkpoint containing metadata.

    Returns:
        dict: A dictionary containing dataset metadata, including:
            - `classes` (list[str]): List of class names.
            - `palette` (str): Color palette for visualization (default: "coco").

    Raises:
        UserWarning: If the dataset metadata or class names are not found in the checkpoint.
    """
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


def build_CoDETR(model_file: str, weights_file: Optional[str] = None, device: str = "cuda") -> Tuple[CoDETR, dict]:
    """Build the CoDETR model from a configuration file and optional weights file.

    This function constructs the CoDETR model using the provided configuration file
    and optionally loads pretrained weights. The model is moved to the specified device
    and set to evaluation mode.

    Args:
        model_file (str): Path to the model configuration file.
        weights_file (str, optional): Path to the pretrained weights file. Default: None.
        device (str): Device to load the model onto (e.g., "cuda" or "cpu"). Default: "cuda".

    Returns:
        Tuple[CoDETR, dict]: A tuple containing:
            - The CoDETR model instance.
            - The dataset metadata (e.g., class names and palette).

    Raises:
        AssertionError: If the model type in the configuration is not "CoDETR".

    Notes:
        - If `weights_file` is not provided, the model is returned without loading weights.
        - The `pretrained` field in the configuration is removed to prevent unnecessary loading.
    """
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
