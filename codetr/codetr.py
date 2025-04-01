import copy
from typing import Tuple, Union, Optional
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
        # rpn_head=None,  # two-stage rpn
        # roi_head=[None],  # two-stage
        # bbox_head=[None],  # one-stage
        train_cfg=[None, None],
        test_cfg=[None, None],
        # Control whether to consider positive samples
        # from the auxiliary head as additional positive queries.
        with_pos_coord=True,
        use_lsj=True,
        eval_module="detr",
        # Evaluate the Nth head.
        # eval_index=0,
        # data_preprocessor: OptConfigType = None,
        # init_cfg: OptMultiConfig = None,
        **kwargs,
    ):
        super().__init__()
        self.with_pos_coord = with_pos_coord
        self.use_lsj = use_lsj

        assert eval_module in ["detr", "one-stage", "two-stage"]
        self.eval_module = eval_module

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

        self.train_cfg = train_cfg
        self.test_cfg = test_cfg

    def forward(self, batch_inputs: Tensor, batch_data_samples: SampleList, rescale: bool = True) -> SampleList:
        """Predict results from a batch of inputs and data samples with post-
        processing.

        Args:
            batch_inputs (Tensor): Inputs, has shape (bs, dim, H, W).
            batch_data_samples (List[:obj:`DetDataSample`]): The batch
                data samples. It usually includes information such
                as `gt_instance` or `gt_panoptic_seg` or `gt_sem_seg`.
            rescale (bool): Whether to rescale the results.
                Defaults to True.

        Returns:
            list[:obj:`DetDataSample`]: Detection results of the input images.
            Each DetDataSample usually contain 'pred_instances'. And the
            `pred_instances` usually contains following keys.

            - scores (Tensor): Classification scores, has a shape
              (num_instance, )
            - labels (Tensor): Labels of bboxes, has a shape
              (num_instances, ).
            - bboxes (Tensor): Has a shape (num_instances, 4),
              the last dimension 4 arrange as (x1, y1, x2, y2).
        """
        assert self.eval_module == "detr"

        if self.use_lsj:
            for data_samples in batch_data_samples:
                img_metas = data_samples.metainfo
                input_img_h, input_img_w = img_metas["batch_input_shape"]
                img_metas["img_shape"] = [input_img_h, input_img_w]

        # (bs,dim,H,W) -> List[ (bs,dim,H,W), ...]
        image_height, image_width = batch_inputs.shape[2:]
        image_feats = self.backbone(batch_inputs)
        image_feats = self.neck(image_feats)
        results_list = self.predict_query_head(image_feats, batch_data_samples, image_height, image_width)
        return results_list

    def predict_query_head(
        self, mlvl_feats: Tuple[Tensor], batch_data_samples: SampleList, image_height: int, image_width: int
    ) -> InstanceList:
        img_h, img_w = batch_data_samples[0].img_shape
        unpad_h, unpad_w = batch_data_samples[0].img_unpadded_shape
        # 0 within image, 1 in padded region
        img_masks = torch.ones((1, img_h, img_w), device=mlvl_feats[0].device)
        img_masks[0, :unpad_h, :unpad_w] = 0

        predictions = self.query_head(mlvl_feats, img_masks)
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
