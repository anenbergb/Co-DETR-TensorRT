import copy
from typing import Tuple, Union

import torch
import torch.nn as nn
from torch import Tensor

from mmdet.models.detectors.base import BaseDetector
from mmdet.registry import MODELS
from mmdet.structures import OptSampleList, SampleList
from mmdet.utils import InstanceList, OptConfigType, OptMultiConfig


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

    def extract_feat(self, batch_inputs: Tensor) -> Tuple[Tensor]:
        """Extract features.

        Args:
            batch_inputs (Tensor): Image tensor, has shape (bs, dim, H, W).

        Returns:
            tuple[Tensor]: Tuple of feature maps from neck. Each feature map
            has shape (bs, dim, H, W).
        """
        x = self.backbone(batch_inputs)
        if self.with_neck:
            x = self.neck(x)
        return x

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

        img_feats = self.extract_feat(batch_inputs)
        results_list = self.predict_query_head(img_feats, batch_data_samples, rescale=rescale)
        return results_list

    def predict_query_head(
        self, mlvl_feats: Tuple[Tensor], batch_data_samples: SampleList, rescale: bool = True
    ) -> InstanceList:
        img_h, img_w = batch_data_samples[0].img_shape
        unpad_h, unpad_w = batch_data_samples[0].img_unpadded_shape
        # 0 within image, 1 in padded region
        img_masks = torch.ones((1, img_h, img_w), device=mlvl_feats[0].device)
        img_masks[0, :unpad_h, :unpad_w] = 0

        outs = self.query_head(mlvl_feats, img_masks)
        batch_img_metas = [data_samples.metainfo for data_samples in batch_data_samples]
        predictions = self.query_head.predict_by_feat(*outs, batch_img_metas=batch_img_metas, rescale=rescale)
        return predictions


def build_CoDETR(model_file: str, weights_file: str, device: str = "cuda") -> CoDETR:
    """Build CoDETR model from model file and weights file."""
    cfg = Config.fromfile(model_file)
    checkpoint = _load_checkpoint(weights_file, map_location="cpu")
    # Delete the `pretrained` field to prevent model from loading the
    # the pretrained weights unnecessarily.
    if cfg.model.get("pretrained") is not None:
        del cfg.model.pretrained
    assert cfg.model.pop("type") == "CoDETR"
    model = CoDETR(**cfg.model)
    model.cfg = cfg
    _load_checkpoint_to_model(model, checkpoint)
    model.to(device)
    model.eval()
    return model
