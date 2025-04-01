# Copyright (c) OpenMMLab. All rights reserved.
import copy
from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import Linear
from mmcv.ops import batched_nms
from mmengine.structures import InstanceData
from torch import Tensor

from mmdet.models import DINOHead
from mmdet.models.layers import CdnQueryGenerator
from mmdet.models.layers.transformer import inverse_sigmoid
from mmdet.models.utils import multi_apply
from mmdet.registry import MODELS
from mmdet.structures import SampleList
from mmdet.structures.bbox import bbox_cxcywh_to_xyxy, bbox_overlaps, bbox_xyxy_to_cxcywh
from mmdet.utils import InstanceList, reduce_mean

from codetr.transformer import CoDinoTransformer


class CoDINOHead(DINOHead):

    def __init__(
        self,
        *args,
        num_query=900,
        transformer=None,
        in_channels=2048,
        max_pos_coords=300,
        dn_cfg=None,
        use_zero_padding=False,
        positional_encoding=dict(type="SinePositionalEncoding", num_feats=128, normalize=True),
        **kwargs,
    ):
        self.with_box_refine = True
        self.mixed_selection = True
        self.in_channels = in_channels
        self.max_pos_coords = max_pos_coords
        self.positional_encoding = positional_encoding
        self.num_query = num_query
        self.use_zero_padding = use_zero_padding

        if "two_stage_num_proposals" in transformer:
            assert (
                transformer["two_stage_num_proposals"] == num_query
            ), "two_stage_num_proposals must be equal to num_query for DINO"
        else:
            transformer["two_stage_num_proposals"] = num_query
        transformer["as_two_stage"] = True
        if self.mixed_selection:
            transformer["mixed_selection"] = self.mixed_selection
        self.transformer = transformer
        self.act_cfg = transformer.get("act_cfg", dict(type="ReLU", inplace=True))

        super().__init__(*args, **kwargs)

        self.activate = MODELS.build(self.act_cfg)
        self.positional_encoding = MODELS.build(self.positional_encoding)

    def _init_layers(self):
        assert self.transformer.pop("type") == "CoDinoTransformer"
        self.transformer = CoDinoTransformer(**self.transformer)
        self.embed_dims = self.transformer.embed_dims
        assert hasattr(self.positional_encoding, "num_feats")
        num_feats = self.positional_encoding.num_feats
        assert num_feats * 2 == self.embed_dims, (
            "embed_dims should" f" be exactly 2 times of num_feats. Found {self.embed_dims}" f" and {num_feats}."
        )
        """Initialize classification branch and regression branch of head."""
        fc_cls = Linear(self.embed_dims, self.cls_out_channels)
        reg_branch = []
        for _ in range(self.num_reg_fcs):
            reg_branch.append(Linear(self.embed_dims, self.embed_dims))
            reg_branch.append(nn.ReLU())
        reg_branch.append(Linear(self.embed_dims, 4))
        reg_branch = nn.Sequential(*reg_branch)

        def _get_clones(module, N):
            return nn.ModuleList([copy.deepcopy(module) for i in range(N)])

        # last reg_branch is used to generate proposal from
        # encode feature map when as_two_stage is True.
        num_pred = (
            (self.transformer.decoder.num_layers + 1) if self.as_two_stage else self.transformer.decoder.num_layers
        )

        self.cls_branches = _get_clones(fc_cls, num_pred)
        self.reg_branches = _get_clones(reg_branch, num_pred)

        self.downsample = nn.Sequential(
            nn.Conv2d(self.embed_dims, self.embed_dims, kernel_size=3, stride=2, padding=1),
            nn.GroupNorm(32, self.embed_dims),
        )

    # specializing this to only work for batch_size = 1
    # batch input shape and img_shape are the same
    def forward(
        self,
        mlvl_feats: List[Tensor],
        img_masks: Tensor,
        # rescale: bool= True
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor, List[Tensor]]:
        mlvl_masks = []
        mlvl_positional_encodings = []
        for feat in mlvl_feats:
            mlvl_masks.append(F.interpolate(img_masks[None], size=feat.shape[-2:]).to(torch.bool).squeeze(0))
            mlvl_positional_encodings.append(self.positional_encoding(mlvl_masks[-1]))

        # (1,900,256), (1,900,256)
        final_decoder_state, final_decoder_references = self.transformer(
            mlvl_feats,
            mlvl_masks,
            mlvl_positional_encodings,
            reg_branches=self.reg_branches if self.with_box_refine else None,  # noqa:E501
            cls_branches=self.cls_branches if self.as_two_stage else None,  # noqa:E501
        )

        lvl = len(self.transformer.decoder.layers) - 1
        reference = inverse_sigmoid(final_decoder_references, eps=1e-3)
        outputs_classes = self.cls_branches[lvl](final_decoder_state)
        tmp = self.reg_branches[lvl](final_decoder_state)
        if reference.shape[-1] == 4:
            tmp += reference
        else:
            assert reference.shape[-1] == 2
            tmp[..., :2] += reference
        outputs_coords = tmp.sigmoid()

        return outputs_classes, outputs_coords

    def predict_by_feat(self, cls_scores, bbox_preds, batch_img_metas, rescale=True):

        result_list = []
        for img_id in range(len(batch_img_metas)):
            cls_score = cls_scores[img_id]
            bbox_pred = bbox_preds[img_id]
            img_meta = batch_img_metas[img_id]
            results = self._predict_by_feat_single(cls_score, bbox_pred, img_meta, rescale)
            result_list.append(results)
        return result_list

    def _predict_by_feat_single(
        self, cls_score: Tensor, bbox_pred: Tensor, img_meta: dict, rescale: bool = True
    ) -> InstanceData:
        """Transform outputs from the last decoder layer into bbox predictions
        for each image.

        Args:
            cls_score (Tensor): Box score logits from the last decoder layer
                for each image. Shape [num_queries, cls_out_channels].
            bbox_pred (Tensor): Sigmoid outputs from the last decoder layer
                for each image, with coordinate format (cx, cy, w, h) and
                shape [num_queries, 4].
            img_meta (dict): Image meta info.
            rescale (bool): If True, return boxes in original image
                space. Default True.

        Returns:
            :obj:`InstanceData`: Detection results of each image
            after the post process.
            Each item usually contains following keys.

                - scores (Tensor): Classification scores, has a shape
                  (num_instance, )
                - labels (Tensor): Labels of bboxes, has a shape
                  (num_instances, ).
                - bboxes (Tensor): Has a shape (num_instances, 4),
                  the last dimension 4 arrange as (x1, y1, x2, y2).
        """
        assert len(cls_score) == len(bbox_pred)  # num_queries
        max_per_img = self.test_cfg.get("max_per_img", self.num_query)
        score_thr = self.test_cfg.get("score_thr", 0)
        with_nms = self.test_cfg.get("nms", None)

        img_shape = img_meta["img_shape"]
        # exclude background
        if self.loss_cls.use_sigmoid:
            cls_score = cls_score.sigmoid()
            scores, indexes = cls_score.view(-1).topk(max_per_img)
            det_labels = indexes % self.num_classes
            bbox_index = indexes // self.num_classes
            bbox_pred = bbox_pred[bbox_index]
        else:
            scores, det_labels = F.softmax(cls_score, dim=-1)[..., :-1].max(-1)
            scores, bbox_index = scores.topk(max_per_img)
            bbox_pred = bbox_pred[bbox_index]
            det_labels = det_labels[bbox_index]

        if score_thr > 0:
            valid_mask = scores > score_thr
            scores = scores[valid_mask]
            bbox_pred = bbox_pred[valid_mask]
            det_labels = det_labels[valid_mask]

        det_bboxes = bbox_cxcywh_to_xyxy(bbox_pred)
        det_bboxes[:, 0::2] = det_bboxes[:, 0::2] * img_shape[1]
        det_bboxes[:, 1::2] = det_bboxes[:, 1::2] * img_shape[0]
        det_bboxes[:, 0::2].clamp_(min=0, max=img_shape[1])
        det_bboxes[:, 1::2].clamp_(min=0, max=img_shape[0])
        if rescale:
            assert img_meta.get("scale_factor") is not None
            det_bboxes /= det_bboxes.new_tensor(img_meta["scale_factor"]).repeat((1, 2))

        results = InstanceData()
        results.bboxes = det_bboxes
        results.scores = scores
        results.labels = det_labels

        if with_nms and results.bboxes.numel() > 0:
            det_bboxes, keep_idxs = batched_nms(results.bboxes, results.scores, results.labels, self.test_cfg.nms)
            results = results[keep_idxs]
            results.scores = det_bboxes[:, -1]
            results = results[:max_per_img]

        return results
