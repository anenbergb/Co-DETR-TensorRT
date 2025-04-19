# Modified from MMDetection (https://github.com/open-mmlab/mmdetection)
# Original license: Apache License 2.0
import copy
from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import Linear
from mmdet.models import DINOHead
from mmdet.structures.bbox import bbox_cxcywh_to_xyxy

from codetr.positional_encoding import SinePositionalEncoding
from codetr.transformer import CoDinoTransformer


class CoDINOHead(DINOHead):
    """CoDINOHead: A detection head for CoDETR using a CoDINO transformer.

    This class implements the CoDINO head, which includes positional encoding,
    classification, and regression branches. It supports multi-scale feature
    processing and two-stage decoding.

    Args:
        num_query (int): Number of queries for the transformer. Default: 900.
        transformer (dict): Configuration for the CoDINO transformer.
        in_channels (int): Number of input channels (ignored). Default: 2048.
        max_pos_coords (int): Maximum positional coordinates (ignored). Default: 300.
        dn_cfg (dict, optional): Configuration for denoising (ignored). Default: None.
        use_zero_padding (bool, optional): Whether to use zero padding (ignored). Default: False.
        positional_encoding (dict): Configuration for positional encoding. Default:
            `dict(type="SinePositionalEncoding", num_feats=128, normalize=True)`.
        **kwargs: Additional keyword arguments for the parent class.
    """

    def __init__(
        self,
        *args,
        num_query=900,
        transformer=None,
        in_channels=2048,  # ignore this input
        max_pos_coords=300,  # ignore this input
        dn_cfg=None,  # ignore this input
        use_zero_padding=False,  # ignore this input
        positional_encoding=dict(type="SinePositionalEncoding", num_feats=128, normalize=True),
        **kwargs,
    ):
        """Initializes the CoDINOHead.

        Raises:
            AssertionError: If `two_stage_num_proposals` in the transformer configuration
                is not equal to `num_query`.
            AssertionError: If the positional encoding type is not "SinePositionalEncoding".
        """
        self.positional_encoding = positional_encoding
        self.num_query = num_query

        if "two_stage_num_proposals" in transformer:
            assert (
                transformer["two_stage_num_proposals"] == num_query
            ), "two_stage_num_proposals must be equal to num_query for DINO"
        else:
            transformer["two_stage_num_proposals"] = num_query
        transformer["as_two_stage"] = True
        self.transformer = transformer

        super().__init__(*args, **kwargs)
        assert positional_encoding.pop("type") == "SinePositionalEncoding"
        self.positional_encoding = SinePositionalEncoding(**positional_encoding)

        self.max_per_img = self.test_cfg.get("max_per_img", self.num_query)
        self.use_sigmoid = self.loss_cls.use_sigmoid

    def _init_layers(self):
        """Initialize the layers of the CoDINO head.

        - Initializes the CoDINO transformer.
        - Sets up the classification and regression branches.
        - Ensures compatibility between embedding dimensions and positional encoding features.

        Raises:
            AssertionError: If the transformer type is not "CoDinoTransformer".
            AssertionError: If `embed_dims` is not twice the number of positional encoding features.
        """
        assert self.transformer.pop("type") == "CoDinoTransformer"
        self.transformer = CoDinoTransformer(**self.transformer)
        self.embed_dims = self.transformer.embed_dims
        assert hasattr(self.positional_encoding, "num_feats")
        num_feats = self.positional_encoding.num_feats
        assert (
            num_feats * 2 == self.embed_dims
        ), "embed_dims should be exactly 2 times of num_feats. Found {self.embed_dims} and {num_feats}."

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

    def forward(
        self,
        mlvl_feats: List[torch.Tensor],
        img_masks: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass for the CoDINO head.

        Processes multi-level features and image masks to produce bounding boxes,
        scores, and labels for object detection.

        Args:
            mlvl_feats (List[torch.Tensor]): Multi-level feature maps, where each tensor
                has shape `(bs, embed_dims, h, w)`.
            img_masks (torch.Tensor): Image masks of shape `(bs, H, W)`, where:
                - `bs`: Batch size.
                - `H, W`: Height and width of the input images.

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
                - det_bboxes (torch.Tensor): Detected bounding boxes of shape `(bs, num_boxes, 4)`,
                  where `num_boxes` is typically 300. Each box is represented as `(x1, y1, x2, y2)`.
                - scores (torch.Tensor): Confidence scores of shape `(bs, num_boxes)`.
                - det_labels (torch.Tensor): Class labels of shape `(bs, num_boxes)`.

        Raises:
            AssertionError: If the shape of `final_decoder_references_unact` is not compatible
                with the regression branch.
        """
        image_height, image_width = img_masks.shape[-2:]
        mlvl_masks = []
        mlvl_positional_encodings = []

        img_masks = img_masks.unsqueeze(1)  # (bs,H,W) -> (bs,1,H,W)
        for feat in mlvl_feats:
            # (bs,1,h,w) -> (bs,h,w)
            img_mask_interp = F.interpolate(img_masks, size=feat.shape[-2:]).to(torch.bool).squeeze(1)
            pos_encoding = self.positional_encoding(img_mask_interp, dtype=feat.dtype)
            mlvl_masks.append(img_mask_interp)
            mlvl_positional_encodings.append(pos_encoding)

        # (1,900,256), (1,900,256)
        final_decoder_state, final_decoder_references_unact = self.transformer(
            mlvl_feats,
            mlvl_masks,
            mlvl_positional_encodings,
            reg_branches=self.reg_branches,
            cls_branches=self.cls_branches if self.as_two_stage else None,
        )

        lvl = len(self.transformer.decoder.layers) - 1
        outputs_classes = self.cls_branches[lvl](final_decoder_state)  # (bs,900,80)
        tmp = self.reg_branches[lvl](final_decoder_state)
        if final_decoder_references_unact.shape[-1] == 4:
            tmp += final_decoder_references_unact
        else:
            assert final_decoder_references_unact.shape[-1] == 2
            tmp[..., :2] += final_decoder_references_unact
        outputs_coords = tmp.sigmoid()  # (bs,900,4)

        batch_size = outputs_coords.shape[0]

        if self.use_sigmoid:
            cls_score = outputs_classes.sigmoid()  # (bs, num_queries, num_classes)
            scores, indexes = torch.topk(cls_score.view(batch_size, -1), self.max_per_img, dim=-1)

            det_labels = indexes % self.num_classes  # (bs, 300)
            bbox_index = indexes // self.num_classes  # (bs, 300)

            # Reshape and expand bbox_index for gathering
            expanded_indices = bbox_index.unsqueeze(-1).expand(-1, -1, 4)
            # Gather along dimension 1, the # boxes dimension
            # (bs,num_queries,4) -> (bs,300,4)
            bbox_pred = torch.gather(outputs_coords, 1, expanded_indices)

        else:
            scores, det_labels = F.softmax(outputs_classes, dim=-1)[..., :-1].max(-1)
            scores, bbox_index = torch.topk(scores, self.max_per_img, dim=-1)

            det_labels = torch.gather(det_labels, 1, bbox_index)  # (bs,num_queries) -> (bs,300)
            # Reshape and expand bbox_index for gathering
            expanded_indices = bbox_index.unsqueeze(-1).expand(-1, -1, 4)
            # Gather along dimension 1, the # boxes dimension
            # (bs,num_queries,4) -> (bs,300,4)
            bbox_pred = torch.gather(outputs_coords, 1, expanded_indices)

        det_bboxes = bbox_cxcywh_to_xyxy(bbox_pred)  # (300,4)
        det_bboxes[..., 0::2] = det_bboxes[..., 0::2] * image_width
        det_bboxes[..., 1::2] = det_bboxes[..., 1::2] * image_height
        det_bboxes[..., 0::2].clamp_(min=0, max=image_width)
        det_bboxes[..., 1::2].clamp_(min=0, max=image_height)
        return det_bboxes, scores, det_labels
