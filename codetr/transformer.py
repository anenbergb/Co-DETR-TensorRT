# Modified from MMDetection (https://github.com/open-mmlab/mmdetection)
# Original license: Apache License 2.0
import math

import torch
import torch.nn as nn
from mmcv.cnn import build_norm_layer
from mmengine.model import BaseModule
from mmengine.model.weight_init import xavier_init
from torch.nn.init import normal_

from codetr.multi_scale_deformable_attention import MultiScaleDeformableAttention
from codetr.transformer_mmcv import BaseTransformerLayer


class DetrTransformerEncoder(BaseModule):
    """Transformer Encoder for DETR.

    Args:
        post_norm_cfg (dict, optional): Configuration for the last normalization layer.
            Default: `dict(type="LN")`. Only used when `self.pre_norm` is `True`.
        with_cp (int, optional): Placeholder argument, currently ignored. Default: -1.
        transformerlayers (dict): Configuration for the transformer layers.
            Must include the key `"type": "BaseTransformerLayer"`.
        num_layers (int): Number of transformer layers.
        init_cfg (dict, optional): Initialization configuration. Default: None.
    """

    def __init__(
        self,
        post_norm_cfg=dict(type="LN"),
        with_cp=-1,  # ignore_input
        transformerlayers=None,
        num_layers=None,
        init_cfg=None,
    ):
        super().__init__(init_cfg)
        assert isinstance(transformerlayers, dict)
        assert transformerlayers.pop("type") == "BaseTransformerLayer"
        self.num_layers = num_layers
        self.layers = nn.ModuleList()
        for i in range(num_layers):
            self.layers.append(BaseTransformerLayer(**transformerlayers))
        self.embed_dims = self.layers[0].embed_dims
        self.pre_norm = self.layers[0].pre_norm

        if post_norm_cfg is not None:
            self.post_norm = build_norm_layer(post_norm_cfg, self.embed_dims)[1] if self.pre_norm else None
        else:
            assert not self.pre_norm, f"Use prenorm in " f"{self.__class__.__name__}," f"Please specify post_norm_cfg"
            self.post_norm = None

    def forward(
        self,
        query,
        key,
        value,
        query_pos=None,
        key_pos=None,
        attn_masks=None,
        query_key_padding_mask=None,
        key_padding_mask=None,
        **kwargs,
    ):
        """Forward pass for the Transformer Encoder.

        Args:
            query (Tensor): Input query tensor of shape `(num_queries, bs, embed_dims)`.
            key (Tensor): Key tensor of shape `(num_keys, bs, embed_dims)`.
            value (Tensor): Value tensor of shape `(num_keys, bs, embed_dims)`.
            query_pos (Tensor, optional): Positional encoding for the query. Default: None.
            key_pos (Tensor, optional): Positional encoding for the key. Default: None.
            attn_masks (list[Tensor], optional): Attention masks for each layer. Default: None.
            query_key_padding_mask (Tensor, optional): Padding mask for the query of shape `(bs, num_queries)`.
                Default: None.
            key_padding_mask (Tensor, optional): Padding mask for the key of shape `(bs, num_keys)`. Default: None.

        Returns:
            Tensor: Output tensor of shape `(num_queries, bs, embed_dims)`.
        """
        for layer in self.layers:
            query = layer(
                query,
                key,
                value,
                query_pos=query_pos,
                key_pos=key_pos,
                attn_masks=attn_masks,
                query_key_padding_mask=query_key_padding_mask,
                key_padding_mask=key_padding_mask,
                **kwargs,
            )
        return query


def build_MLP(input_dim, hidden_dim, output_dim, num_layers):
    """Builds a Multi-Layer Perceptron (MLP).

    Args:
        input_dim (int): Input dimension of the MLP.
        hidden_dim (int): Hidden layer dimension.
        output_dim (int): Output dimension of the MLP.
        num_layers (int): Number of layers in the MLP. Must be greater than 1.

    Returns:
        nn.Sequential: A sequential container of MLP layers.
    """
    assert num_layers > 1, f"num_layers should be greater than 1 but got {num_layers}"
    h = [hidden_dim] * (num_layers - 1)
    layers = list()
    for n, k in zip([input_dim] + h[:-1], h):
        layers.extend((nn.Linear(n, k), nn.ReLU()))
    # Note that the relu func of MLP in original DETR repo is set
    # 'inplace=False', however the ReLU cfg of FFN in mmdet is set
    # 'inplace=True' by default.
    layers.append(nn.Linear(hidden_dim, output_dim))
    return nn.Sequential(*layers)


class DinoTransformerDecoder(BaseModule):
    """Transformer Decoder for DINO.

    Args:
        return_intermediate (bool, optional): Whether to return intermediate results.
            Default: False.
        transformerlayers (dict): Configuration for the transformer layers.
            Must include the key `"type": "DetrTransformerDecoderLayer"`.
        num_layers (int): Number of transformer layers.
        init_cfg (dict, optional): Initialization configuration. Default: None.
    """

    def __init__(
        self,
        return_intermediate=False,
        transformerlayers=None,
        num_layers=None,
        init_cfg=None,
    ):
        super().__init__(init_cfg)
        assert isinstance(transformerlayers, dict)
        assert transformerlayers.pop("type") == "DetrTransformerDecoderLayer"
        self.num_layers = num_layers
        self.layers = nn.ModuleList()
        for i in range(num_layers):
            self.layers.append(DetrTransformerDecoderLayer(**transformerlayers))
        self.embed_dims = self.layers[0].embed_dims
        self.pre_norm = self.layers[0].pre_norm
        self.return_intermediate = return_intermediate

        self._init_layers()

    def _init_layers(self):
        self.ref_point_head = build_MLP(self.embed_dims * 2, self.embed_dims, self.embed_dims, 2)
        self.norm = nn.LayerNorm(self.embed_dims)

    @staticmethod
    def gen_sineembed_for_position(pos_tensor, pos_feat):
        """Generates sine embeddings for positional encoding.

        Args:
            pos_tensor (Tensor): Input position tensor of shape `(n_query, bs, 2 or 4)`.
            pos_feat (int): Number of positional features.

        Returns:
            Tensor: Sine embeddings of shape `(n_query, bs, pos_feat)`.
        """
        scale = 2 * math.pi
        dim_t = torch.arange(pos_feat, dtype=pos_tensor.dtype, device=pos_tensor.device)
        dim_t = 10000 ** (2 * (dim_t // 2) / pos_feat)
        x_embed = pos_tensor[:, :, 0] * scale
        y_embed = pos_tensor[:, :, 1] * scale
        pos_x = x_embed[:, :, None] / dim_t
        pos_y = y_embed[:, :, None] / dim_t
        pos_x = torch.stack((pos_x[:, :, 0::2].sin(), pos_x[:, :, 1::2].cos()), dim=3).flatten(2)
        pos_y = torch.stack((pos_y[:, :, 0::2].sin(), pos_y[:, :, 1::2].cos()), dim=3).flatten(2)
        if pos_tensor.size(-1) == 2:
            pos = torch.cat((pos_y, pos_x), dim=2)
        elif pos_tensor.size(-1) == 4:
            w_embed = pos_tensor[:, :, 2] * scale
            pos_w = w_embed[:, :, None] / dim_t
            pos_w = torch.stack((pos_w[:, :, 0::2].sin(), pos_w[:, :, 1::2].cos()), dim=3).flatten(2)

            h_embed = pos_tensor[:, :, 3] * scale
            pos_h = h_embed[:, :, None] / dim_t
            pos_h = torch.stack((pos_h[:, :, 0::2].sin(), pos_h[:, :, 1::2].cos()), dim=3).flatten(2)

            pos = torch.cat((pos_y, pos_x, pos_w, pos_h), dim=2)
        else:
            raise ValueError("Unknown pos_tensor shape(-1):{}".format(pos_tensor.size(-1)))
        return pos

    def forward(self, query, *args, reference_points=None, valid_ratios=None, reg_branches=None, **kwargs):
        """Forward pass for the Transformer Decoder.

        Args:
            query (Tensor): Input query tensor of shape `(num_queries, bs, embed_dims)`.
            reference_points (Tensor, optional): Reference points for the decoder.
                Default: None.
            valid_ratios (Tensor, optional): Ratios of valid points on the feature map.
                Default: None.
            reg_branches (list[nn.Module], optional): Regression branches for bounding box refinement.
                Default: None.

        Returns:
            Tuple[Tensor, Tensor]: Final state and reference points after decoding.
        """
        output = query
        for lid, layer in enumerate(self.layers):
            if reference_points.shape[-1] == 4:
                reference_points_input = (
                    reference_points[:, :, None].sigmoid() * torch.cat([valid_ratios, valid_ratios], -1)[:, None]
                )
            else:
                assert reference_points.shape[-1] == 2
                reference_points_input = reference_points[:, :, None].sigmoid() * valid_ratios[:, None]

            query_sine_embed = self.gen_sineembed_for_position(reference_points_input[:, :, 0, :], self.embed_dims // 2)
            query_pos = self.ref_point_head(query_sine_embed)  # (1,900,256)

            query_pos = query_pos.permute(1, 0, 2)  # (900, 1, 256)
            output = layer(output, *args, query_pos=query_pos, reference_points=reference_points_input, **kwargs)

            if reg_branches is not None:
                tmp = reg_branches[lid](output.permute(1, 0, 2))
                assert reference_points.shape[-1] == 4
                reference_points = tmp + reference_points

        output = output.permute(1, 0, 2)  # (900,1,256) -> (1,900,256)
        output = self.norm(output)
        return output, reference_points  # (1,900,4)


class DetrTransformerDecoderLayer(BaseTransformerLayer):
    """Implements decoder layer in DETR transformer.

    Args:
        attn_cfgs (list[`mmcv.ConfigDict`] | list[dict] | dict )):
            Configs for self_attention or cross_attention, the order
            should be consistent with it in `operation_order`. If it is
            a dict, it would be expand to the number of attention in
            `operation_order`.
        feedforward_channels (int): The hidden dimension for FFNs.
        ffn_dropout (float): Probability of an element to be zeroed
            in ffn. Default 0.0.
        operation_order (tuple[str]): The execution order of operation
            in transformer. Such as ('self_attn', 'norm', 'ffn', 'norm').
            Default：None
        act_cfg (dict): The activation config for FFNs. Default: `LN`
        norm_cfg (dict): Config dict for normalization layer.
            Default: `LN`.
        ffn_num_fcs (int): The number of fully-connected layers in FFNs.
            Default：2.
    """

    def __init__(
        self,
        attn_cfgs,
        feedforward_channels,
        ffn_dropout=0.0,
        operation_order=None,
        act_cfg=dict(type="ReLU", inplace=True),
        norm_cfg=dict(type="LN"),
        ffn_num_fcs=2,
        **kwargs,
    ):
        super(DetrTransformerDecoderLayer, self).__init__(
            attn_cfgs=attn_cfgs,
            feedforward_channels=feedforward_channels,
            ffn_dropout=ffn_dropout,
            operation_order=operation_order,
            act_cfg=act_cfg,
            norm_cfg=norm_cfg,
            ffn_num_fcs=ffn_num_fcs,
            **kwargs,
        )
        assert len(operation_order) == 6
        assert set(operation_order) == set(["self_attn", "norm", "cross_attn", "ffn"])


def get_reference_points(mlvl_feats, valid_ratios, device):
    """Generates reference points for the decoder.

    Args:
        mlvl_feats (list[Tensor]): Feature maps from different levels, each of shape `(bs, embed_dims, h, w)`.
        valid_ratios (Tensor): Ratios of valid points on the feature map, of shape `(bs, num_levels, 2)`.
        device (torch.device): Device where the reference points should be created.

    Returns:
        Tensor: Reference points of shape `(bs, num_keys, num_levels, 2)`.
    """
    reference_points_list = []
    for lvl, feat in enumerate(mlvl_feats):
        bs, _, H, W = feat.shape
        ref_y1, ref_x1 = torch.meshgrid(
            torch.linspace(0.5, H - 0.5, H, dtype=feat.dtype, device=device),
            torch.linspace(0.5, W - 0.5, W, dtype=feat.dtype, device=device),
            indexing="ij",
        )
        ref_y = ref_y1.reshape(1, -1) / (valid_ratios[:, lvl, 1].reshape(bs, 1) * H)
        ref_x = ref_x1.reshape(1, -1) / (valid_ratios[:, lvl, 0].reshape(bs, 1) * W)
        ref = torch.stack((ref_x, ref_y), -1)
        reference_points_list.append(ref)
    # (bs,num_keys,2)
    reference_points = torch.cat(reference_points_list, 1)
    return reference_points


def make_encoder_output_proposals(reference_points, level_counts):
    """Generates proposals from encoded memory.

    Args:
        reference_points (Tensor): Reference points of shape `(bs, num_keys, 2)`.
        level_counts (Tensor): Number of points on each feature map level, of shape `(num_levels,)`.

    Returns:
        Tensor: Normalized proposals of shape `(bs, num_keys, 4)`.
    """
    batch_size, num_keys = reference_points.shape[:2]
    # Create values tensor [0, 1, 2, 3, 4]
    lvl_indices = torch.arange(level_counts.shape[0], dtype=reference_points.dtype, device=reference_points.device)
    # Repeat each value based on its count
    lvl_repeated = torch.repeat_interleave(lvl_indices, level_counts)  # (num_keys,)
    width = 0.05 * (2.0**lvl_repeated)  # (num_keys,)
    width = width.expand(batch_size, num_keys).view(batch_size, num_keys, 1)  # (1,num_keys,1)
    # reference points (bs,num_keys,2)
    output_proposals = torch.cat([reference_points, width, width], dim=-1)  # (bs,num_keys,4)
    output_proposals = torch.log(output_proposals / (1 - output_proposals))
    return output_proposals


def make_encoder_output_proposals_export(reference_points, mlvl_masks):
    batch_size, num_keys = reference_points.shape[:2]
    lvl_repeated = get_lvl_repeated(mlvl_masks, dtype=reference_points.dtype)  # (num_keys,)
    width = 0.05 * (2.0**lvl_repeated)  # (num_keys,)
    width = width.expand(batch_size, num_keys).view(batch_size, num_keys, 1)  # (1,num_keys,1)
    # reference points (bs,num_keys,2)
    output_proposals = torch.cat([reference_points, width, width], dim=-1)  # (bs,num_keys,4)
    output_proposals = torch.log(output_proposals / (1 - output_proposals))
    return output_proposals


def get_lvl_repeated(mlvl_masks, dtype=torch.float32):
    lvl_repeated = []
    for lvl, mask in enumerate(mlvl_masks):
        H, W = mask.shape[-2:]
        lvl_repeated.append(lvl * torch.ones(H * W, dtype=dtype, device=mask.device))
    lvl_repeated = torch.cat(lvl_repeated, 0)
    return lvl_repeated


def apply_mask_to_proposal_and_memory(output_proposals, memory, memory_padding_mask):
    """Applies masking to proposals and memory.

    Args:
        output_proposals (Tensor): Normalized proposals of shape `(bs, num_keys, 4)`.
        memory (Tensor): Encoder output of shape `(bs, num_keys, embed_dim)`.
        memory_padding_mask (Tensor): Padding mask for memory of shape `(bs, num_keys)`.

    Returns:
        Tuple[Tensor, Tensor]: Masked proposals and memory.
    """
    # log(0.1 / (1 - 0.1)) = -4.6
    # log(0.99 / (1 - 0.99)) = 4.6

    valid_min = -4.6
    valid_max = 4.6

    # Instead of masking, compute a multiplicative mask
    # 1.0 for valid, 0.0 for invalid
    in_bounds = ((output_proposals > valid_min) & (output_proposals < valid_max)).to(output_proposals.dtype)
    output_proposals_valid = torch.prod(in_bounds, dim=-1, keepdim=True)

    # memory_padding_mask: (bs, num_keys) -> (bs, num_keys, 1)
    mask = (~memory_padding_mask).to(output_proposals.dtype).unsqueeze(-1)

    # Combine both masks
    total_mask = output_proposals_valid * mask

    output_proposals = output_proposals * total_mask + (1.0 - total_mask) * torch.finfo(output_proposals.dtype).max
    output_memory = memory * total_mask + (1.0 - total_mask) * 0.0
    return output_proposals, output_memory


def get_valid_ratio(mask, dtype=torch.float32):
    """Calculates valid ratios for feature maps at all levels.

    Args:
        mask (Tensor): Mask tensor of shape `(bs, h, w)`.
        dtype (torch.dtype, optional): Data type for the output. Default: torch.float32.

    Returns:
        Tensor: Valid ratios of shape `(bs, 2)`.
    """
    _, H, W = mask.shape
    valid_H = torch.sum(~mask[:, :, 0], 1).to(dtype)
    valid_W = torch.sum(~mask[:, 0, :], 1).to(dtype)
    valid_ratio_h = valid_H / H
    valid_ratio_w = valid_W / W
    valid_ratio = torch.stack([valid_ratio_w, valid_ratio_h], -1)
    return valid_ratio


class CoDinoTransformer(BaseModule):
    """Transformer for Co-DINO.

    Args:
        with_pos_coord (bool, optional): Whether to use positional coordinates. Default: True.
        with_coord_feat (bool, optional): Whether to use coordinate features. Default: True.
        num_co_heads (int, optional): Number of co-heads. Default: 1.
        as_two_stage (bool, optional): Whether to use two-stage decoding. Default: False.
        num_feature_levels (int, optional): Number of feature levels. Default: 4.
        two_stage_num_proposals (int, optional): Number of proposals for two-stage decoding. Default: 300.
        encoder (dict): Configuration for the encoder.
        decoder (dict): Configuration for the decoder.
        init_cfg (dict, optional): Initialization configuration. Default: None.
    """

    def __init__(
        self,
        with_pos_coord=True,
        with_coord_feat=True,
        num_co_heads=1,
        as_two_stage=False,
        num_feature_levels=4,
        two_stage_num_proposals=300,
        encoder=None,
        decoder=None,
        init_cfg=None,
    ):
        super(CoDinoTransformer, self).__init__(init_cfg=init_cfg)
        assert encoder.pop("type") == "DetrTransformerEncoder"
        self.encoder = DetrTransformerEncoder(**encoder)
        assert decoder.pop("type") == "DinoTransformerDecoder"
        self.decoder = DinoTransformerDecoder(**decoder)
        self.embed_dims = self.encoder.embed_dims

        self.with_pos_coord = with_pos_coord
        self.with_coord_feat = with_coord_feat
        self.num_co_heads = num_co_heads

        self.as_two_stage = as_two_stage
        self.num_feature_levels = num_feature_levels
        self.two_stage_num_proposals = two_stage_num_proposals

        self.init_layers()

    def init_layers(self):
        """Initialize layers of the DinoTransformer."""
        self.level_embeds = nn.Parameter(torch.Tensor(self.num_feature_levels, self.embed_dims))
        self.enc_output = nn.Linear(self.embed_dims, self.embed_dims)
        self.enc_output_norm = nn.LayerNorm(self.embed_dims)
        self.query_embed = nn.Embedding(self.two_stage_num_proposals, self.embed_dims)

        if self.with_pos_coord:
            if self.num_co_heads > 0:
                self.aux_pos_trans = nn.ModuleList()
                self.aux_pos_trans_norm = nn.ModuleList()
                self.pos_feats_trans = nn.ModuleList()
                self.pos_feats_norm = nn.ModuleList()
                for i in range(self.num_co_heads):
                    self.aux_pos_trans.append(nn.Linear(self.embed_dims * 2, self.embed_dims))
                    self.aux_pos_trans_norm.append(nn.LayerNorm(self.embed_dims))
                    if self.with_coord_feat:
                        self.pos_feats_trans.append(nn.Linear(self.embed_dims, self.embed_dims))
                        self.pos_feats_norm.append(nn.LayerNorm(self.embed_dims))

    def init_weights(self):
        """Initializes the weights of the transformer."""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        for m in self.modules():
            if isinstance(m, MultiScaleDeformableAttention):
                m.init_weights()
        if not self.as_two_stage:
            xavier_init(self.reference_points, distribution="uniform", bias=0.0)
        normal_(self.level_embeds)
        nn.init.normal_(self.query_embed.weight.data)

    def forward(
        self,
        mlvl_feats,
        mlvl_masks,
        mlvl_pos_embeds,
        reg_branches=None,
        cls_branches=None,
        **kwargs,
    ):
        """Forward pass for the Co-DINO Transformer.

        Args:
            mlvl_feats (list[Tensor]): Multi-level feature maps, each of shape `(bs, c, h, w)`.
            mlvl_masks (list[Tensor]): Multi-level masks, each of shape `(bs, h, w)`.
            mlvl_pos_embeds (list[Tensor]): Multi-level positional embeddings, each of shape `(bs, c, h, w)`.
            reg_branches (list[nn.Module], optional): Regression branches for bounding box refinement. Default: None.
            cls_branches (list[nn.Module], optional): Classification branches for object detection. Default: None.

        Returns:
            Tuple[Tensor, Tensor]: Final state and reference points after decoding.
        """
        assert self.as_two_stage, "as_two_stage must be True for DINO"

        feat_flatten = []
        mask_flatten = []
        lvl_pos_embed_flatten = []
        spatial_shapes = []
        # feat: (B,C,H,W), mask: (B,H,W), pos_embed: (B,C,H,W)
        for lvl, (feat, mask, pos_embed) in enumerate(zip(mlvl_feats, mlvl_masks, mlvl_pos_embeds)):
            bs, c, h, w = feat.shape
            spatial_shape = (h, w)
            spatial_shapes.append(spatial_shape)
            feat = feat.flatten(2).transpose(1, 2)  # (B,C,H,W) -> (B,H*W,C)
            mask = mask.flatten(1)  # (B,H,W) -> (B,H*W)
            pos_embed = pos_embed.flatten(2).transpose(1, 2)  # (B,C,H,W) -> (B,H*W,C)
            lvl_pos_embed = pos_embed + self.level_embeds[lvl].view(1, 1, -1)
            lvl_pos_embed_flatten.append(lvl_pos_embed)
            feat_flatten.append(feat)
            mask_flatten.append(mask)
        feat_flatten = torch.cat(feat_flatten, 1)
        mask_flatten = torch.cat(mask_flatten, 1)
        lvl_pos_embed_flatten = torch.cat(lvl_pos_embed_flatten, 1)
        spatial_shapes = torch.as_tensor(spatial_shapes, dtype=torch.long, device=feat_flatten.device)
        level_counts = spatial_shapes.prod(1)
        level_start_index = torch.cat((spatial_shapes.new_zeros((1,)), level_counts.cumsum(0)[:-1]))
        valid_ratios = torch.stack([get_valid_ratio(m, dtype=feat_flatten.dtype) for m in mlvl_masks], 1)

        reference_points = get_reference_points(mlvl_feats, valid_ratios, device=feat_flatten.device)  # (1,12276,5,2)
        # (bs,num_keys,2) * (bs,num_levels,2)
        # (bs,num_keys,1,2) * (bs,1,num_levels,2) -> (bs,num_keys,num_levels,2)
        reference_points_by_level = reference_points[:, :, None] * valid_ratios[:, None]

        feat_flatten = feat_flatten.permute(1, 0, 2)  # (H*W, bs, embed_dims)
        lvl_pos_embed_flatten = lvl_pos_embed_flatten.permute(1, 0, 2)  # (H*W, bs, embed_dims)
        memory = self.encoder(
            query=feat_flatten,
            key=None,
            value=None,
            query_pos=lvl_pos_embed_flatten,
            query_key_padding_mask=mask_flatten,
            spatial_shapes=spatial_shapes,
            reference_points=reference_points_by_level,
            level_start_index=level_start_index,
            valid_ratios=valid_ratios,
            **kwargs,
        )
        memory = memory.permute(1, 0, 2)
        bs, _, c = memory.shape

        output_proposals = make_encoder_output_proposals_export(reference_points, mlvl_masks)  # (bs,spatial_len,4)
        output_proposals, output_memory = apply_mask_to_proposal_and_memory(output_proposals, memory, mask_flatten)
        output_memory = self.enc_output_norm(self.enc_output(output_memory))

        # decoder num_layers = 6, but # cls_branches = 7, so cls_branches[6] is the final classification branch
        # (bs,spatial_len,80) for 80 classes
        enc_outputs_class = cls_branches[self.decoder.num_layers](output_memory)
        # (bs,spatial_len,4)
        enc_outputs_coord_unact = reg_branches[self.decoder.num_layers](output_memory) + output_proposals
        topk = self.two_stage_num_proposals
        # # NOTE In DeformDETR, enc_outputs_class[..., 0] is used for topk
        topk_indices = torch.topk(enc_outputs_class.max(-1)[0], topk, dim=1)[1]  # (bs, topk)
        topk_coords_unact = torch.gather(enc_outputs_coord_unact, 1, topk_indices.unsqueeze(-1).repeat(1, 1, 4))
        query = self.query_embed.weight[:, None, :].repeat(1, bs, 1).transpose(0, 1)

        # decoder
        query = query.permute(1, 0, 2)
        memory = memory.permute(1, 0, 2)

        # (1,900,256), (1,900,256)
        final_state, final_references_unact = self.decoder(
            query=query,
            key=None,
            value=memory,
            attn_masks=None,
            key_padding_mask=mask_flatten,
            reference_points=topk_coords_unact,
            spatial_shapes=spatial_shapes,
            level_start_index=level_start_index,
            valid_ratios=valid_ratios,
            reg_branches=reg_branches,
            **kwargs,
        )
        return final_state, final_references_unact
