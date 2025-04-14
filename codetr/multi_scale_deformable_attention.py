import math
import warnings
from typing import Optional

import mmengine
import torch
import torch.nn as nn
from mmengine.model import BaseModule, constant_init, xavier_init

from codetr.ops import multi_scale_deformable_attention_pytorch


class MultiScaleDeformableAttention(BaseModule):
    """An attention module used in Deformable-DETR.

    Implements the multi-scale deformable attention mechanism as described in:
    "Deformable DETR: Deformable Transformers for End-to-End Object Detection"
    (https://arxiv.org/pdf/2010.04159.pdf).

    Args:
        embed_dims (int): The embedding dimension of the attention module. Default: 256.
        num_heads (int): Number of parallel attention heads. Default: 8.
        num_levels (int): Number of feature map levels used in attention. Default: 4.
        num_points (int): Number of sampling points for each query in each head. Default: 4.
        im2col_step (int): Step size used in the image-to-column operation. Default: 64.
        dropout (float): Dropout probability for the attention output. Default: 0.1.
        batch_first (bool): Whether the input tensors are batch-first (shape: `(batch, n, embed_dim)`).
            Default: False.
        norm_cfg (dict, optional): Configuration for the normalization layer. Default: None.
        init_cfg (mmengine.ConfigDict, optional): Initialization configuration. Default: None.
        value_proj_ratio (float): Expansion ratio for the value projection layer. Default: 1.0.
    """

    def __init__(
        self,
        embed_dims: int = 256,
        num_heads: int = 8,
        num_levels: int = 4,
        num_points: int = 4,
        im2col_step: int = 64,
        dropout: float = 0.1,
        batch_first: bool = False,
        norm_cfg: Optional[dict] = None,
        init_cfg: Optional[mmengine.ConfigDict] = None,
        value_proj_ratio: float = 1.0,
    ):
        """Initializes the MultiScaleDeformableAttention module.

        Raises:
            ValueError: If `embed_dims` is not divisible by `num_heads`.
            Warning: If the dimension per head is not a power of 2, which may reduce CUDA efficiency.
        """
        super().__init__(init_cfg)
        if embed_dims % num_heads != 0:
            raise ValueError(f"embed_dims must be divisible by num_heads, " f"but got {embed_dims} and {num_heads}")
        dim_per_head = embed_dims // num_heads
        self.norm_cfg = norm_cfg
        self.dropout = nn.Dropout(dropout)
        self.batch_first = batch_first

        # you'd better set dim_per_head to a power of 2
        # which is more efficient in the CUDA implementation
        def _is_power_of_2(n):
            if (not isinstance(n, int)) or (n < 0):
                raise ValueError("invalid input for _is_power_of_2: {} (type: {})".format(n, type(n)))
            return (n & (n - 1) == 0) and n != 0

        if not _is_power_of_2(dim_per_head):
            warnings.warn(
                "You'd better set embed_dims in "
                "MultiScaleDeformAttention to make "
                "the dimension of each attention head a power of 2 "
                "which is more efficient in our CUDA implementation."
            )

        self.im2col_step = im2col_step
        self.embed_dims = embed_dims
        self.num_levels = num_levels
        self.num_heads = num_heads
        self.num_points = num_points
        self.sampling_offsets = nn.Linear(embed_dims, num_heads * num_levels * num_points * 2)
        self.attention_weights = nn.Linear(embed_dims, num_heads * num_levels * num_points)
        value_proj_size = int(embed_dims * value_proj_ratio)
        self.value_proj = nn.Linear(embed_dims, value_proj_size)
        self.output_proj = nn.Linear(value_proj_size, embed_dims)
        self.init_weights()

    def init_weights(self) -> None:
        """Initializes the weights of the module.

        - Sampling offsets are initialized with a grid pattern.
        - Attention weights are initialized to zero.
        - Value and output projection layers are initialized using Xavier initialization.
        """
        constant_init(self.sampling_offsets, 0.0)
        params = next(self.parameters())
        device = params.device
        dtype = params.dtype
        thetas = torch.arange(self.num_heads, dtype=dtype, device=device) * (2.0 * math.pi / self.num_heads)
        grid_init = torch.stack([thetas.cos(), thetas.sin()], -1)
        grid_init = (
            (grid_init / grid_init.abs().max(-1, keepdim=True)[0])
            .view(self.num_heads, 1, 1, 2)
            .repeat(1, self.num_levels, self.num_points, 1)
        )
        for i in range(self.num_points):
            grid_init[:, :, i, :] *= i + 1

        self.sampling_offsets.bias.data = grid_init.view(-1)
        constant_init(self.attention_weights, val=0.0, bias=0.0)
        xavier_init(self.value_proj, distribution="uniform", bias=0.0)
        xavier_init(self.output_proj, distribution="uniform", bias=0.0)
        self._is_init = True

    def forward(
        self,
        query: torch.Tensor,
        key: Optional[torch.Tensor] = None,
        value: Optional[torch.Tensor] = None,
        identity: Optional[torch.Tensor] = None,
        query_pos: Optional[torch.Tensor] = None,
        key_padding_mask: Optional[torch.Tensor] = None,
        reference_points: Optional[torch.Tensor] = None,
        spatial_shapes: Optional[torch.Tensor] = None,
        level_start_index: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        """Forward pass for the MultiScaleDeformableAttention module.

        Args:
            query (torch.Tensor): Query tensor of shape `(num_query, bs, embed_dims)`.
            key (torch.Tensor, optional): Key tensor of shape `(num_key, bs, embed_dims)`. Default: None.
            value (torch.Tensor, optional): Value tensor of shape `(num_key, bs, embed_dims)`. Default: None.
            identity (torch.Tensor, optional): Tensor for residual connection, with the same shape as `query`.
                If None, `query` is used. Default: None.
            query_pos (torch.Tensor, optional): Positional encoding for the query. Default: None.
            key_padding_mask (torch.Tensor, optional): Mask for the key tensor, of shape `(bs, num_key)`. Default: None.
            reference_points (torch.Tensor, optional): Normalized reference points of shape
                `(bs, num_query, num_levels, 2)` or `(bs, num_query, num_levels, 4)`. Default: None.
                In the final dimension, the first two elements are the reference point centers in the range [0, 1].
                The last two elements are the width and height of the reference boxes.
            spatial_shapes (torch.Tensor, optional): Spatial shapes of feature maps at different levels, of shape
                `(num_levels, 2)`. The last dimension represents (h, w). Default: None.
            level_start_index (torch.Tensor, optional): Start index of each level in the flattened feature map,
                of shape `(num_levels,)`. It can be represented as [0, h_0*w_0, h_0*w_0+h_1*w_1, ...]. Default: None.

        Returns:
            torch.Tensor: Output tensor of shape `(num_query, bs, embed_dims)`.

        Raises:
            ValueError: If the last dimension of `reference_points` is not 2 or 4.
        """

        if value is None:
            value = query

        if identity is None:
            identity = query
        if query_pos is not None:
            query = query + query_pos
        if not self.batch_first:
            # change to (bs, num_query ,embed_dims)
            query = query.permute(1, 0, 2)
            value = value.permute(1, 0, 2)

        bs, num_query, _ = query.shape
        bs, num_value, _ = value.shape
        # skip the assertion to avoid graph breaks when exporting to TensorRT
        # assert (spatial_shapes[:, 0] * spatial_shapes[:, 1]).sum() == num_value

        value = self.value_proj(value)
        if key_padding_mask is not None:
            value = value.masked_fill(key_padding_mask[..., None], 0.0)
        value = value.view(bs, num_value, self.num_heads, -1)
        sampling_offsets = self.sampling_offsets(query).view(
            bs, num_query, self.num_heads, self.num_levels, self.num_points, 2
        )
        attention_weights = self.attention_weights(query).view(
            bs, num_query, self.num_heads, self.num_levels * self.num_points
        )
        attention_weights = attention_weights.softmax(-1)

        attention_weights = attention_weights.view(bs, num_query, self.num_heads, self.num_levels, self.num_points)
        if reference_points.shape[-1] == 2:
            offset_normalizer = torch.stack([spatial_shapes[..., 1], spatial_shapes[..., 0]], -1)
            sampling_locations = (
                reference_points[:, :, None, :, None, :]
                + sampling_offsets / offset_normalizer[None, None, None, :, None, :]
            )
        elif reference_points.shape[-1] == 4:
            sampling_locations = (
                reference_points[:, :, None, :, None, :2]
                + sampling_offsets / self.num_points * reference_points[:, :, None, :, None, 2:] * 0.5
            )
        else:
            raise ValueError(
                f"Last dim of reference_points must be" f" 2 or 4, but get {reference_points.shape[-1]} instead."
            )

        # Assume that cuda is available
        if value.is_cuda:
            output = torch.ops.codetr.multi_scale_deformable_attention(
                value, spatial_shapes, level_start_index, sampling_locations, attention_weights, self.im2col_step
            )
        else:
            output = multi_scale_deformable_attention_pytorch(
                value, spatial_shapes, sampling_locations, attention_weights
            )

        output = self.output_proj(output)

        if not self.batch_first:
            # (num_query, bs ,embed_dims)
            output = output.permute(1, 0, 2)

        return self.dropout(output) + identity
