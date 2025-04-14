import copy
import math
import warnings
from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
from mmengine.config import ConfigDict
from mmengine.model import BaseModule, ModuleList, Sequential
from mmengine.utils import to_2tuple

from mmcv.cnn import Linear, build_activation_layer, build_conv_layer, build_norm_layer
from mmcv.cnn.bricks.drop import build_dropout
from mmcv.cnn.bricks.scale import LayerScale


from codetr.multi_scale_deformable_attention import MultiScaleDeformableAttention


class AdaptivePadding(nn.Module):
    """Applies padding adaptively to the input.

    This module ensures that the input is fully covered by the specified filter.
    It supports two modes: "same" and "corner". The "same" mode pads zeros
    around the input, while the "corner" mode pads zeros to the bottom-right.

    Args:
        kernel_size (int | tuple): Size of the kernel. Default: 1.
        stride (int | tuple): Stride of the filter. Default: 1.
        dilation (int | tuple): Spacing between kernel elements. Default: 1.
        padding (str): Padding mode, either "same" or "corner". Default: "corner".

    Example:
        >>> kernel_size = 16
        >>> stride = 16
        >>> dilation = 1
        >>> input = torch.rand(1, 1, 15, 17)
        >>> adap_pad = AdaptivePadding(
        >>>     kernel_size=kernel_size,
        >>>     stride=stride,
        >>>     dilation=dilation,
        >>>     padding="corner")
        >>> out = adap_pad(input)
        >>> assert (out.shape[2], out.shape[3]) == (16, 32)
        >>> input = torch.rand(1, 1, 16, 17)
        >>> out = adap_pad(input)
        >>> assert (out.shape[2], out.shape[3]) == (16, 32)
    """

    def __init__(self, kernel_size=1, stride=1, dilation=1, padding="corner"):
        super().__init__()
        assert padding in ("same", "corner")

        kernel_size = to_2tuple(kernel_size)
        stride = to_2tuple(stride)
        dilation = to_2tuple(dilation)

        self.padding = padding
        self.kernel_size = kernel_size
        self.stride = stride
        self.dilation = dilation

    def get_pad_shape(self, input_shape):
        """Calculate the padding size for the input.

        Args:
            input_shape (torch.Size): Input shape as (H, W).

        Returns:
            Tuple[int]: Padding size along the height and width directions.
        """
        input_h, input_w = input_shape
        kernel_h, kernel_w = self.kernel_size
        stride_h, stride_w = self.stride
        output_h = math.ceil(input_h / stride_h)
        output_w = math.ceil(input_w / stride_w)
        pad_h = max((output_h - 1) * stride_h + (kernel_h - 1) * self.dilation[0] + 1 - input_h, 0)
        pad_w = max((output_w - 1) * stride_w + (kernel_w - 1) * self.dilation[1] + 1 - input_w, 0)
        return pad_h, pad_w

    def forward(self, x):
        """Add adaptive padding to the input tensor.

        Args:
            x (Tensor): Input tensor of shape (B, C, H, W).

        Returns:
            Tensor: Tensor with adaptive padding applied.
        """
        pad_h, pad_w = self.get_pad_shape(x.size()[-2:])
        if pad_h > 0 or pad_w > 0:
            if self.padding == "corner":
                x = F.pad(x, [0, pad_w, 0, pad_h])
            elif self.padding == "same":
                x = F.pad(x, [pad_w // 2, pad_w - pad_w // 2, pad_h // 2, pad_h - pad_h // 2])
        return x


class PatchEmbed(BaseModule):
    """Image to Patch Embedding.

    Converts an input image into patch embeddings using a convolutional layer.

    Args:
        in_channels (int): Number of input channels. Default: 3.
        embed_dims (int): Dimension of the embedding. Default: 768.
        conv_type (str): Type of convolution to generate patch embeddings. Default: "Conv2d".
        kernel_size (int): Kernel size of the embedding convolution. Default: 16.
        stride (int): Stride of the embedding convolution. Default: 16.
        padding (int | tuple | str): Padding for the embedding convolution. Supports "same" and "corner". Default: "corner".
        dilation (int): Dilation rate of the embedding convolution. Default: 1.
        bias (bool): Whether to use bias in the convolution. Default: True.
        norm_cfg (dict, optional): Configuration for the normalization layer. Default: None.
        input_size (int | tuple | None): Input size for calculating output size. Default: None.
        init_cfg (mmcv.ConfigDict, optional): Initialization configuration. Default: None.
    """

    def __init__(
        self,
        in_channels=3,
        embed_dims=768,
        conv_type="Conv2d",
        kernel_size=16,
        stride=16,
        padding="corner",
        dilation=1,
        bias=True,
        norm_cfg=None,
        input_size=None,
        init_cfg=None,
    ):
        super().__init__(init_cfg=init_cfg)

        self.embed_dims = embed_dims
        if stride is None:
            stride = kernel_size

        kernel_size = to_2tuple(kernel_size)
        stride = to_2tuple(stride)
        dilation = to_2tuple(dilation)

        if isinstance(padding, str):
            self.adaptive_padding = AdaptivePadding(
                kernel_size=kernel_size, stride=stride, dilation=dilation, padding=padding
            )
            # disable the padding of conv
            padding = 0
        else:
            self.adaptive_padding = None
        padding = to_2tuple(padding)

        self.projection = build_conv_layer(
            dict(type=conv_type),
            in_channels=in_channels,
            out_channels=embed_dims,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            bias=bias,
        )

        if norm_cfg is not None:
            self.norm = build_norm_layer(norm_cfg, embed_dims)[1]
        else:
            self.norm = None

        if input_size:
            input_size = to_2tuple(input_size)
            # `init_out_size` would be used outside to
            # calculate the num_patches
            # e.g. when `use_abs_pos_embed` outside
            self.init_input_size = input_size
            if self.adaptive_padding:
                pad_h, pad_w = self.adaptive_padding.get_pad_shape(input_size)
                input_h, input_w = input_size
                input_h = input_h + pad_h
                input_w = input_w + pad_w
                input_size = (input_h, input_w)

            # https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html
            h_out = (input_size[0] + 2 * padding[0] - dilation[0] * (kernel_size[0] - 1) - 1) // stride[0] + 1
            w_out = (input_size[1] + 2 * padding[1] - dilation[1] * (kernel_size[1] - 1) - 1) // stride[1] + 1
            self.init_out_size = (h_out, w_out)
        else:
            self.init_input_size = None
            self.init_out_size = None

    def forward(self, x):
        """Forward pass for Patch Embedding.

        Args:
            x (Tensor): Input tensor of shape (B, C, H, W).

        Returns:
            Tuple[Tensor, Tuple[int, int]]:
                - x (Tensor): Patch embeddings of shape (B, num_patches, embed_dims).
                - out_size (Tuple[int, int]): Spatial shape of the output as (H, W).
        """
        if self.adaptive_padding:
            x = self.adaptive_padding(x)

        x = self.projection(x)
        out_size = (x.shape[2], x.shape[3])
        x = x.flatten(2).transpose(1, 2)
        if self.norm is not None:
            x = self.norm(x)
        return x, out_size


class PatchMerging(BaseModule):
    """Merge patch feature maps.

    Groups feature maps by kernel size and applies normalization and linear layers
    to the grouped feature maps. This is used in Swin Transformer.

    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        kernel_size (int | tuple, optional): Kernel size for the unfolding layer. Default: 2.
        stride (int | tuple, optional): Stride for the sliding blocks. Default: None (set to kernel_size).
        padding (int | tuple | str): Padding for the unfolding layer. Supports "same" and "corner". Default: "corner".
        dilation (int | tuple, optional): Dilation for the unfolding layer. Default: 1.
        bias (bool, optional): Whether to add bias in the linear layer. Default: False.
        norm_cfg (dict, optional): Configuration for the normalization layer. Default: dict(type='LN').
        init_cfg (dict, optional): Initialization configuration. Default: None.
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=2,
        stride=None,
        padding="corner",
        dilation=1,
        bias=False,
        norm_cfg=dict(type="LN"),
        init_cfg=None,
    ):
        super().__init__(init_cfg=init_cfg)
        self.in_channels = in_channels
        self.out_channels = out_channels
        if stride:
            stride = stride
        else:
            stride = kernel_size

        kernel_size = to_2tuple(kernel_size)
        stride = to_2tuple(stride)
        dilation = to_2tuple(dilation)

        if isinstance(padding, str):
            self.adaptive_padding = AdaptivePadding(
                kernel_size=kernel_size, stride=stride, dilation=dilation, padding=padding
            )
            # disable the padding of unfold
            padding = 0
        else:
            self.adaptive_padding = None

        padding = to_2tuple(padding)
        self.sampler = nn.Unfold(kernel_size=kernel_size, dilation=dilation, padding=padding, stride=stride)

        sample_dim = kernel_size[0] * kernel_size[1] * in_channels

        if norm_cfg is not None:
            self.norm = build_norm_layer(norm_cfg, sample_dim)[1]
        else:
            self.norm = None

        self.reduction = nn.Linear(sample_dim, out_channels, bias=bias)

    def forward(self, x, input_size):
        """Forward pass for Patch Merging.

        Args:
            x (Tensor): Input tensor of shape (B, H*W, C_in).
            input_size (Tuple[int, int]): Spatial shape of the input as (H, W).

        Returns:
            Tuple[Tensor, Tuple[int, int]]:
                - x (Tensor): Merged feature map of shape (B, Merged_H * Merged_W, C_out).
                - output_size (Tuple[int, int]): Spatial shape of the output as (Merged_H, Merged_W).
        """
        B, L, C = x.shape
        assert isinstance(input_size, Sequence), f"Expect " f"input_size is " f"`Sequence` " f"but get {input_size}"

        H, W = input_size
        assert L == H * W, "input feature has wrong size"

        x = x.view(B, H, W, C).permute([0, 3, 1, 2])  # B, C, H, W

        if self.adaptive_padding:
            x = self.adaptive_padding(x)
            H, W = x.shape[-2:]

        # Use nn.Unfold to merge patch. About 25% faster than original method,
        # but need to modify pretrained model for compatibility
        # if kernel_size=2 and stride=2, x should has shape (B, 4*C, H/2*W/2)
        x = self.sampler(x)

        out_h = (
            H + 2 * self.sampler.padding[0] - self.sampler.dilation[0] * (self.sampler.kernel_size[0] - 1) - 1
        ) // self.sampler.stride[0] + 1
        out_w = (
            W + 2 * self.sampler.padding[1] - self.sampler.dilation[1] * (self.sampler.kernel_size[1] - 1) - 1
        ) // self.sampler.stride[1] + 1

        output_size = (out_h, out_w)
        x = x.transpose(1, 2)  # B, H/2*W/2, 4*C
        x = self.norm(x) if self.norm else x
        x = self.reduction(x)
        return x, output_size


class MultiheadAttention(BaseModule):
    """A wrapper for `torch.nn.MultiheadAttention`.

    Implements multi-head attention with identity connection and optional positional encoding.

    Args:
        embed_dims (int): Embedding dimension.
        num_heads (int): Number of attention heads.
        attn_drop (float): Dropout probability for attention weights. Default: 0.0.
        proj_drop (float): Dropout probability after the attention layer. Default: 0.0.
        dropout_layer (dict): Configuration for the dropout layer. Default: dict(type="Dropout", drop_prob=0.0).
        init_cfg (mmcv.ConfigDict, optional): Initialization configuration. Default: None.
        batch_first (bool): Whether the input tensors are batch-first (B, N, C). Default: False.
    """

    def __init__(
        self,
        embed_dims,
        num_heads,
        attn_drop=0.0,
        proj_drop=0.0,
        dropout_layer=dict(type="Dropout", drop_prob=0.0),
        init_cfg=None,
        batch_first=False,
        **kwargs,
    ):
        super().__init__(init_cfg)
        if "dropout" in kwargs:
            warnings.warn(
                "The arguments `dropout` in MultiheadAttention "
                "has been deprecated, now you can separately "
                "set `attn_drop`(float), proj_drop(float), "
                "and `dropout_layer`(dict) ",
                DeprecationWarning,
            )
            attn_drop = kwargs["dropout"]
            dropout_layer["drop_prob"] = kwargs.pop("dropout")

        self.embed_dims = embed_dims
        self.num_heads = num_heads
        self.batch_first = batch_first

        self.attn = nn.MultiheadAttention(embed_dims, num_heads, attn_drop, **kwargs)

        self.proj_drop = nn.Dropout(proj_drop)
        self.dropout_layer = build_dropout(dropout_layer) if dropout_layer else nn.Identity()

    def forward(
        self,
        query,
        key=None,
        value=None,
        identity=None,
        query_pos=None,
        key_pos=None,
        attn_mask=None,
        key_padding_mask=None,
        **kwargs,
    ):
        """Forward pass for Multihead Attention.

        Args:
            query (Tensor): Query tensor of shape (num_queries, bs, embed_dims) or (bs, num_queries, embed_dims).
            key (Tensor, optional): Key tensor of shape (num_keys, bs, embed_dims) or (bs, num_keys, embed_dims). Default: None.
            value (Tensor, optional): Value tensor of the same shape as `key`. Default: None.
            identity (Tensor, optional): Identity tensor for residual connection. Default: None.
            query_pos (Tensor, optional): Positional encoding for the query. Default: None.
            key_pos (Tensor, optional): Positional encoding for the key. Default: None.
            attn_mask (Tensor, optional): Attention mask of shape (num_queries, num_keys). Default: None.
            key_padding_mask (Tensor, optional): Padding mask for the key of shape (bs, num_keys). Default: None.

        Returns:
            Tensor: Output tensor of shape (num_queries, bs, embed_dims) or (bs, num_queries, embed_dims).
        """
        if key is None:
            key = query
        if value is None:
            value = key
        if identity is None:
            identity = query
        if key_pos is None:
            if query_pos is not None:
                # use query_pos if key_pos is not available
                if query_pos.shape == key.shape:
                    key_pos = query_pos
                else:
                    warnings.warn(f"position encoding of key is" f"missing in {self.__class__.__name__}.")
        if query_pos is not None:
            query = query + query_pos
        if key_pos is not None:
            key = key + key_pos

        # Because the dataflow('key', 'query', 'value') of
        # ``torch.nn.MultiheadAttention`` is (num_query, batch,
        # embed_dims), We should adjust the shape of dataflow from
        # batch_first (batch, num_query, embed_dims) to num_query_first
        # (num_query ,batch, embed_dims), and recover ``attn_output``
        # from num_query_first to batch_first.
        if self.batch_first:
            query = query.transpose(0, 1)
            key = key.transpose(0, 1)
            value = value.transpose(0, 1)

        out = self.attn(query=query, key=key, value=value, attn_mask=attn_mask, key_padding_mask=key_padding_mask)[0]

        if self.batch_first:
            out = out.transpose(0, 1)

        return identity + self.dropout_layer(self.proj_drop(out))


class FFN(BaseModule):
    """Feed-Forward Network (FFN) with identity connection.

    Args:
        embed_dims (int): Feature dimension. Default: 256.
        feedforward_channels (int): Hidden dimension of the FFN. Default: 1024.
        num_fcs (int): Number of fully connected layers. Default: 2.
        act_cfg (dict): Configuration for the activation function. Default: dict(type="ReLU").
        ffn_drop (float): Dropout probability for the FFN. Default: 0.0.
        add_identity (bool): Whether to add an identity connection. Default: True.
        dropout_layer (dict, optional): Configuration for the dropout layer. Default: None.
        init_cfg (mmcv.ConfigDict, optional): Initialization configuration. Default: None.
        layer_scale_init_value (float): Initial value for LayerScale. Default: 0.0.
    """

    def __init__(
        self,
        embed_dims=256,
        feedforward_channels=1024,
        num_fcs=2,
        act_cfg=dict(type="ReLU", inplace=True),
        ffn_drop=0.0,
        dropout_layer=None,
        add_identity=True,
        init_cfg=None,
        layer_scale_init_value=0.0,
    ):
        super().__init__(init_cfg)
        assert num_fcs >= 2, "num_fcs should be no less " f"than 2. got {num_fcs}."
        self.embed_dims = embed_dims
        self.feedforward_channels = feedforward_channels
        self.num_fcs = num_fcs

        layers = []
        in_channels = embed_dims
        for _ in range(num_fcs - 1):
            layers.append(
                Sequential(
                    Linear(in_channels, feedforward_channels), build_activation_layer(act_cfg), nn.Dropout(ffn_drop)
                )
            )
            in_channels = feedforward_channels
        layers.append(Linear(feedforward_channels, embed_dims))
        layers.append(nn.Dropout(ffn_drop))
        self.layers = Sequential(*layers)
        self.dropout_layer = build_dropout(dropout_layer) if dropout_layer else torch.nn.Identity()
        self.add_identity = add_identity

        if layer_scale_init_value > 0:
            self.gamma2 = LayerScale(embed_dims, scale=layer_scale_init_value)
        else:
            self.gamma2 = nn.Identity()

    def forward(self, x, identity=None):
        """Forward pass for the FFN.

        Args:
            x (Tensor): Input tensor of shape (B, N, embed_dims).
            identity (Tensor, optional): Identity tensor for residual connection. Default: None.

        Returns:
            Tensor: Output tensor of shape (B, N, embed_dims).
        """
        out = self.layers(x)
        out = self.gamma2(out)
        if not self.add_identity:
            return self.dropout_layer(out)
        if identity is None:
            identity = x
        return identity + self.dropout_layer(out)


class BaseTransformerLayer(BaseModule):
    """Base `TransformerLayer` for vision transformer.

    It can be built from `mmcv.ConfigDict` and support more flexible
    customization, for example, using any number of `FFN or LN ` and
    use different kinds of `attention` by specifying a list of `ConfigDict`
    named `attn_cfgs`. It is worth mentioning that it supports `prenorm`
    when you specifying `norm` as the first element of `operation_order`.
    More details about the `prenorm`: `On Layer Normalization in the
    Transformer Architecture <https://arxiv.org/abs/2002.04745>`_ .

    Args:
        attn_cfgs (list[`mmcv.ConfigDict`] | obj:`mmcv.ConfigDict` | None )):
            Configs for `self_attention` or `cross_attention` modules,
            The order of the configs in the list should be consistent with
            corresponding attentions in operation_order.
            If it is a dict, all of the attention modules in operation_order
            will be built with this config. Default: None.
        ffn_cfgs (list[`mmcv.ConfigDict`] | obj:`mmcv.ConfigDict` | None )):
            Configs for FFN, The order of the configs in the list should be
            consistent with corresponding ffn in operation_order.
            If it is a dict, all of the attention modules in operation_order
            will be built with this config.
        operation_order (tuple[str]): The execution order of operation
            in transformer. Such as ('self_attn', 'norm', 'ffn', 'norm').
            Support `prenorm` when you specifying first element as `norm`.
            Default：None.
        norm_cfg (dict): Config dict for normalization layer.
            Default: dict(type='LN').
        init_cfg (obj:`mmcv.ConfigDict`): The Config for initialization.
            Default: None.
        batch_first (bool): Key, Query and Value are shape
            of (batch, n, embed_dim)
            or (n, batch, embed_dim). Default to False.
    """

    def __init__(
        self,
        attn_cfgs=None,
        ffn_cfgs=dict(
            type="FFN",
            embed_dims=256,
            feedforward_channels=1024,
            num_fcs=2,
            ffn_drop=0.0,
            act_cfg=dict(type="ReLU", inplace=True),
        ),
        operation_order=None,
        norm_cfg=dict(type="LN"),
        init_cfg=None,
        batch_first=False,
        **kwargs,
    ):

        deprecated_args = dict(
            feedforward_channels="feedforward_channels", ffn_dropout="ffn_drop", ffn_num_fcs="num_fcs"
        )
        for ori_name, new_name in deprecated_args.items():
            if ori_name in kwargs:
                warnings.warn(
                    f"The arguments `{ori_name}` in BaseTransformerLayer "
                    f"has been deprecated, now you should set `{new_name}` "
                    f"and other FFN related arguments "
                    f"to a dict named `ffn_cfgs`. ",
                    DeprecationWarning,
                )
                ffn_cfgs[new_name] = kwargs[ori_name]

        super().__init__(init_cfg)

        self.batch_first = batch_first

        assert set(operation_order) & {"self_attn", "norm", "ffn", "cross_attn"} == set(operation_order), (
            f"The operation_order of"
            f" {self.__class__.__name__} should "
            f"contains all four operation type "
            f"{['self_attn', 'norm', 'ffn', 'cross_attn']}"
        )

        num_attn = operation_order.count("self_attn") + operation_order.count("cross_attn")
        if isinstance(attn_cfgs, dict):
            attn_cfgs = [copy.deepcopy(attn_cfgs) for _ in range(num_attn)]
        else:
            assert num_attn == len(attn_cfgs), (
                f"The length "
                f"of attn_cfg {num_attn} is "
                f"not consistent with the number of attention"
                f"in operation_order {operation_order}."
            )
            attn_cfgs = [copy.deepcopy(x) for x in attn_cfgs]

        self.num_attn = num_attn
        self.operation_order = operation_order
        self.norm_cfg = norm_cfg
        self.pre_norm = operation_order[0] == "norm"
        self.attentions = ModuleList()

        index = 0
        for operation_name in operation_order:
            if (operation_name == "self_attn") or (operation_name == "cross_attn"):
                if "batch_first" in attn_cfgs[index]:
                    assert self.batch_first == attn_cfgs[index]["batch_first"]
                else:
                    attn_cfgs[index]["batch_first"] = self.batch_first

                attn_type = attn_cfgs[index].pop("type")
                if attn_type == "MultiheadAttention":
                    attention = MultiheadAttention(**attn_cfgs[index])
                elif attn_type == "MultiScaleDeformableAttention":
                    attention = MultiScaleDeformableAttention(**attn_cfgs[index])
                else:
                    raise NotImplementedError(f"Not implemented {attn_type}")

                # Some custom attentions used as `self_attn`
                # or `cross_attn` can have different behavior.
                attention.operation_name = operation_name
                self.attentions.append(attention)
                index += 1

        self.embed_dims = self.attentions[0].embed_dims

        self.ffns = ModuleList()
        num_ffns = operation_order.count("ffn")
        if isinstance(ffn_cfgs, dict):
            ffn_cfgs = ConfigDict(ffn_cfgs)
        if isinstance(ffn_cfgs, dict):
            ffn_cfgs = [copy.deepcopy(ffn_cfgs) for _ in range(num_ffns)]
        assert len(ffn_cfgs) == num_ffns
        for ffn_index in range(num_ffns):
            if "embed_dims" not in ffn_cfgs[ffn_index]:
                ffn_cfgs[ffn_index]["embed_dims"] = self.embed_dims
            else:
                assert ffn_cfgs[ffn_index]["embed_dims"] == self.embed_dims

            ffn_type = ffn_cfgs[ffn_index].pop("type")
            if ffn_type == "FFN":
                ffn_module = FFN(**ffn_cfgs[ffn_index])
            else:
                raise NotImplementedError(f"Not implemented {ffn_type}")
            self.ffns.append(ffn_module)

        self.norms = ModuleList()
        num_norms = operation_order.count("norm")
        for _ in range(num_norms):
            self.norms.append(build_norm_layer(norm_cfg, self.embed_dims)[1])

    def forward(
        self,
        query,
        key=None,
        value=None,
        query_pos=None,
        key_pos=None,
        attn_masks=None,
        query_key_padding_mask=None,
        key_padding_mask=None,
        **kwargs,
    ):
        """Forward function for `TransformerDecoderLayer`.

        **kwargs contains some specific arguments of attentions.

        Args:
            query (Tensor): The input query with shape
                [num_queries, bs, embed_dims] if
                self.batch_first is False, else
                [bs, num_queries embed_dims].
            key (Tensor): The key tensor with shape [num_keys, bs,
                embed_dims] if self.batch_first is False, else
                [bs, num_keys, embed_dims] .
            value (Tensor): The value tensor with same shape as `key`.
            query_pos (Tensor): The positional encoding for `query`.
                Default: None.
            key_pos (Tensor): The positional encoding for `key`.
                Default: None.
            attn_masks (List[Tensor] | None): 2D Tensor used in
                calculation of corresponding attention. The length of
                it should equal to the number of `attention` in
                `operation_order`. Default: None.
            query_key_padding_mask (Tensor): ByteTensor for `query`, with
                shape [bs, num_queries]. Only used in `self_attn` layer.
                Defaults to None.
            key_padding_mask (Tensor): ByteTensor for `query`, with
                shape [bs, num_keys]. Default: None.

        Returns:
            Tensor: forwarded results with shape [num_queries, bs, embed_dims].
        """

        norm_index = 0
        attn_index = 0
        ffn_index = 0
        identity = query
        if attn_masks is None:
            attn_masks = [None for _ in range(self.num_attn)]
        elif isinstance(attn_masks, torch.Tensor):
            attn_masks = [copy.deepcopy(attn_masks) for _ in range(self.num_attn)]
            warnings.warn(f"Use same attn_mask in all attentions in " f"{self.__class__.__name__} ")
        else:
            assert len(attn_masks) == self.num_attn, (
                f"The length of "
                f"attn_masks {len(attn_masks)} must be equal "
                f"to the number of attention in "
                f"operation_order {self.num_attn}"
            )

        for layer in self.operation_order:
            if layer == "self_attn":
                temp_key = temp_value = query
                query = self.attentions[attn_index](
                    query,
                    temp_key,
                    temp_value,
                    identity if self.pre_norm else None,
                    query_pos=query_pos,
                    key_pos=query_pos,
                    attn_mask=attn_masks[attn_index],
                    key_padding_mask=query_key_padding_mask,
                    **kwargs,
                )
                attn_index += 1
                identity = query

            elif layer == "norm":
                query = self.norms[norm_index](query)
                norm_index += 1

            elif layer == "cross_attn":
                query = self.attentions[attn_index](
                    query,
                    key,
                    value,
                    identity if self.pre_norm else None,
                    query_pos=query_pos,
                    key_pos=key_pos,
                    attn_mask=attn_masks[attn_index],
                    key_padding_mask=key_padding_mask,
                    **kwargs,
                )
                attn_index += 1
                identity = query

            elif layer == "ffn":
                query = self.ffns[ffn_index](query, identity if self.pre_norm else None)
                ffn_index += 1

        return query
