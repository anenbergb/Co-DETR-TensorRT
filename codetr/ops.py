import torch
import torch.nn.functional as F
import numpy as np
import tensorrt as trt
from typing import Tuple, Dict, Union, Sequence


import torch_tensorrt
from torch_tensorrt.dynamo.conversion.converter_utils import (
    get_trt_tensor,  # helper to map a torch.Tensor or Python scalar to a TRT ITensor
)
from torch.fx.node import Argument, Target

__all__ = ["multi_scale_deformable_attention_pytorch"]


# Registers a FakeTensor kernel (aka "meta kernel", "abstract impl")
# that describes the properties of the output Tensor given the properties
# of the input Tensor. The FakeTensor kernel is necessary for the op to
# work efficiently with `torch.compile`.


@torch.library.register_fake("codetr::multi_scale_deformable_attention")
def _(value, spatial_shapes, level_start_index, sampling_loc, attn_weight, im2col_step):
    """
    FakeTensor kernel for the `codetr::multi_scale_deformable_attention` operation.

    This kernel defines the expected output tensor properties based on the input
    tensor properties. It is used for meta-programming and optimization purposes
    with `torch.compile`.

    Args:
        value (torch.Tensor): Input tensor of shape
            `(bs, num_keys, num_heads, embed_dims // num_heads)`, where:
            - `bs`: Batch size.
            - `num_keys`: Number of keys.
            - `num_heads`: Number of attention heads.
            - `embed_dims`: Embedding dimensions.
        spatial_shapes (torch.Tensor): Tensor of shape `(num_levels, 2)` representing
            the spatial shape `(height, width)` of each feature map.
        level_start_index (torch.Tensor): Tensor of shape `(num_levels,)` representing
            the start index of each level in the flattened feature map.
        sampling_loc (torch.Tensor): Tensor of shape
            `(bs, num_queries, num_heads, num_levels, num_points, 2)` representing
            the sampling locations. The last dimension `(2)` corresponds to `(x, y)` coordinates.
        attn_weight (torch.Tensor): Tensor of shape
            `(bs, num_queries, num_heads, num_levels, num_points)` representing
            the attention weights for the sampling points.
        im2col_step (int): Step size used in the image-to-column operation.

    Returns:
        torch.Tensor: Output tensor of shape `(bs, num_queries, embed_dims)`, where:
            - `bs`: Batch size.
            - `num_queries`: Number of queries.
            - `embed_dims`: Embedding dimensions.
    """
    torch._check(value.dim() == 4)
    torch._check(spatial_shapes.dim() == 2)
    torch._check(level_start_index.dim() == 1)
    torch._check(sampling_loc.dim() == 6)
    torch._check(attn_weight.dim() == 5)

    torch._check(value.dtype == attn_weight.dtype)
    torch._check(value.dtype == sampling_loc.dtype)
    torch._check(spatial_shapes.dtype == torch.int64)
    torch._check(level_start_index.dtype == torch.int64)

    bs, num_keys, num_heads, dim_per_head = value.shape
    num_levels = spatial_shapes.shape[0]
    torch._check(spatial_shapes.shape[1] == 2)
    torch._check(level_start_index.shape[0] == num_levels)
    torch._check(sampling_loc.shape[0] == bs)
    num_queries = sampling_loc.shape[1]
    torch._check(sampling_loc.shape[2] == num_heads)
    torch._check(sampling_loc.shape[3] == num_levels)
    num_points = sampling_loc.shape[4]
    torch._check(sampling_loc.shape[5] == 2)
    torch._check(attn_weight.shape[0] == bs)
    torch._check(attn_weight.shape[1] == num_queries)
    torch._check(attn_weight.shape[2] == num_heads)
    torch._check(attn_weight.shape[3] == num_levels)
    torch._check(attn_weight.shape[4] == num_points)

    embed_dims = dim_per_head * num_heads
    return torch.empty((bs, num_queries, embed_dims), dtype=value.dtype, device=value.device)


def _backward(ctx, grad):
    value, value_spatial_shapes, value_level_start_index, sampling_locations, attention_weights = ctx.saved_tensors
    grad_value = torch.zeros_like(value)
    grad_sampling_loc = torch.zeros_like(sampling_locations)
    grad_attn_weight = torch.zeros_like(attention_weights)

    torch.ops.codetr.multi_scale_deformable_attention_backward(
        value,
        value_spatial_shapes,
        value_level_start_index,
        sampling_locations,
        attention_weights,
        grad.contiguous(),
        grad_value,
        grad_sampling_loc,
        grad_attn_weight,
        im2col_step=ctx.im2col_step,
    )

    return grad_value, None, None, grad_sampling_loc, grad_attn_weight, None


def _setup_context(ctx, inputs, output):
    value, spatial_shapes, level_start_index, sampling_loc, attn_weight, im2col_step = inputs
    ctx.im2col_step = im2col_step
    ctx.save_for_backward(value, spatial_shapes, level_start_index, sampling_loc, attn_weight)


torch.library.register_autograd("codetr::multi_scale_deformable_attention", _backward, setup_context=_setup_context)


def multi_scale_deformable_attention_pytorch(
    value: torch.Tensor,
    value_spatial_shapes: torch.Tensor,
    sampling_locations: torch.Tensor,
    attention_weights: torch.Tensor,
) -> torch.Tensor:
    """CPU version of multi-scale deformable attention.

    Args:
        value (torch.Tensor): The value has shape
            (bs, num_keys, num_heads, embed_dims//num_heads)
        value_spatial_shapes (torch.Tensor): Spatial shape of
            each feature map, has shape (num_levels, 2),
            last dimension 2 represent (h, w)
        sampling_locations (torch.Tensor): The location of sampling points,
            has shape
            (bs ,num_queries, num_heads, num_levels, num_points, 2),
            the last dimension 2 represent (x, y).
        attention_weights (torch.Tensor): The weight of sampling points used
            when calculate the attention, has shape
            (bs ,num_queries, num_heads, num_levels, num_points),

    Returns:
        torch.Tensor: has shape (bs, num_queries, embed_dims)
    """

    bs, _, num_heads, embed_dims = value.shape
    _, num_queries, num_heads, num_levels, num_points, _ = sampling_locations.shape
    value_list = value.split([H_ * W_ for H_, W_ in value_spatial_shapes], dim=1)
    sampling_grids = 2 * sampling_locations - 1
    sampling_value_list = []
    for level, (H_, W_) in enumerate(value_spatial_shapes):
        # bs, H_*W_, num_heads, embed_dims ->
        # bs, H_*W_, num_heads*embed_dims ->
        # bs, num_heads*embed_dims, H_*W_ ->
        # bs*num_heads, embed_dims, H_, W_
        value_l_ = value_list[level].flatten(2).transpose(1, 2).reshape(bs * num_heads, embed_dims, H_, W_)
        # bs, num_queries, num_heads, num_points, 2 ->
        # bs, num_heads, num_queries, num_points, 2 ->
        # bs*num_heads, num_queries, num_points, 2
        sampling_grid_l_ = sampling_grids[:, :, :, level].transpose(1, 2).flatten(0, 1)
        # bs*num_heads, embed_dims, num_queries, num_points
        sampling_value_l_ = F.grid_sample(
            value_l_, sampling_grid_l_, mode="bilinear", padding_mode="zeros", align_corners=False
        )
        sampling_value_list.append(sampling_value_l_)
    # (bs, num_queries, num_heads, num_levels, num_points) ->
    # (bs, num_heads, num_queries, num_levels, num_points) ->
    # (bs, num_heads, 1, num_queries, num_levels*num_points)
    attention_weights = attention_weights.transpose(1, 2).reshape(
        bs * num_heads, 1, num_queries, num_levels * num_points
    )
    output = (
        (torch.stack(sampling_value_list, dim=-2).flatten(-2) * attention_weights)
        .sum(-1)
        .view(bs, num_heads * embed_dims, num_queries)
    )
    return output.transpose(1, 2).contiguous()


@torch_tensorrt.dynamo.conversion.dynamo_tensorrt_converter(torch.ops.codetr.multi_scale_deformable_attention.default)
def multi_scale_deformable_attention_converter(
    ctx: torch_tensorrt.dynamo.conversion.ConversionContext,
    target: Target,
    args: Tuple[Argument, ...],
    kwargs: Dict[str, Argument],
    name: str,
) -> trt.ITensor:
    """
    Converts the `codetr::multi_scale_deformable_attention` custom operation into a
    TensorRT DeformableAttentionPlugin node within the TensorRT computation graph.

    This function dynamically maps the custom PyTorch operation to a TensorRT plugin
    during the compilation process, enabling efficient execution on NVIDIA GPUs.

    Args:
        ctx (torch_tensorrt.dynamo.conversion.ConversionContext):
            The current state of the compiler. This includes the TensorRT
            `INetworkDefinition` being constructed.
        target (Target):
            The target key for the operation, such as
            `torch.ops.codetr.multi_scale_deformable_attention.default`.
        args (Tuple[Argument, ...]):
            The arguments passed to the operation. These include:
            - `args[0]` (torch.Tensor): `value` tensor of shape
              `(bs, num_keys, num_heads, embed_dims // num_heads)`.
            - `args[1]` (torch.Tensor): `spatial_shapes` tensor of shape `(num_levels, 2)`.
            - `args[2]` (torch.Tensor): `level_start_index` tensor of shape `(num_levels,)`.
            - `args[3]` (torch.Tensor): `sampling_loc` tensor of shape
              `(bs, num_queries, num_heads, num_levels, num_points, 2)`.
            - `args[4]` (torch.Tensor): `attn_weight` tensor of shape
              `(bs, num_queries, num_heads, num_levels, num_points)`.
            - `args[5]` (int): `im2col_step`, the step size for the image-to-column operation.
        kwargs (Dict[str, Argument]):
            Additional keyword arguments passed to the operation.
        name (str):
            The name of the target operation.

    Returns:
        trt.ITensor:
            The output TensorRT ITensor generated by the plugin.

    Raises:
        RuntimeError: If the plugin creation or layer addition fails.

    Notes:
        - The `ctx.net` object represents the TensorRT `INetworkDefinition` being constructed.
        - The plugin is created using the TensorRT Plugin Registry and added to the network
          as a custom layer.
    """
    PLUGIN_NAME = "DeformableAttentionPlugin"
    PLUGIN_VERSION = "1"

    # convert the Argument to TensorRT ITensor
    # adds tensor to network by ctx.net.add_constant
    # https://github.com/pytorch/TensorRT/blob/v2.6.0/py/torch_tensorrt/dynamo/conversion/converter_utils.py#L357
    trt_value = get_trt_tensor(ctx, args[0], f"{name}_value")
    trt_spatial_shapes = get_trt_tensor(ctx, args[1], f"{name}_spatial_shapes")
    trt_level_start_index = get_trt_tensor(ctx, args[2], f"{name}_level_start_index")
    trt_sampling_loc = get_trt_tensor(ctx, args[3], f"{name}_sampling_loc")
    trt_attn_weight = get_trt_tensor(ctx, args[4], f"{name}_attn_weight")

    im2col_step = args[5]  # default 64

    field_im2col_step = trt.PluginField(
        "im2col_step",
        np.array([im2col_step], dtype=np.int64),
        trt.PluginFieldType.INT64,
    )
    field_collection = trt.PluginFieldCollection([field_im2col_step])

    registry = trt.get_plugin_registry()
    # return the plugin creator, e.g. tensorrt_bindings.tensorrt.IPluginCreatorV3One
    plugin_creator = registry.get_creator(PLUGIN_NAME, PLUGIN_VERSION)
    assert (
        plugin_creator is not None
    ), f"Plugin creator for {PLUGIN_NAME} not found. Make sure the plugin library is loaded."

    plugin = plugin_creator.create_plugin(
        name=PLUGIN_NAME, field_collection=field_collection, phase=trt.TensorRTPhase.BUILD
    )
    if not plugin:
        raise RuntimeError(f"Could not create {PLUGIN_NAME}. Make sure the plugin library is loaded.")

    layer_inputs = [
        trt_value,
        trt_spatial_shapes,
        trt_level_start_index,
        trt_sampling_loc,
        trt_attn_weight,
    ]
    # returns the tensorrt.IPluginV3Layer
    plugin_layer = ctx.net.add_plugin_v3(
        inputs=layer_inputs,
        shape_inputs=[],
        plugin=plugin,
    )
    if not plugin_layer:
        raise RuntimeError(f"Failed to create plugin layer for {PLUGIN_NAME}.")

    # The plugin has exactly one output, type trt.trt.ITensor
    trt_output = plugin_layer.get_output(0)
    return trt_output
