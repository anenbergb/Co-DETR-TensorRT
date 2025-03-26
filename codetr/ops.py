import torch
import torch.nn.functional as F

__all__ = ["multi_scale_deformable_attention_pytorch"]


# Registers a FakeTensor kernel (aka "meta kernel", "abstract impl")
# that describes what the properties of the output Tensor are given
# the properties of the input Tensor. The FakeTensor kernel is necessary
# for the op to work performantly with torch.compile.
@torch.library.register_fake("codetr::multi_scale_deformable_attention")
def _(value, spatial_shapes, level_start_index, sampling_loc, attn_weight, im2col_step):
    """
    Args:
        value (torch.Tensor): The value has shape
            (bs, num_keys, mum_heads, embed_dims//num_heads)
        spatial_shapes (torch.Tensor): Spatial shape of
            each feature map, has shape (num_levels, 2),
            last dimension 2 represent (h, w)
        level_start_index (torch.Tensor): The start index of each level.
            A tensor has shape ``(num_levels, )`` and can be represented
        sampling_loc (torch.Tensor): The location of sampling points,
            has shape
            (bs ,num_queries, num_heads, num_levels, num_points, 2),
            the last dimension 2 represent (x, y).
        attn_weight (torch.Tensor): The weight of sampling points
            used when calculate the attention, has shape
            (bs ,num_queries, num_heads, num_levels, num_points),
        im2col_step (int): The step used in image to column.

    Returns:
        torch.Tensor: has shape (bs, num_queries, embed_dims)
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
