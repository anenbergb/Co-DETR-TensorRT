import torch
import pytest
from codetr.ops.ms_deform_attn import ms_deform_attn_forward, ms_deform_attn_backward
from codetr.ops.multi_scale_deform_attn import multi_scale_deformable_attn_pytorch


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
def test_ms_deform_attn_forward(dtype):
    # Define the input parameters
    batch_size = 2
    num_heads = 4
    num_queries = 8
    embed_dim = 16
    num_levels = 3
    num_points = 4
    height, width = 32, 32
    device = "cuda:0"

    # Create random input tensors
    spatial_shapes = torch.tensor([[height, width], [height // 2, width // 2], [height // 4, width // 4]], device=device, dtype=torch.int64)
    level_start_index = torch.tensor([0, height * width, height * width + (height // 2) * (width // 2)], device=device, dtype=torch.int64)
    value = torch.rand(batch_size, sum([h * w for h, w in spatial_shapes]), num_heads, embed_dim, device=device, dtype=dtype)
    sampling_loc = torch.rand(batch_size, num_queries, num_heads, num_levels, num_points, 2, device=device, dtype=dtype)
    attn_weight = torch.rand(batch_size, num_queries, num_heads, num_levels, num_points, device=device, dtype=dtype)
    im2col_step = 2

    # Run the forward function
    output = ms_deform_attn_forward(value, spatial_shapes, level_start_index, sampling_loc, attn_weight, im2col_step)

    # Run the PyTorch implementation
    output_pytorch = multi_scale_deformable_attn_pytorch(value, spatial_shapes, sampling_loc, attn_weight)

    # Check the output shape
    assert output.shape == (batch_size, num_queries, num_heads * embed_dim)

    # Check that the output is not all zeros
    assert not torch.all(output == 0)

    # Check that the output is on the correct device
    assert output.device == torch.device(device)

    # Check that the outputs are close
    assert torch.allclose(output, output_pytorch, rtol=1e-2, atol=1e-3)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
def test_ms_deform_attn_backward(dtype):
    # Define the input parameters
    batch_size = 2
    num_heads = 4
    num_queries = 8
    embed_dim = 16
    num_levels = 3
    num_points = 4
    height, width = 32, 32
    device = "cuda:0"

    # Create random input tensors
    spatial_shapes = torch.tensor([[height, width], [height // 2, width // 2], [height // 4, width // 4]], device=device, dtype=torch.int64)
    level_start_index = torch.tensor([0, height * width, height * width + (height // 2) * (width // 2)], device=device, dtype=torch.int64)
    value = torch.rand(batch_size, sum([h * w for h, w in spatial_shapes]), num_heads, embed_dim, device=device, dtype=dtype)
    sampling_loc = torch.rand(batch_size, num_queries, num_heads, num_levels, num_points, 2, device=device, dtype=dtype)
    attn_weight = torch.rand(batch_size, num_queries, num_heads, num_levels, num_points, device=device, dtype=dtype)
    grad_output = torch.rand(batch_size, num_queries, num_heads * embed_dim, device=device, dtype=dtype)
    im2col_step = 2

    # Create tensors for gradients
    grad_value = torch.zeros_like(value)
    grad_sampling_loc = torch.zeros_like(sampling_loc)
    grad_attn_weight = torch.zeros_like(attn_weight)

    # Run the backward function
    ms_deform_attn_backward(
        value,
        spatial_shapes,
        level_start_index,
        sampling_loc,
        attn_weight,
        grad_output,
        grad_value,
        grad_sampling_loc,
        grad_attn_weight,
        im2col_step,
    )

    # Check the gradients are not all zeros
    assert not torch.all(grad_value == 0)
    assert not torch.all(grad_sampling_loc == 0)
    assert not torch.all(grad_attn_weight == 0)

    # Check that the gradients are on the correct device
    assert grad_value.device == torch.device(device)
    assert grad_sampling_loc.device == torch.device(device)
    assert grad_attn_weight.device == torch.device(device)


if __name__ == "__main__":
    pytest.main([__file__])
