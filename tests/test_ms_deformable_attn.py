import torch
import pytest
from codetr.ops.ms_deformable_attn import ms_deformable_attn_forward

def test_ms_deformable_attn():
    # Define the input parameters
    batch_size = 2
    num_heads = 4
    num_queries = 8
    embed_dim = 16
    num_levels = 3
    num_points = 4
    height, width = 32, 32
    device = "cuda:0"
    dtype = torch.float32

    # Create random input tensors
    query = torch.rand(batch_size, num_heads, num_queries, embed_dim, dtype = dtype, device=device)
    value = torch.rand(batch_size, num_levels, height, width, embed_dim, dtype = dtype , device=device)
    spatial_shapes = torch.tensor([[height, width], [height // 2, width // 2], [height // 4, width // 4]], dtype = torch.int, device=device)
    level_start_index = torch.tensor([0, height * width, height * width + (height // 2) * (width // 2)], dtype = torch.int, device=device)
    sampling_locations = torch.rand(batch_size, num_heads, num_queries, num_levels, num_points, 2, dtype = dtype, device=device)
    attention_weights = torch.rand(batch_size, num_heads, num_queries, num_levels, num_points, dtype = dtype, device=device)

    # Run the forward function
    output = ms_deformable_attn_forward(query, value, spatial_shapes, level_start_index, sampling_locations, attention_weights)

    # Check the output shape
    assert output.shape == (batch_size, num_heads, num_queries, embed_dim)

    # Check that the output is not all zeros
    assert not torch.all(output == 0)

    # Check that the output is on the correct device
    assert output.device == torch.device(device)

    assert output.dtype == dtype

if __name__ == "__main__":
    pytest.main([__file__])