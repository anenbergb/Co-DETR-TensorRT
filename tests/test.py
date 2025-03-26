import torch
# torch.ops.load_library("/home/bryan/src/Co-DETR-TensorRT/codetr/ops/_C.abi3.so")
from torch.ops.codetr_cpp import ms_deform_attn_forward

# from codetr.ops import ms_deform_attn_forward
# from codetr_cpp import ms_deform_attn_forward


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
spatial_shapes = torch.tensor(
    [[height, width], [height // 2, width // 2], [height // 4, width // 4]], device=device, dtype=torch.int64
)
level_start_index = torch.tensor(
    [0, height * width, height * width + (height // 2) * (width // 2)], device=device, dtype=torch.int64
)
value = torch.rand(
    batch_size, sum([h * w for h, w in spatial_shapes]), num_heads, embed_dim, device=device, dtype=dtype
)
sampling_loc = torch.rand(batch_size, num_queries, num_heads, num_levels, num_points, 2, device=device, dtype=dtype)
attn_weight = torch.rand(batch_size, num_queries, num_heads, num_levels, num_points, device=device, dtype=dtype)
im2col_step = 2

# Run the forward function
output = ms_deform_attn_forward(value, spatial_shapes, level_start_index, sampling_loc, attn_weight, im2col_step)

# Check the output shape
assert output.shape == (batch_size, num_queries, num_heads * embed_dim)

# Check that the output is not all zeros
assert not torch.all(output == 0)

# Check that the output is on the correct device
assert output.device == torch.device(device)