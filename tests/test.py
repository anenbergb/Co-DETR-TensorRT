import torch
import codetr


print(dir(torch.ops))

# If codetr is there, check what ops it has
if hasattr(torch.ops, 'codetr'):
    print(dir(torch.ops.codetr))


# Define the input parameters
batch_size = 2
num_heads = 4
num_queries = 8
embed_dim = 16
num_levels = 3
num_points = 4
height, width = 32, 32
device = "cuda:0"
dtype = torch.float16

# Create random input tensors
spatial_shapes = torch.tensor(
    [[height, width], [height // 2, width // 2], [height // 4, width // 4]], device=device, dtype=torch.int64, requires_grad=False
)
level_start_index = torch.tensor(
    [0, height * width, height * width + (height // 2) * (width // 2)], device=device, dtype=torch.int64, requires_grad=False
)
value = torch.rand(
    batch_size, sum([h * w for h, w in spatial_shapes]), num_heads, embed_dim, device=device, dtype=dtype, requires_grad=False
)
sampling_loc = torch.rand(batch_size, num_queries, num_heads, num_levels, num_points, 2, device=device, dtype=dtype, requires_grad=False)
attn_weight = torch.rand(batch_size, num_queries, num_heads, num_levels, num_points, device=device, dtype=dtype, requires_grad=False)
im2col_step = 2

# Run the forward function
args = [value, spatial_shapes, level_start_index, sampling_loc, attn_weight, im2col_step]
output = torch.ops.codetr.multi_scale_deformable_attention(*args)

# Check the output shape
assert output.shape == (batch_size, num_queries, num_heads * embed_dim)

# Check that the output is not all zeros
assert not torch.all(output == 0)

# Check that the output is on the correct device
assert output.device == torch.device(device)

# Use opcheck to check for incorrect usage of operator registration APIs
torch.library.opcheck(torch.ops.codetr.multi_scale_deformable_attention.default, args)