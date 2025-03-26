import torch
import pytest
from torch.autograd import gradcheck

# from codetr.ops.ms_deform_attn import ms_deform_attn_forward, ms_deform_attn_backward
# from codetr.ops.ops import (
#     multi_scale_deformable_attn_pytorch,
#     MultiScaleDeformableAttention,
#     MultiScaleDeformableAttnFunction,
# )
import codetr

from codetr.ops import multi_scale_deformable_attention_pytorch
from codetr.multi_scale_deformable_attention import MultiScaleDeformableAttention


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

    with torch.no_grad():
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
        sampling_loc = torch.rand(
            batch_size, num_queries, num_heads, num_levels, num_points, 2, device=device, dtype=dtype
        )
        attn_weight = torch.rand(batch_size, num_queries, num_heads, num_levels, num_points, device=device, dtype=dtype)
        im2col_step = 2

        args = value, spatial_shapes, level_start_index, sampling_loc, attn_weight, im2col_step
        torch.library.opcheck(torch.ops.codetr.multi_scale_deformable_attention.default, args)

        # Run the forward function
        output = torch.ops.codetr.multi_scale_deformable_attention(*args)

        # Run the PyTorch implementation
        output_pytorch = multi_scale_deformable_attention_pytorch(value, spatial_shapes, sampling_loc, attn_weight)

    # Check the output shape
    assert output.shape == (batch_size, num_queries, num_heads * embed_dim)

    # Check that the output is not all zeros
    assert not torch.all(output == 0)

    # Check that the output is on the correct device
    assert output.device == torch.device(device)

    # Check that the outputs are close
    torch.testing.assert_close(output, output_pytorch, rtol=1e-2, atol=1e-3)


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
    grad_output = torch.rand(batch_size, num_queries, num_heads * embed_dim, device=device, dtype=dtype)
    im2col_step = 2

    # Create tensors for gradients
    grad_value = torch.zeros_like(value)
    grad_sampling_loc = torch.zeros_like(sampling_loc)
    grad_attn_weight = torch.zeros_like(attn_weight)

    # Run the backward function
    torch.ops.codetr.multi_scale_deformable_attention_backward(
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


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
def test_MultiScaleDeformableAttention(dtype):
    # Define the input parameters
    batch_size = 2
    num_heads = 4
    num_queries = 8
    embed_dim = 16
    num_levels = 3
    num_points = 4
    height, width = 32, 32
    device = "cuda:0"
    im2col_step = 2

    # Create random input tensors
    spatial_shapes = torch.tensor(
        [[height, width], [height // 2, width // 2], [height // 4, width // 4]], device=device, dtype=torch.int64
    )
    level_start_index = torch.tensor(
        [0, height * width, height * width + (height // 2) * (width // 2)], device=device, dtype=torch.int64
    )

    # Create query tensor with proper shape (batch_size, num_queries, embed_dim)
    query = torch.rand(num_queries, batch_size, embed_dim, device=device, dtype=dtype)

    # Create value tensor with proper shape
    spatial_len = sum([h * w for h, w in spatial_shapes])
    value = torch.rand(spatial_len, batch_size, embed_dim, device=device, dtype=dtype)

    # Create reference points (needed for sampling_locations calculation)
    # Shape: (batch_size, num_queries, num_levels, 2)
    reference_points = torch.rand(batch_size, num_queries, num_levels, 2, device=device, dtype=dtype)

    # Create the MultiScaleDeformableAttention module
    msda = (
        MultiScaleDeformableAttention(embed_dim, num_heads, num_levels, num_points, im2col_step, batch_first=False)
        .to(device)
        .to(dtype)
    )

    # Run the forward function with correct arguments
    output = msda(
        query=query,
        value=value,
        reference_points=reference_points,
        spatial_shapes=spatial_shapes,
        level_start_index=level_start_index,
    )

    # Check the output shape
    assert output.shape == (num_queries, batch_size, embed_dim)

    # Check that the output is not all zeros
    assert not torch.all(output == 0)

    # Check that the output is on the correct device
    assert output.device == torch.device(device)


@pytest.mark.parametrize("device", ["cpu", "cuda:0"])
def test_multiscale_deformable_attention(device):
    with pytest.raises(ValueError):
        # embed_dims must be divisible by num_heads,
        MultiScaleDeformableAttention(
            embed_dims=256,
            num_heads=7,
        )
    device = torch.device(device)
    msda = MultiScaleDeformableAttention(embed_dims=3, num_levels=2, num_heads=3)
    msda.init_weights()
    num_query = 5
    bs = 1
    embed_dims = 3
    query = torch.rand(num_query, bs, embed_dims).to(device)
    key = torch.rand(num_query, bs, embed_dims).to(device)
    spatial_shapes = torch.Tensor([[2, 2], [1, 1]]).long().to(device)
    level_start_index = torch.Tensor([0, 4]).long().to(device)
    reference_points = torch.rand(bs, num_query, 2, 2).to(device)
    msda.to(device)
    msda(
        query,
        key,
        key,
        reference_points=reference_points,
        spatial_shapes=spatial_shapes,
        level_start_index=level_start_index,
    )

    # test with value_proj_ratio
    embed_dims = 6
    value_proj_ratio = 0.5
    query = torch.rand(num_query, bs, embed_dims).to(device)
    key = torch.rand(num_query, bs, embed_dims).to(device)
    msda = MultiScaleDeformableAttention(
        embed_dims=embed_dims, num_levels=2, num_heads=3, value_proj_ratio=value_proj_ratio
    )
    msda.init_weights()
    msda.to(device)
    msda(
        query,
        key,
        key,
        reference_points=reference_points,
        spatial_shapes=spatial_shapes,
        level_start_index=level_start_index,
    )


def test_forward_multi_scale_deformable_attn_pytorch():
    N, M, D = 1, 2, 2
    Lq, L, P = 2, 2, 2
    shapes = torch.as_tensor([(6, 4), (3, 2)], dtype=torch.long)
    S = sum((H * W).item() for H, W in shapes)

    torch.manual_seed(3)
    value = torch.rand(N, S, M, D) * 0.01
    sampling_locations = torch.rand(N, Lq, M, L, P, 2)
    attention_weights = torch.rand(N, Lq, M, L, P) + 1e-5
    attention_weights /= attention_weights.sum(-1, keepdim=True).sum(-2, keepdim=True)

    multi_scale_deformable_attention_pytorch(
        value.double(), shapes, sampling_locations.double(), attention_weights.double()
    ).detach()


def test_forward_equal_with_pytorch_double():
    N, M, D = 1, 2, 2
    Lq, L, P = 2, 2, 2
    shapes = torch.as_tensor([(6, 4), (3, 2)], dtype=torch.long)
    level_start_index = torch.cat((shapes.new_zeros((1,)), shapes.prod(1).cumsum(0)[:-1]))
    S = sum((H * W).item() for H, W in shapes)

    torch.manual_seed(3)
    value = torch.rand(N, S, M, D) * 0.01
    sampling_locations = torch.rand(N, Lq, M, L, P, 2)
    attention_weights = torch.rand(N, Lq, M, L, P) + 1e-5
    attention_weights /= attention_weights.sum(-1, keepdim=True).sum(-2, keepdim=True)
    im2col_step = 2
    output_pytorch = (
        multi_scale_deformable_attention_pytorch(
            value.double(), shapes, sampling_locations.double(), attention_weights.double()
        )
        .detach()
        .cpu()
    )

    output_cuda = (
        torch.ops.codetr.multi_scale_deformable_attention(
            value.cuda().double(),
            shapes.cuda(),
            level_start_index.cuda(),
            sampling_locations.cuda().double(),
            attention_weights.cuda().double(),
            im2col_step,
        )
        .detach()
        .cpu()
    )
    torch.testing.assert_close(output_cuda, output_pytorch)
    max_abs_err = (output_cuda - output_pytorch).abs().max()
    max_rel_err = ((output_cuda - output_pytorch).abs() / output_pytorch.abs()).max()
    assert max_abs_err < 1e-18
    assert max_rel_err < 1e-15


def test_forward_equal_with_pytorch_float():
    device = "cuda:0"
    N, M, D = 1, 2, 2
    Lq, L, P = 2, 2, 2
    shapes = torch.as_tensor([(6, 4), (3, 2)], dtype=torch.long)
    level_start_index = torch.cat((shapes.new_zeros((1,)), shapes.prod(1).cumsum(0)[:-1]))
    S = sum((H * W).item() for H, W in shapes)

    torch.manual_seed(3)
    value = torch.rand(N, S, M, D) * 0.01
    sampling_locations = torch.rand(N, Lq, M, L, P, 2)
    attention_weights = torch.rand(N, Lq, M, L, P) + 1e-5
    attention_weights /= attention_weights.sum(-1, keepdim=True).sum(-2, keepdim=True)
    im2col_step = 2
    output_pytorch = (
        multi_scale_deformable_attention_pytorch(value, shapes, sampling_locations, attention_weights).detach().cpu()
    )

    output_device = (
        torch.ops.codetr.multi_scale_deformable_attention(
            value.to(device),
            shapes.to(device),
            level_start_index.to(device),
            sampling_locations.to(device),
            attention_weights.to(device),
            im2col_step,
        )
        .detach()
        .cpu()
    )
    torch.testing.assert_close(output_device, output_pytorch, rtol=1e-2, atol=1e-3)
    max_abs_err = (output_device - output_pytorch).abs().max()
    max_rel_err = ((output_device - output_pytorch).abs() / output_pytorch.abs()).max()
    assert max_abs_err < 1e-9
    assert max_rel_err < 1e-6


def test_forward_equal_with_pytorch_half():
    device = "cuda:0"
    N, M, D = 1, 2, 2
    Lq, L, P = 2, 2, 2
    shapes = torch.as_tensor([(6, 4), (3, 2)], dtype=torch.long)
    level_start_index = torch.cat((shapes.new_zeros((1,)), shapes.prod(1).cumsum(0)[:-1]))
    S = sum((H * W).item() for H, W in shapes)

    torch.manual_seed(3)
    value = torch.rand(N, S, M, D) * 0.01
    sampling_locations = torch.rand(N, Lq, M, L, P, 2)
    attention_weights = torch.rand(N, Lq, M, L, P) + 1e-5
    attention_weights /= attention_weights.sum(-1, keepdim=True).sum(-2, keepdim=True)
    im2col_step = 2
    output_pytorch = (
        multi_scale_deformable_attention_pytorch(
            value.half(), shapes, sampling_locations.half(), attention_weights.half()
        )
        .detach()
        .cpu()
    )

    output_device = (
        torch.ops.codetr.multi_scale_deformable_attention(
            value.to(device).half(),
            shapes.to(device),
            level_start_index.to(device),
            sampling_locations.to(device).half(),
            attention_weights.to(device).half(),
            im2col_step,
        )
        .detach()
        .cpu()
    )
    torch.testing.assert_close(output_device, output_pytorch, rtol=1e-2, atol=1e-3)
    max_abs_err = (output_device - output_pytorch).abs().max()
    max_rel_err = ((output_device - output_pytorch).abs() / output_pytorch.abs()).max()
    assert max_abs_err < 1e-5
    assert max_rel_err < 1e-2


@pytest.mark.parametrize(
    "channels",
    [
        4,
        30,
        32,
        64,
        71,
        1025,
    ],
)
@pytest.mark.parametrize("dtype", [torch.float, torch.double, torch.half])
def test_gradient_numerical(channels, dtype, grad_value=True, grad_sampling_loc=True, grad_attn_weight=True):
    device = "cuda:0"
    N, M, _ = 1, 2, 2
    Lq, L, P = 2, 2, 2
    shapes = torch.as_tensor([(3, 2), (2, 1)], dtype=torch.long).to(device)
    level_start_index = torch.cat((shapes.new_zeros((1,)), shapes.prod(1).cumsum(0)[:-1]))
    S = sum((H * W).item() for H, W in shapes)

    value = torch.rand(N, S, M, channels).to(device) * 0.01
    sampling_locations = torch.rand(N, Lq, M, L, P, 2).to(device)
    attention_weights = torch.rand(N, Lq, M, L, P).to(device) + 1e-5
    attention_weights /= attention_weights.sum(-1, keepdim=True).sum(-2, keepdim=True)
    im2col_step = 2

    func = torch.ops.codetr.multi_scale_deformable_attention

    value.requires_grad = grad_value
    sampling_locations.requires_grad = grad_sampling_loc
    attention_weights.requires_grad = grad_attn_weight

    dtype = torch.double
    eps = 1e-6

    assert gradcheck(
        func,
        (
            value.to(dtype),
            shapes,
            level_start_index,
            sampling_locations.to(dtype),
            attention_weights.to(dtype),
            im2col_step,
        ),
        eps=eps,
        atol=1e-2,
    )


def test_export():
    device = torch.device("cuda:0")

    embed_dims = 64
    num_levels = 3
    num_heads = 8
    num_points = 4
    im2col_step = 2
    msda = MultiScaleDeformableAttention(
        embed_dims=embed_dims,
        num_heads=num_heads,
        num_levels=num_levels,
        num_points=num_points,
        im2col_step=im2col_step,
        batch_first=False,
    )
    msda.init_weights()
    num_query = 10
    bs = 1
    height, width = 32, 32
    query = torch.rand(num_query, bs, embed_dims, device=device)
    spatial_shapes = torch.tensor(
        [[height, width], [height // 2, width // 2], [height // 4, width // 4]], device=device, dtype=torch.int64
    )
    level_start_index = torch.tensor(
        [0, height * width, height * width + (height // 2) * (width // 2)], device=device, dtype=torch.int64
    )
    reference_points = torch.rand(bs, num_query, num_levels, 2, device=device)

    spatial_len = sum([h * w for h, w in spatial_shapes])
    value = torch.rand(spatial_len, bs, embed_dims, device=device)

    msda.to(device)

    with torch.inference_mode():
        output = msda(
            query,
            value=value,
            reference_points=reference_points,
            spatial_shapes=spatial_shapes,
            level_start_index=level_start_index,
        )

        msda_export = torch.export.export(
            msda,
            args=(query,),
            kwargs={
                "value": value,
                "reference_points": reference_points,
                "spatial_shapes": spatial_shapes,
                "level_start_index": level_start_index,
            },
            strict=True,
        )
        msda_export_out = msda_export.module()(
            query,
            value=value,
            reference_points=reference_points,
            spatial_shapes=spatial_shapes,
            level_start_index=level_start_index,
        )
        torch.testing.assert_close(output, msda_export_out, rtol=1e-3, atol=1e-3)


def test_benchmark_performance():
    """Compare performance between CUDA and PyTorch implementations"""
    import torch.utils.benchmark as benchmark

    iterations = 100

    device = "cuda:0"
    N, M, D = 1, 8, 64  # batch size, num_heads, embed_dim_per_head
    Lq, L, P = 100, 4, 4  # num_queries, num_levels, num_points
    shapes = torch.as_tensor([(64, 64), (32, 32), (16, 16), (8, 8)], dtype=torch.long)
    level_start_index = torch.cat((shapes.new_zeros((1,)), shapes.prod(1).cumsum(0)[:-1]))
    S = sum((H * W).item() for H, W in shapes)
    im2col_step = 2

    print(f"\nPerformance benchmark: N={N}, M={M}, D={D}, Lq={Lq}, L={L}, P={P}")

    for dtype in [torch.float32, torch.float16]:
        print(f"\nTesting with dtype: {dtype}")

        # Create input tensors
        torch.manual_seed(42)
        value = torch.rand(N, S, M, D, device=device, dtype=dtype)
        sampling_locations = torch.rand(N, Lq, M, L, P, 2, device=device, dtype=dtype)
        attention_weights = torch.rand(N, Lq, M, L, P, device=device, dtype=dtype)
        attention_weights /= attention_weights.sum(-1, keepdim=True).sum(-2, keepdim=True)

        # Define the test functions
        def run_cuda_impl():
            return torch.ops.codetr.multi_scale_deformable_attention(
                value,
                shapes.to(device),
                level_start_index.to(device),
                sampling_locations,
                attention_weights,
                im2col_step,
            )

        def run_pytorch_impl():
            return multi_scale_deformable_attention_pytorch(
                value, shapes.to(device), sampling_locations, attention_weights
            )

        # Warm-up run
        _ = run_cuda_impl()
        _ = run_pytorch_impl()

        # Benchmark
        t0 = benchmark.Timer(
            stmt="run_cuda_impl()",
            globals={"run_cuda_impl": run_cuda_impl},
            num_threads=1,
        )

        t1 = benchmark.Timer(
            stmt="run_pytorch_impl()",
            globals={"run_pytorch_impl": run_pytorch_impl},
            num_threads=1,
        )

        print(f"CUDA implementation: {t0.timeit(iterations)}")
        print(f"PyTorch implementation: {t1.timeit(iterations)}")

        # Verify outputs match
        with torch.no_grad():
            output_cuda = run_cuda_impl()
            output_pytorch = run_pytorch_impl()

            max_abs_err = (output_cuda - output_pytorch).abs().max().item()
            if output_pytorch.abs().max().item() > 0:
                max_rel_err = ((output_cuda - output_pytorch).abs() / output_pytorch.abs().max()).max().item()
            else:
                max_rel_err = 0.0

            print(f"Max absolute error: {max_abs_err:.6e}")
            print(f"Max relative error: {max_rel_err:.6e}")

            rtol = 1e-2 if dtype == torch.float16 else 1e-5
            atol = 1e-3 if dtype == torch.float16 else 1e-6

            torch.testing.assert_close(
                output_cuda,
                output_pytorch,
                rtol=rtol,
                atol=atol,
                msg="CUDA and PyTorch implementation outputs differ significantly",
            )


if __name__ == "__main__":
    pytest.main([__file__])
