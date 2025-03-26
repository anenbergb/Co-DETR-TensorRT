import torch
import pytest
from torch.utils import benchmark

from codetr.swin import SwinTransformer
from mmengine.config import Config

import torch_tensorrt

torch_tensorrt.runtime.set_multi_device_safe_mode(False)

# model settings
swin_config = dict(
    pretrain_img_size=384,
    embed_dims=192,
    depths=[2, 2, 18, 2],
    num_heads=[6, 12, 24, 48],
    window_size=12,
    mlp_ratio=4,
    qkv_bias=True,
    qk_scale=None,
    drop_rate=0.0,
    attn_drop_rate=0.0,
    drop_path_rate=0.3,
    patch_norm=True,
    out_indices=(0, 1, 2, 3),
    # Please only add indices that would be used
    # in FPN, otherwise some parameter will not be used
    with_cp=True,
    convert_weights=True,
)

#     neck=dict(in_channels=[192, 384, 768, 1536]),
#     query_head=dict(
#         dn_cfg=dict(box_noise_scale=0.4, group_cfg=dict(num_dn_queries=500)),
#         transformer=dict(encoder=dict(with_cp=6)))
# )


def _benchmark_runtime(run_pytorch_model, run_exported_model, run_tensorrt_model, iterations=10):
    with torch.inference_mode():
        # Warm-up runs
        _ = run_pytorch_model()
        _ = run_exported_model()
        _ = run_tensorrt_model()

        # Benchmark
        t0 = benchmark.Timer(
            stmt="run_pytorch_model()",
            globals={"run_pytorch_model": run_pytorch_model},
            num_threads=1,
        )

        t1 = benchmark.Timer(
            stmt="run_exported_model()",
            globals={"run_exported_model": run_exported_model},
            num_threads=1,
        )

        t2 = benchmark.Timer(
            stmt="run_tensorrt_model()",
            globals={"run_tensorrt_model": run_tensorrt_model},
            num_threads=1,
        )

        # Run benchmarks
        pytorch_time = t0.timeit(iterations)
        exported_time = t1.timeit(iterations)
        tensorrt_time = t2.timeit(iterations)

        print(f"\nPyTorch implementation: {pytorch_time}")
        print(f"Exported implementation: {exported_time}")
        print(f"TensorRT implementation: {tensorrt_time}")

        # Calculate speedups
        speedup_export = pytorch_time.mean / exported_time.mean
        speedup_trt = pytorch_time.mean / tensorrt_time.mean

        print(f"\nExported speedup: {speedup_export:.2f}x")
        print(f"TensorRT speedup: {speedup_trt:.2f}x")


def test_swin_transformer():
    torch.manual_seed(42)  # For reproducibility

    iterations = 3
    device = "cuda:0"
    optimization_level = 3  # default is 3, max is 5

    cfg = Config(swin_config)
    model = SwinTransformer(**cfg).to(device)
    model.init_weights()
    model.eval()

    batch_inputs = torch.randn(1, 3, 1280, 1920, dtype=torch.float32, device=device)

    def run_pytorch_model():
        return model(batch_inputs)

    # pytest.set_trace()
    with torch.inference_mode():
        model_export = torch.export.export(
            model,
            args=(batch_inputs,),
            strict=True,
        )
        print(f"✅ Exported {type(model)} to {type(model_export)}")

        def run_exported_model():
            return model_export.module()(batch_inputs)

        model_trt = torch_tensorrt.dynamo.compile(
            model_export,
            inputs=(batch_inputs,),
            enabled_precisions={torch.float32},
            optimization_level=optimization_level,
        )
        print(f"✅ Compiled {type(model_export)} to TensorRT")

        def run_tensorrt_model():
            return model_trt(batch_inputs)

        # Verify outputs match
        output_pytorch = run_pytorch_model()
        output_export = run_exported_model()
        output_trt = run_tensorrt_model()

    for i in range(len(output_pytorch)):
        torch.testing.assert_close(output_pytorch[i], output_export[i], rtol=1e-3, atol=1e-3)
        torch.testing.assert_close(output_pytorch[i], output_trt[i], rtol=1e-1, atol=1e-1)

    _benchmark_runtime(run_pytorch_model, run_exported_model, run_tensorrt_model, iterations=iterations)


if __name__ == "__main__":
    pytest.main([__file__])
