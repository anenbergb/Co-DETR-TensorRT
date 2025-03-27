import torch
from torch.utils import benchmark


def benchmark_runtime(run_pytorch_model, run_exported_model, run_tensorrt_model, iterations=10):
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
