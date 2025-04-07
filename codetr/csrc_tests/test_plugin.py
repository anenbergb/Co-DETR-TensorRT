import os
import pytest
import tensorrt as trt
import ctypes
import numpy as np

from common import EXPLICIT_BATCH, allocate_buffers, do_inference, free_buffers
from common_runtime import load_plugin_lib

TRT_LOGGER = trt.Logger(trt.Logger.INFO)


PLUGIN_NAME = "DeformableAttentionPlugin"
PLUGIN_VERSION = "1"

"""
pip install cuda-python
"""


class DeformAttnInputs:
    def __init__(
        self,
        batch_size: int = 2,
        num_heads: int = 4,
        num_queries: int = 8,
        embed_dim: int = 16,
        num_levels: int = 3,
        num_points: int = 4,
        height: int = 32,
        width: int = 32,
        dtype: np.dtype = np.float32,
    ):
        self.batch_size = batch_size
        self.num_heads = num_heads
        self.num_queries = num_queries
        self.embed_dim = embed_dim
        self.num_levels = num_levels
        self.num_points = num_points
        self.spatial_shapes = np.array(
            [[height, width], [height // 2, width // 2], [height // 4, width // 4]], dtype=np.int64
        )
        self.level_start_index = np.array(
            [0, height * width, height * width + (height // 2) * (width // 2)], dtype=np.int64
        )
        self.value = np.random.rand(
            batch_size, sum([h * w for h, w in self.spatial_shapes]), num_heads, embed_dim
        ).astype(dtype)
        self.sampling_loc = np.random.rand(batch_size, num_queries, num_heads, num_levels, num_points, 2).astype(dtype)
        self.attn_weight = np.random.rand(batch_size, num_queries, num_heads, num_levels, num_points).astype(dtype)

    def iter_shapes(self):
        names = ["value", "spatial_shapes", "level_start_index", "sampling_loc", "attn_weight"]
        outputs = [
            {"name": name, "dtype": getattr(self, name).dtype, "shape": getattr(self, name).shape} for name in names
        ]
        return iter(outputs)


# Build engine from a plugin.
def build_engine_from_plugin(
    plugin_lib_file_path: str, deform_attn_inputs: DeformAttnInputs, im2col_step: int = 2, dtype: str = "float32"
) -> trt.ICudaEngine:
    assert dtype in ["float32", "float16"], "dtype must be float32 or float16"

    builder = trt.Builder(TRT_LOGGER)
    network = builder.create_network(EXPLICIT_BATCH)
    config = builder.create_builder_config()
    # TensorRT runtime usually has to be initialized before `plugin_creator.create_plugin` is called.
    # Because the plugin creator may need to access the some functions, such as `getLogger`, from `NvInferRuntime.h`.
    # Otherwise, segmentation fault will occur because those functions are not accessible.
    # However, if the plugin creator does not need to access the functions from `NvInferRuntime.h`, the runtime can be initialized later.
    runtime = trt.Runtime(TRT_LOGGER)

    load_plugin_lib(plugin_lib_file_path)
    registry = trt.get_plugin_registry()
    plugin_creator = registry.get_creator(PLUGIN_NAME, PLUGIN_VERSION)
    assert plugin_creator is not None
    field_im2col_step = trt.PluginField(
        "im2col_step",
        np.array([im2col_step], dtype=np.int64),
        trt.PluginFieldType.INT64,
    )
    field_collection = trt.PluginFieldCollection([field_im2col_step])

    # Create inputs
    plugin_inputs = []
    for dinput in deform_attn_inputs.iter_shapes():
        np_type = dinput["dtype"]
        if np_type == np.float32:
            trt_type = trt.float32
        elif np_type == np.float16:
            trt_type = trt.float16
        else:
            assert np_type == np.int64
            trt_type = trt.int64

        plugin_inputs.append(
            network.add_input(
                name=dinput["name"],
                dtype=trt_type,
                shape=dinput["shape"],
            )
        )

    plugin = plugin_creator.create_plugin(
        name=PLUGIN_NAME, field_collection=field_collection, phase=trt.TensorRTPhase.BUILD
    )
    plugin_layer = network.add_plugin_v3(inputs=plugin_inputs, shape_inputs=[], plugin=plugin)

    network.mark_output(plugin_layer.get_output(0))
    plan = builder.build_serialized_network(network, config)
    engine = runtime.deserialize_cuda_engine(plan)

    registry.deregister_creator(plugin_creator)

    return engine


@pytest.mark.parametrize(
    "dtype",
    [
        "float32",
    ],
)
def test_plugin(dtype):
    plugin_lib_file_path = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "../csrc/build/libdeformable_attention_plugin.so")
    )

    deform_attn_inputs = DeformAttnInputs()

    engine = build_engine_from_plugin(plugin_lib_file_path, deform_attn_inputs, im2col_step=2, dtype=dtype)

    inputs, outputs, bindings, stream = allocate_buffers(engine=engine, profile_idx=None)

    for host_device_buffer in inputs:
        name = host_device_buffer.name
        data = getattr(deform_attn_inputs, name)
        host_device_buffer.host = data

    context = engine.create_execution_context()
    do_inference(
        context=context,
        engine=engine,
        inputs=inputs,
        outputs=outputs,
        bindings=bindings,
        stream=stream,
    )

    for input_host_device_buffer, output_host_device_buffer in zip(inputs, outputs):
        np.testing.assert_equal(input_host_device_buffer.host, output_host_device_buffer.host)

    free_buffers(inputs=inputs, outputs=outputs, stream=stream)


if __name__ == "__main__":
    pytest.main([__file__])
