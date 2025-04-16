# flake8: noqa

import ctypes
import os

import torch

_cpp_ext_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "codetr_cpp_extension.so"))
if os.path.isfile(_cpp_ext_path):
    torch.ops.load_library(_cpp_ext_path)
else:
    raise ImportError(f"C++ CUDA Extension .so not found at: {_cpp_ext_path}")


_plugin_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "libdeformable_attention_plugin.so"))
if os.path.isfile(_plugin_path):
    ctypes.CDLL(_plugin_path)
else:
    raise ImportError(f"TensorRT plugin .so not found at: {_plugin_path}")

from . import ops
from .codetr import build_CoDETR
