# flake8: noqa

import ctypes
import os
from pathlib import Path

import torch

so_files = list(Path(__file__).parent.glob("_C*.so"))
assert len(so_files) == 1, f"Expected one _C*.so file, found {len(so_files)}"
torch.ops.load_library(so_files[0])


_plugin_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "libdeformable_attention_plugin.so"))
if os.path.isfile(_plugin_path):
    ctypes.CDLL(_plugin_path)
else:
    raise ImportError(f"TensorRT plugin .so not found at: {_plugin_path}")

from . import ops
from .codetr import build_CoDETR
