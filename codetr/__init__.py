# flake8: noqa

import ctypes
import os
from pathlib import Path

import torch

so_files = list(Path(__file__).parent.glob("_C*.so"))
assert len(so_files) == 1, f"Expected one _C*.so file, found {len(so_files)}"
torch.ops.load_library(so_files[0])


plugin_lib_file_path = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "./csrc/build/libdeformable_attention_plugin.so")
)
ctypes.CDLL(plugin_lib_file_path)

from . import ops
from .codetr import build_CoDETR
