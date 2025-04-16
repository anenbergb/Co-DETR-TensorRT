import os
import pytest

def pytest_addoption(parser):
    parser.addoption(
        "--plugin-lib",
        action="store",
        default="../csrc/build/codetr_libdeformable_attention_plugin.so",
        help="Path to the TensorRT plugin library (.so file).",
    )

@pytest.fixture
def plugin_lib_file_path(request):
    return os.path.abspath(request.config.getoption("--plugin-lib"))
