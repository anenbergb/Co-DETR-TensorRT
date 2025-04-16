import os
from setuptools import find_packages, setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

def get_cuda_gencode_flags(cuda_arch: str):
    """
    Given a CUDA_ARCH like "89" or "75", return appropriate -gencode flags.
    Supports multi-arch input like "75;86;89"
    """
    arches = cuda_arch.replace(" ", "").split(";")
    flags = []
    for arch in arches:
        sm = f"sm_{arch}"
        compute = f"compute_{arch}"
        flags.extend([
            f"-gencode=arch={compute},code={sm}"
        ])
    return flags

 # Default targets CUDA Compute Capability 8.9 (RTX 4090)
cuda_arch = os.environ.get("CUDA_ARCH", "89")
cuda_gencode_flags = get_cuda_gencode_flags(cuda_arch)

setup(
    name="codetr",
    version="1.0.2",
    url="https://github.com/anenbergb/Co-DETR-TensorRT",
    author="Bryan Anenberg",
    author_email="anenbergb@gmail.com",
    description="MMDetection Co-DETR exported to TensorRT",
    python_requires=">=3.10",
    packages=find_packages(),
    include_package_data=True,
    package_data={
        "codetr": ["libdeformable_attention_plugin.so"],
    },
    install_requires=["numpy", "pytest"],
    extras_require={
        "full": [
            # tested with torch 2.6.0, torch_tensorrt 2.6.0, tensorrt 10.7.0
            "torch",
            "torchvision",
            "torch-tensorrt",
            "tensorrt",
            # copied from mmdetection/requirements/mminstall.txt v3.3.0
            "mmcv>=2.0.0rc4,<2.2.0",
            "mmengine>=0.7.1,<1.0.0",
            "mmdet @ git+https://github.com/open-mmlab/mmdetection.git@v3.3.0",
        ],
        "notebook": [
            "jupyter",
            "jupyter_contrib_nbextensions",
        ],
        "dev": ["black", "flake8", "isort", "ipdb"],
    },
    ext_modules=[
        CUDAExtension(
            name="codetr._C",
            sources=[
                "codetr/csrc/deformable_attention_torch.cpp",
                "codetr/csrc/ms_deform_attn.cu",
            ],
            extra_compile_args={
                "cxx": [
                    "-O3",
                    "-fdiagnostics-color=always",
                    "-DPy_LIMITED_API=0x03090000",  # min CPython version 3.
                ],
                "nvcc": [
                    "-O3",
                    "--use_fast_math"] + cuda_gencode_flags,
            },
            extra_link_args=["-Wl,--no-as-needed", "-lcuda", "-O3"],
            py_limited_api=True,
        ),
    ],
    cmdclass={"build_ext": BuildExtension.with_options(no_python_abi_suffix=True)},
    options={"bdist_wheel": {"py_limited_api": "cp39"}},
)

# Debugging tips: add "-g" flag to include debugging information in the generated shared object file.
