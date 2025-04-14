import os
from setuptools import setup, find_packages
from torch.utils.cpp_extension import CUDAExtension, BuildExtension


setup(
    name="codetr",
    version="1.0.1",
    url="https://github.com/anenbergb/Co-DETR-TensorRT",
    author="Bryan Anenberg",
    author_email="anenbergb@gmail.com",
    description="MMDetection Co-DETR exported to TensorRT",
    python_requires=">=3.10",
    packages=find_packages(),
    install_requires=[
    ],
    extras_require={
        "full": [
            # tested with torch 2.6.0, torch_tensorrt 2.6.0, tensorrt 10.7.0
            "torch",
            "torch-tensorrt",
            "tensorrt",
            # copied from mmdetection/requirements/mminstall.txt v3.3.0
            "mmcv>=2.0.0rc4,<2.2.0",
            "mmengine>=0.7.1,<1.0.0",
            "mmdet @ git+https://github.com/open-mmlab/mmdetection.git@v3.3.0"
        ],
        "notebook": [
            "jupyter",
            "jupyter_contrib_nbextensions",
        ],
        "dev": ["black", "mypy", "flake8", "isort", "ipdb", "pytest", "types-requests"],
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
                    # "-O3",
                    "-O0", # debug mode
                    "-fdiagnostics-color=always",
                    "-DPy_LIMITED_API=0x03090000",  # min CPython version 3.
                    "-g", # debug mode
                ],
            
                "nvcc": [
                    # "-O3",
                    "-O0", # debug mode
                    "--use_fast_math",
                    # Explicitly targeting CUDA Compute Capability 8.9 (RTX 4090)
                    "-gencode=arch=compute_89,code=sm_89",
                    "-g", # debug mode
                ],
            },
            extra_link_args=['-Wl,--no-as-needed', '-lcuda',
                             "-O0", "-g" # debug mode
                             ],
            py_limited_api=True,
        ),
    ],
    cmdclass={"build_ext": BuildExtension.with_options(no_python_abi_suffix=True)},
    options={"bdist_wheel": {"py_limited_api": "cp39"}},
    include_package_data=True,
)

# Debugging tips: add "-g" flag to include debugging information in the generated shared object file.