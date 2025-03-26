import os
from setuptools import setup, find_packages
from torch.utils.cpp_extension import CUDAExtension, BuildExtension


setup(
    name="codetr_cpp",
    version="1.0.0",
    url="https://github.com/anenbergb/Co-DETR-TensorRT",
    author="Bryan Anenberg",
    author_email="anenbergb@gmail.com",
    description="MMDetection Co-DETR exported to TensorRT",
    python_requires=">=3.10",
    packages=find_packages(),
    install_requires=[
        "torch"
    ],
    extras_require={
        "notebook": [
            "jupyter",
            "itkwidgets",
            "jupyter_contrib_nbextensions",
            "plotly",
            "seaborn",
        ],
        "dev": ["black", "mypy", "flake8", "isort", "ipdb", "pytest"],
    },
    ext_modules=[
        CUDAExtension(
            name="codetr_cpp._C",
            sources=[
                "codetr_cpp/csrc/ms_deform_attn.cpp",
                "codetr_cpp/csrc/ms_deform_attn_cuda.cu",
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
            extra_link_args=[
                # '-Wl,--no-as-needed', '-lcuda',
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