from setuptools import setup, find_packages
from torch.utils.cpp_extension import CUDAExtension, BuildExtension


setup(
    name="codetr",
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
            name="codetr_cpp",
            sources=[
                "codetr/ops/src/ms_deform_attn.cpp",
                "codetr/ops/src/ms_deform_attn_cuda.cu",
            ],
            include_dirs=["codetr/ops/include"],
            extra_compile_args={
                "cxx": [
                    "-O3",
                    "-fdiagnostics-color=always",
                    "-DPy_LIMITED_API=0x03090000",  # min CPython version 3.
                ],
            
                "nvcc": [
                    "-O3",
                    "--use_fast_math",
                    # Explicitly targeting CUDA Compute Capability 8.9 (RTX 4090)
                    "-gencode=arch=compute_89,code=sm_89",  
                ],
            },
            # extra_link_args=['-Wl,--no-as-needed', '-lcuda'],
            extra_link_args = [],
            py_limited_api=True,
        ),
    ],
    cmdclass={"build_ext": BuildExtension},
    options={"bdist_wheel": {"py_limited_api": "cp39"}},
)
