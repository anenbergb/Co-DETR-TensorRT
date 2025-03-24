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
    ],
    extras_require={
        "torch": [
            "torch",
            "torchvision",
        ],
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
            name="codetr.ops.ms_deformable_attn",
            sources=[
                "codetr/ops/csrc/ms_deformable_attn.cpp",
                "codetr/ops/csrc/ms_deformable_attn_kernel.cu",
            ],
            extra_compile_args={
                "cxx": ["-O2", "-g"],
                "nvcc": [
                    "-O2",
                    "--use_fast_math",
                    "-gencode=arch=compute_89,code=sm_89",  # Explicitly targeting CUDA Compute Capability 8.9

                ],
            },
            extra_link_args=['-Wl,--no-as-needed', '-lcuda'],
        ),
    ],
    cmdclass={"build_ext": BuildExtension},
)
