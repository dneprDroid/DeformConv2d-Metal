from setuptools import find_packages, setup

name = "DeformConv2dConvert"
version = "0.0.1"

setup(
    name=name,
    version=version,
    description="CoreML converter for torchvision.ops.deform_conv2d",
    url="https://github.com/dneprDroid/DeformConv2d-Metal",
    author="dneprDroid",
    author_email="no@email.com",
    packages=find_packages(),
    install_requires=[
        "coremltools",
        "torch",
        "torchvision",
    ],
    python_requires=">=3.5.10",
)