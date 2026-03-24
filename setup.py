"""Package setup for MT-OPMNet."""

from setuptools import setup, find_packages

setup(
    name="mt-opmnet",
    version="1.0.0",
    description=(
        "Attention-Enhanced Multi-Task Deep Learning for Joint OSNR "
        "Estimation and Modulation Format Recognition"
    ),
    author="Yassir Ameen Ahmed Al-Karawi",
    license="MIT",
    packages=find_packages(),
    python_requires=">=3.9",
    install_requires=[
        "torch>=2.0.0",
        "numpy>=1.24.0",
        "scikit-learn>=1.3.0",
        "matplotlib>=3.7.0",
    ],
    entry_points={
        "console_scripts": [
            "mt-opmnet=main:main",
        ],
    },
)
