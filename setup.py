"""Editable install for MedMamba-XAI."""

from setuptools import setup, find_packages

setup(
    name="MedMamba-XAI",
    version="0.1.0",
    description="Interpretable Mamba Models for High-Fidelity Medical Image Classification",
    python_requires=">=3.9",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    install_requires=[
        "einops>=0.7.0",
        "numpy>=1.24.0",
        "pillow>=10.0.0",
        "scikit-learn>=1.3.0",
        "matplotlib>=3.7.0",
        "tensorboard>=2.14.0",
        "pyyaml>=6.0",
        "tqdm>=4.65.0",
        "python-dotenv>=1.0.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.4.0",
            "pytest-cov>=4.1.0",
            "black>=23.0.0",
            "isort>=5.12.0",
            "mypy>=1.5.0",
        ]
    },
)