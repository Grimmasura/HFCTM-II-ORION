"""
Setup configuration for MIH-IIE package.
"""
from setuptools import setup, find_packages
import os

# Read README safely
readme_path = os.path.join(os.path.dirname(__file__), "README.md")
try:
    with open(readme_path, "r", encoding="utf-8") as f:
        long_description = f.read()
except Exception:
    long_description = "Majorana–Ironwood Hybrid Intrinsic Inference Engine"

setup(
    name="mih-iie",
    version="0.1.0-alpha",
    description="Majorana–Ironwood Hybrid Intrinsic Inference Engine",
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=find_packages(exclude=["tests", "tests.*", "legacy_orion", "legacy_orion.*"]),
    python_requires=">=3.8",
    install_requires=[
        "fastapi>=0.115.0",
        "pydantic>=2.0.0",
        "pydantic-settings>=2.0.0",
        "numpy>=1.21.0",
        "uvicorn>=0.18.0",
    ],
    extras_require={
        "dev": [
            "pytest>=8.0.0",
            "httpx==0.27.0",
            "pytest-cov>=4.0.0",
        ],
        "ml": [
            "torch>=2.0.0",
            "qiskit>=0.39.0",
            "cirq>=1.0.0",
            "jax>=0.4.0",
            "PyWavelets>=1.4.0",
            "scikit-learn>=1.0.0",
        ],
        "metrics": [
            "prometheus-client>=0.16.0",
        ],
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
    ],
)
