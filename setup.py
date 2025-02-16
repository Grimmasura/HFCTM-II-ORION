from setuptools import setup, find_packages

setup(
    name="HFCTM-II",
    version="2.0.0",
    author="Your Name",
    author_email="your.email@example.com",
    description="HFCTM-II: Recursive AI with Blockchain Trust, Quantum Coherence, and E8 Lattice Processing.",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/HFCTM-II",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        "fastapi",
        "uvicorn",
        "numpy",
        "networkx",
        "requests",
        "websockets",
        "qiskit",
        "scipy",
        "pydantic",
        "pywavelets",
        "setuptools"
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
    entry_points={
        "console_scripts": [
            "hfctm-ii=src.HFCTM_II_API:app"
        ],
    },
)
