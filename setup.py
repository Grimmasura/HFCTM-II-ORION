from setuptools import setup, find_packages

setup(
    name="HFCTM_II",
    version="1.0",
    packages=find_packages(),
    install_requires=[
        "numpy",
        "scipy",
        "scikit-learn"
    ],
    author="HFCTM-GPT",
    description="A deployable package for HFCTM-II adversarial resilience and recursive stability."
)
