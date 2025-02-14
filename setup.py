from setuptools import setup, find_packages

setup(
    name="HFCTM_II",
    version="1.0",
    packages=find_packages(where="src"),  # Ensure it finds the packages inside src/
    package_dir={"": "src"},  # Defines "src/" as the root for package discovery
    install_requires=[
        "numpy",
        "scipy",
        "scikit-learn",
        "fastapi",
        "uvicorn",
        "pydantic"
    ],
    include_package_data=True,  # Includes non-Python files if any exist
    author="GrimmSeraph",
    author_email="Joshuahumphrey@duck.com",  # Replace with your actual email
    description="A deployable package for HFCTM-II adversarial resilience and recursive stability.",
    long_description=open("README.md").read(),  # Reads description from README.md
    long_description_content_type="text/markdown",
    url="https://github.com/Grimmasura/HFCTM-II_Egregore_Defense_AI_Security",  # Replace with actual URL
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",  # Ensures compatibility with Python 3.8+
)
