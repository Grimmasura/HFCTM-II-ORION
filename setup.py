from setuptools import setup, find_packages

setup(
    name="hfctm-ii-api",
    version="1.0.0",
    description="HFCTM-II Recursive AI API with Blockchain-Validated Intelligence & Self-Correcting AI Governance",
    author="Your Name",
    author_email="your.email@example.com",
    url="https://github.com/yourusername/hfctm-ii-api",
    packages=find_packages(),
    install_requires=[
        "fastapi",           # API Framework
        "uvicorn",           # ASGI Server
        "numpy",             # Scientific Computing
        "networkx",          # Graph-Based AI Networks
        "cryptography",      # Blockchain Encryption & Security
        "pydantic",          # Data Validation & Serialization
        "python-dotenv",     # Environment Variable Management
        "pyjwt",             # JSON Web Tokens for AI Trust Verification
        "websockets",        # Real-time AI Communication
        "requests",          # External API Calls
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
    entry_points={
        "console_scripts": [
            "hfctm-api=main:app"
        ],
    },
)
