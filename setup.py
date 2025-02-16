
import os
import subprocess
import sys

### 📌 STEP 1: CREATE A VIRTUAL ENVIRONMENT (OPTIONAL, BUT RECOMMENDED)
venv_dir = "venv"
if not os.path.exists(venv_dir):
    print("🔧 Creating virtual environment...")
    subprocess.run([sys.executable, "-m", "venv", venv_dir], check=True)

# Activate virtual environment (Windows)
activate_script = os.path.join(venv_dir, "Scripts", "activate")
if sys.platform == "win32":
    activate_command = f"{activate_script}.bat"
else:
    activate_command = f"source {activate_script}"

print("🔄 Activating virtual environment...")
subprocess.run(activate_command, shell=True, check=True)

### 📌 STEP 2: INSTALL REQUIRED DEPENDENCIES
dependencies = [
    "fastapi",
    "uvicorn",
    "PyWavelets",  # For wavelet-based egregore detection
    "numpy",
    "scipy",
    "pydantic"
]

print("📦 Installing dependencies...")
subprocess.run([sys.executable, "-m", "pip", "install", "--upgrade", "pip"], check=True)
subprocess.run([sys.executable, "-m", "pip", "install"] + dependencies, check=True)

### 📌 STEP 3: CONFIRM INSTALLATION
try:
    import fastapi
    import uvicorn
    import pywt
    import numpy as np
    import scipy.linalg as la
    import pydantic

    print("✅ All dependencies installed successfully!")
except ImportError as e:
    print(f"❌ ERROR: {e}")
    sys.exit(1)

### 📌 STEP 4: START FASTAPI SERVER AUTOMATICALLY
print("🚀 Starting HFCTM-II API...")
subprocess.run(["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"])
