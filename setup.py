MIT License

Copyright (c) 2025 GrimmSeraph

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

1. **Model & API Coverage**  
   This license applies to both the **HFCTM-II Model** and the **HFCTM-II API**, including but not limited to:
   - **Recursive inference engine**
   - **Fractal trust and non-local field inference**
   - **Egregore suppression mechanisms**
   - **FastAPI-based API implementation**
   - **Chiral inversion and Lyapunov stabilization**

2. **Attribution Requirement**  
   Any modified versions, forks, or distributions of this software **must retain** this license and include the original author's credit:  


3. **Warranty Disclaimer**  
THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.


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
