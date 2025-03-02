import os
from stable_baselines3 import PPO

MODEL_PATH = "models/recursive_live_optimization_model.zip"

# Ensure the models directory exists
if not os.path.exists("models"):
    os.makedirs("models")

# Check if the model exists, if not, attempt to download it
if not os.path.exists(MODEL_PATH):
    print(f"Model file '{MODEL_PATH}' not found. Downloading...")
    os.system(f"curl -L -o {MODEL_PATH} https://your-storage-url/model.zip")

# Load the model
try:
    agent = PPO.load(MODEL_PATH)
    print("Model loaded successfully.")
except Exception as e:
    raise RuntimeError(f"Error loading model: {e}")
