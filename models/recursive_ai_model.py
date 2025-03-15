import os
import urllib.request
from stable_baselines3 import PPO
import gym  # Ensure you have the correct environment for training

# Define paths
MODEL_DIR = "models"
MODEL_PATH = os.path.join(MODEL_DIR, "recursive_live_optimization_model.zip")
MODEL_URL = "https://github.com/YOUR_GITHUB_USER/YOUR_REPO/releases/download/v1.0/recursive_live_optimization_model.zip"

# Ensure the models directory exists
os.makedirs(MODEL_DIR, exist_ok=True)

# If the model file doesn't exist, download or train
if not os.path.exists(MODEL_PATH):
    print("🔴 Model file missing! Downloading or training...")

    try:
        # Attempt to download from GitHub Releases
        print(f"⬇ Downloading model from {MODEL_URL}...")
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
        print("✅ Download complete.")

    except Exception as e:
        print(f"⚠ Download failed: {e}\n🚀 Training a new model instead...")

        # Train a new model
        env = gym.make("CartPole-v1")  # Replace with your actual environment
        model = PPO("MlpPolicy", env, verbose=1)
        model.learn(total_timesteps=10000)  # Adjust as needed
        model.save(MODEL_PATH)

        print(f"✅ Model training complete. Saved to: {MODEL_PATH}")

# Load the trained or downloaded model
print("📥 Loading model...")
agent = PPO.load(MODEL_PATH)
print("✅ Model loaded successfully.")

