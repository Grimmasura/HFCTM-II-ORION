import os
import gym
import requests
from stable_baselines3 import PPO
from stable_baselines3.common.envs import DummyVecEnv

# ✅ Define Paths
MODEL_DIR = "models"
MODEL_NAME = "recursive_live_optimization_model.zip"
MODEL_PATH = os.path.join(MODEL_DIR, MODEL_NAME)
MODEL_URL = "https://your-valid-storage-url.com/recursive_live_optimization_model.zip"  # 🔄 FIX URL

# ✅ Create Model Directory If Missing
if not os.path.exists(MODEL_DIR):
    os.makedirs(MODEL_DIR)

# ✅ Download Model If Missing
if not os.path.exists(MODEL_PATH):
    print(f"🚀 Model {MODEL_NAME} missing! Attempting to download from {MODEL_URL}...")
    try:
        response = requests.get(MODEL_URL, stream=True)
        if response.status_code == 200:
            with open(MODEL_PATH, "wb") as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            print(f"✅ Model downloaded successfully: {MODEL_PATH}")
        else:
            print(f"⚠️ Failed to download model! Training a new one instead...")
    except Exception as e:
        print(f"🔥 ERROR: Could not download model: {e}. Training a new one...")

# ✅ Train New Model If No Valid File
if not os.path.exists(MODEL_PATH):
    print("🚀 Training new recursive AI model...")
    class RecursiveAIEnv(gym.Env):
        def __init__(self):
            super(RecursiveAIEnv, self).__init__()
            self.action_space = gym.spaces.Discrete(3)
            self.observation_space = gym.spaces.Box(low=0, high=1, shape=(5,), dtype=float)

        def reset(self):
            return self.observation_space.sample()

        def step(self, action):
            reward = 1 if action == 1 else 0
            done = False
            return self.observation_space.sample(), reward, done, {}

    env = DummyVecEnv([lambda: RecursiveAIEnv()])
    model = PPO("MlpPolicy", env, verbose=1)
    model.learn(total_timesteps=5000)
    model.save(MODEL_PATH)
    print(f"✅ New model trained and saved at {MODEL_PATH}")

# ✅ Load Model
print("🔄 Loading model...")
agent = PPO.load(MODEL_PATH)
print("✅ Model Loaded Successfully!")
