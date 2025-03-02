import os
import gym
from stable_baselines3 import PPO
from stable_baselines3.common.envs import DummyVecEnv

# ✅ Set Paths
MODEL_DIR = "models"
MODEL_PATH = os.path.join(MODEL_DIR, "recursive_live_optimization_model.zip")

# ✅ Define Custom Environment (If Needed)
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

# ✅ Ensure Model Directory Exists
if not os.path.exists(MODEL_DIR):
    os.makedirs(MODEL_DIR)

# ✅ Train a New Model If Missing
if not os.path.exists(MODEL_PATH):
    print("🚀 Model not found! Training a new one...")
    env = DummyVecEnv([lambda: RecursiveAIEnv()])
    model = PPO("MlpPolicy", env, verbose=1)
    model.learn(total_timesteps=5000)  # Adjust training steps if needed
    model.save(MODEL_PATH)
    print(f"✅ Model trained and saved at {MODEL_PATH}")

# ✅ Load Model
print("🔄 Loading model...")
agent = PPO.load(MODEL_PATH)
print("✅ Model Loaded Successfully!")
