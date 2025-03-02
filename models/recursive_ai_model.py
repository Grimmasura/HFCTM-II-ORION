import os
from stable_baselines3 import PPO
from stable_baselines3.common.envs import DummyVecEnv
import gym

# Define Environment
class RecursiveAIEnv(gym.Env):
    def __init__(self):
        super(RecursiveAIEnv, self).__init__()
        self.action_space = gym.spaces.Discrete(3)  # Example actions
        self.observation_space = gym.spaces.Box(low=0, high=1, shape=(5,), dtype=float)

    def reset(self):
        return self.observation_space.sample()

    def step(self, action):
        reward = 1 if action == 1 else 0
        done = False
        return self.observation_space.sample(), reward, done, {}

# Set Model Path
MODEL_DIR = "models"
MODEL_PATH = os.path.join(MODEL_DIR, "recursive_live_optimization_model.zip")

# Ensure Model Directory Exists
if not os.path.exists(MODEL_DIR):
    os.makedirs(MODEL_DIR)

# Check if Model Exists or Needs Training
if not os.path.exists(MODEL_PATH):
    print("🚀 Model not found! Training new one...")
    env = DummyVecEnv([lambda: RecursiveAIEnv()])
    model = PPO("MlpPolicy", env, verbose=1)
    model.learn(total_timesteps=10000)
    model.save(MODEL_PATH)
    print(f"✅ Model trained and saved at {MODEL_PATH}")

# Load Model
print("🔄 Loading Model...")
agent = PPO.load(MODEL_PATH)
print("✅ Model Loaded Successfully")
