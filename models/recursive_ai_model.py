import os
import urllib.request

try:
    from stable_baselines3 import PPO
    import gym
except ImportError:  # Fallback when dependencies are missing
    PPO = None
    gym = None

MODEL_DIR = "models"
MODEL_PATH = os.path.join(MODEL_DIR, "recursive_live_optimization_model.zip")
MODEL_URL = (
    "https://github.com/YOUR_GITHUB_USER/YOUR_REPO/releases/download/v1.0/recursive_live_optimization_model.zip"
)

os.makedirs(MODEL_DIR, exist_ok=True)

agent = None
if PPO is not None:
    if not os.path.exists(MODEL_PATH):
        print("🔴 Model file missing! Downloading or training...")
        try:
            print(f"⬇ Downloading model from {MODEL_URL}...")
            urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
            print("✅ Download complete.")
        except Exception as e:
            print(f"⚠ Download failed: {e}\n🚀 Training a new model instead...")
            env = gym.make("CartPole-v1")
            model = PPO("MlpPolicy", env, verbose=1)
            model.learn(total_timesteps=10000)
            model.save(MODEL_PATH)
            print(f"✅ Model training complete. Saved to: {MODEL_PATH}")
    print("📥 Loading model...")
    agent = PPO.load(MODEL_PATH)
    print("✅ Model loaded successfully.")


def recursive_model_live(query: str, depth: int):
    """Recursively generate a response for the given query."""
    if agent is None:
        optimal_depth = depth
    else:
        optimal_depth = agent.predict(depth)[0]
    if optimal_depth <= 0:
        return f"Base case: {query}"
    response = f"Recursive Expansion of '{query}' at depth {optimal_depth}"
    return response + "\n" + recursive_model_live(query, optimal_depth - 1)

