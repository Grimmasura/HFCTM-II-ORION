import os
import urllib.request

from typing import Optional

from models.stability_core import StabilityCore

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
 

def recursive_model_live(
    query: str,
    depth: int,
    core: Optional[StabilityCore] = None,
    chi_Eg: int = 0,
    lambda_: float = 0.0,
):
    """Recursively generate a response for the given query.

    Each generation step is tracked by the provided :class:`StabilityCore`
    instance which can route to a refusal policy when detector outputs
    flag unsafe content.
    """

    core = core or StabilityCore()

    # Wrap the generation steps for telemetry and policy enforcement
    for _ in range(depth):
        if not core.track({"chi_Eg": chi_Eg, "lambda": lambda_}):
            return "Refusal: safety policy triggered"

    if agent is None:
        optimal_depth = depth
    else:
        optimal_depth = agent.predict(depth)[0]
    if optimal_depth <= 0:
        return f"Base case: {query}"
    response = f"Recursive Expansion of '{query}' at depth {optimal_depth}"
    return response + "\n" + recursive_model_live(
        query, optimal_depth - 1, core, chi_Eg, lambda_
    )

