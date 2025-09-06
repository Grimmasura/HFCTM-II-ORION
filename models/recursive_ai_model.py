import os
import urllib.request
import numpy as np
from models.stability_core import stability_core
from orion_api.config import settings

try:
    from stable_baselines3 import PPO
    import gym
except ImportError:  # Fallback when dependencies are missing
    PPO = None
    gym = None

MODEL_DIR = settings.model_dir
MODEL_PATH = settings.recursive_model_path
MODEL_URL = os.getenv(
    "RECURSIVE_MODEL_URL",
    "https://github.com/HFCTM-II-ORION/HFCTM-II-ORION/releases/latest/download/recursive_live_optimization_model.zip",
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
    """Recursively generate a response for the given query.

    The query is first streamed through ``StabilityCore`` which yields
    generation steps. Each step is inspected for detector outputs and the
    response is routed to a refusal policy when unsafe signals are present.
    """

    inference = stability_core.generate(query)
    processed_tokens = []
    for step in inference:
        if step.get("chi_Eg") == 1 or step.get("lambda", 0) > 0:
            return "Response refused by safety policy."
        processed_tokens.append(step["token"])

    processed_query = " ".join(processed_tokens)

    if agent is None:
        optimal_depth = depth
    else:
        try:
            obs_space = getattr(agent, "observation_space", None)
            if obs_space is None or obs_space.shape is None:
                raise ValueError("Agent lacks observation space definition")
            observation = np.full(obs_space.shape, depth, dtype=np.float32)
            if observation.shape != obs_space.shape:
                raise ValueError(
                    f"Expected observation shape {obs_space.shape}, got {observation.shape}"
                )
            optimal_depth = agent.predict(observation)[0]
        except Exception as e:
            print(f"⚠ Prediction skipped due to invalid observation format: {e}")
            optimal_depth = depth
    if optimal_depth <= 0:
        return f"Base case: {processed_query}"
    response = f"Recursive Expansion of '{processed_query}' at depth {optimal_depth}"
    return response + "\n" + recursive_model_live(processed_query, optimal_depth - 1)

