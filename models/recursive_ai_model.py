import os
import urllib.request
import numpy as np
from models.stability_core import stability_core
from orion_api.config import settings

from transformers import AutoModelForCausalLM, AutoTokenizer

try:
    import torch_xla.core.xla_model as xm  # type: ignore
    _xla_available = True
except Exception:  # pragma: no cover - optional dependency
    _xla_available = False

try:
    import jax
    _jax_available = True
except Exception:  # pragma: no cover - optional dependency
    _jax_available = False

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


# ---------------------------------------------------------------------------
# Hugging Face model setup
# ---------------------------------------------------------------------------

HF_MODEL_NAME = os.getenv("RECURSIVE_HF_MODEL", "sshleifer/tiny-gpt2")
USING_FLAX = False

if _xla_available:
    device = xm.xla_device()
    tokenizer = AutoTokenizer.from_pretrained(HF_MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(HF_MODEL_NAME).to(device)
elif _jax_available:
    from transformers import FlaxAutoModelForCausalLM
    from jax.experimental import pjit

    USING_FLAX = True
    tokenizer = AutoTokenizer.from_pretrained(HF_MODEL_NAME)
    model = FlaxAutoModelForCausalLM.from_pretrained(HF_MODEL_NAME)
    tpu = jax.devices("tpu")[0] if jax.devices("tpu") else jax.devices()[0]
    device = tpu

    @pjit
    def _to_device(params):
        return jax.device_put(params, tpu)

    model.params = _to_device(model.params)
else:
    import torch

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(HF_MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(HF_MODEL_NAME).to(device)


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

    # Generate a response using the loaded language model
    if USING_FLAX:
        import jax

        key = jax.random.PRNGKey(0)
        input_ids = tokenizer(processed_query, return_tensors="jax")["input_ids"]
        sequences = model.generate(
            input_ids=input_ids,
            max_new_tokens=settings.max_tokens,
            do_sample=settings.temperature > 0,
            temperature=settings.temperature,
            prng_key=key,
            params=model.params,
        ).sequences
        generated_text = tokenizer.decode(sequences[0], skip_special_tokens=True)
    else:
        import torch

        inputs = tokenizer(processed_query, return_tensors="pt").to(device)
        with torch.no_grad():
            sequences = model.generate(
                **inputs,
                max_new_tokens=settings.max_tokens,
                do_sample=settings.temperature > 0,
                temperature=settings.temperature,
            )
        generated_text = tokenizer.decode(sequences[0], skip_special_tokens=True)

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
        return f"Base case: {generated_text}"
    response = f"Recursive Expansion of '{generated_text}' at depth {optimal_depth}"
    return response + "\n" + recursive_model_live(processed_query, optimal_depth - 1)

