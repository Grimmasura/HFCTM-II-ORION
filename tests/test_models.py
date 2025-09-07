from models.recursive_ai_model import recursive_model_live
from orion_api.config import settings
import random

def test_recursive_model_base_case():
    result = recursive_model_live("test", 0)
    assert "Base case" in result


def test_deterministic_generation_seed():
    settings.max_tokens = 5
    settings.temperature = 0.7
    random.seed(1234)
    first = recursive_model_live("hello", 0)
    random.seed(1234)
    second = recursive_model_live("hello", 0)
    assert first == second
