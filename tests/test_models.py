from models.recursive_ai_model import recursive_model_live

def test_recursive_model_base_case():
    result = recursive_model_live("test", 0)
    assert "Base case" in result
