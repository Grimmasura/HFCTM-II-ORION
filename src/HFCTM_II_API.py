from fastapi import FastAPI
from pydantic import BaseModel
from typing import List

app = FastAPI()

# Define Request Model
class PredictionInput(BaseModel):
    sequence: List[float]

@app.post("/predict/")
def predict_adversarial_attack(data: PredictionInput):
    """Predict if an adversarial attack is occurring."""
    prediction = sum(data.sequence) > 0  # Dummy logic, replace with real model
    return {"adversarial_attack": bool(prediction)}
