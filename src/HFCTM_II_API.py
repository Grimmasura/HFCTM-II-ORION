from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
import uvicorn

app = FastAPI()

# Define Request Model
class PredictionInput(BaseModel):
    sequence: List[float]

@app.get("/")
def root():
    return {"message": "HFCTM-II API is running!"}

@app.post("/predict/")
def predict_adversarial_attack(data: PredictionInput):
    """Predict if an adversarial attack is occurring based on the input sequence."""
    # Replace this logic with your real prediction function
    prediction = sum(data.sequence) > 0  # Dummy logic, replace with real model
    return {"adversarial_attack": bool(prediction)}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)
