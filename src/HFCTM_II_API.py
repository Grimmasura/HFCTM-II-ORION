from fastapi import FastAPI
import numpy as np
from src.hfctm_ii import HFCTMII
from pydantic import BaseModel
from typing import List

# Initialize API & Model
app = FastAPI()
model = HFCTMII()

# 🚀 Train the model with sample data
X_train = np.random.rand(100, 10)  # 100 samples, 10 features each
y_train = np.random.randint(0, 2, 100)  # Binary labels (0 or 1)
model.train_adversarial_detector(X_train, y_train)

# Define Input Schema
class AttackPredictionRequest(BaseModel):
    sequence: List[float]

class KnowledgeStateRequest(BaseModel):
    knowledge_state: float
    attack_pred: int

@app.get("/")
def root():
    return {"message": "HFCTM-II API is running!"}

@app.post("/predict-attack/")
def predict_attack(data: AttackPredictionRequest):
    prediction = model.predict_adversarial_attack(data.sequence)
    return {"attack_prediction": prediction}

@app.post("/chiral-inversion/")
def chiral_inversion(data: KnowledgeStateRequest):
    inverted = model.apply_chiral_inversion(data.knowledge_state)
    return {"inverted_knowledge_state": inverted}

@app.post("/recursive-stabilization/")
def recursive_stabilization(data: KnowledgeStateRequest):
    stabilized = model.apply_recursive_stabilization(data.knowledge_state, data.attack_pred)
    return {"stabilized_knowledge_state": stabilized}

