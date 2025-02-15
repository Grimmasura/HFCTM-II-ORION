from fastapi import FastAPI
import numpy as np
from hfctm_ii import HFCTMII

app = FastAPI()
hfctm = HFCTMII()

@app.get("/")
def read_root():
    return {"message": "HFCTM-II API is running with Recursive Ethics and Stability Monitoring!"}

@app.post("/predict/")
def predict_adversarial_attack(sequence: list):
    """Predict if an adversarial attack is occurring."""
    prediction = hfctm.predict_adversarial_attack(sequence)
    return {"adversarial_attack": bool(prediction)}

@app.post("/stabilize/")
def stabilize_knowledge_state(state: float, attack_predicted: bool):
    """Apply recursive stabilization based on predicted adversarial attack."""
    stabilized_state = hfctm.apply_recursive_stabilization(state, attack_predicted)
    return {"stabilized_state": stabilized_state}

@app.post("/wavelet_analysis/")
def wavelet_analysis(sequence: list):
    """Perform wavelet anomaly detection on a sequence of knowledge states."""
    anomaly_matrix = hfctm.wavelet_anomaly_detection(np.array(sequence)).tolist()
    return {"wavelet_transform": anomaly_matrix}

@app.post("/blockchain_verify/")
def blockchain_verify_state(knowledge_state: float):
    """Verify AI knowledge integrity using blockchain-based cryptographic validation."""
    hash_value = hfctm.blockchain_validate(knowledge_state)
    return {"blockchain_hash": hash_value}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
