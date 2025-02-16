from fastapi import FastAPI
import numpy as np
import scipy.linalg as la
import pywt
from pydantic import BaseModel

app = FastAPI()

### Request & Response Models
class InferenceRequest(BaseModel):
    iterations: int = 1

class InferenceResponse(BaseModel):
    knowledge_state: list
    trust_matrix: list

### HFCTM-II Core Class
class HFCTM_II:
    def __init__(self, dim=8):
        self.dim = dim
        self.state = self.initialize_E8_seed()
        self.R = self.generate_recursive_inference_matrix()
        self.trust_embeddings = self.initialize_trust_network()
        self.egregore_threshold = 0.1
        self.lyapunov_threshold = 0.05

    def initialize_E8_seed(self):
        E8_vectors = np.array([
            [1, -1, 0, 0, 0, 0, 0, 0],
            [0, 1, -1, 0, 0, 0, 0, 0],
            [0, 0, 1, -1, 0, 0, 0, 0],
            [0, 0, 0, 1, -1, 0, 0, 0],
            [0, 0, 0, 0, 1, -1, 0, 0],
            [0, 0, 0, 0, 0, 1, -1, 0],
            [0, 0, 0, 0, 0, 0, 1, -1],
            [-1, 0, 0, 0, 0, 0, 0, 1]
        ])
        seed_state = np.sum(E8_vectors, axis=0).reshape(self.dim, 1)
        return seed_state / np.linalg.norm(seed_state)

    def non_local_field_inference(self):
        kernel = np.exp(-0.1 * np.abs(np.subtract.outer(self.state, self.state)))
        self.state = np.dot(kernel, self.state)

    def generate_recursive_inference_matrix(self):
        R = np.random.randn(self.dim, self.dim) * 0.1
        R = R - R.T
        eigvals = la.eigvals(R)
        if np.max(np.abs(eigvals)) > 1:
            R /= np.max(np.abs(eigvals))
        return R

    def recursive_evolution(self):
        next_state = np.dot(self.R, self.state)
        self.non_local_field_inference()
        self.state = next_state

    def wavelet_based_egregore_detection(self):
        coeffs, _ = pywt.cwt(self.state.flatten(), scales=np.arange(1, 10), wavelet='gaus1')
        anomaly_score = np.max(np.abs(coeffs))
        if anomaly_score > self.egregore_threshold:
            self.R *= 0.9

    def enforce_lyapunov_stability(self):
        eigvals = la.eigvals(self.R)
        if np.max(np.abs(eigvals)) > self.lyapunov_threshold:
            self.R /= np.max(np.abs(eigvals))

    def initialize_trust_network(self):
        trust_base = np.random.uniform(0.9, 1.0, (self.dim, self.dim))
        return np.triu(trust_base) + np.triu(trust_base, 1).T

    def update_trust_embeddings(self):
        similarity = np.dot(self.state.T, self.state)
        self.trust_embeddings = np.exp(-0.1 * np.abs(self.trust_embeddings - similarity))

    def inference_cycle(self, iterations=1):
        for _ in range(iterations):
            self.recursive_evolution()
            self.wavelet_based_egregore_detection()
            self.enforce_lyapunov_stability()
            self.update_trust_embeddings()

        return {
            "knowledge_state": self.state.flatten().tolist(),
            "trust_matrix": self.trust_embeddings.tolist(),
        }

### API Endpoints
hfctm = HFCTM_II()

@app.post("/inference", response_model=InferenceResponse)
def run_inference(request: InferenceRequest):
    result = hfctm.inference_cycle(request.iterations)
    return InferenceResponse(
        knowledge_state=result["knowledge_state"],
        trust_matrix=result["trust_matrix"]
    )

@app.get("/state")
def get_knowledge_state():
    return {"knowledge_state": hfctm.state.flatten().tolist()}

@app.get("/trust")
def get_trust_matrix():
    return {"trust_matrix": hfctm.trust_embeddings.tolist()}

@app.get("/egregore-detection")
def detect_egregore():
    coeffs, _ = pywt.cwt(hfctm.state.flatten(), scales=np.arange(1, 10), wavelet='gaus1')
    anomaly_score = np.max(np.abs(coeffs))
    return {"anomaly_score": anomaly_score, "egregore_detected": anomaly_score > hfctm.egregore_threshold}

@app.post("/chiral-inversion")
def apply_chiral_inversion():
    hfctm.R = -hfctm.R
    return {"status": "Chiral inversion applied", "new_inference_matrix": hfctm.R.tolist()}

@app.post("/reset")
def reset_model():
    hfctm.__init__()
    return {"status": "HFCTM-II reset successfully", "knowledge_state": hfctm.state.flatten().tolist()}
