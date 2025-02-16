from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import networkx as nx
import numpy as np
import hashlib
import time
import uvicorn

app = FastAPI(
    title="HFCTM-II Recursive AI API",
    description="Blockchain-Validated AI with Recursive Intelligence & Self-Correcting Nodes",
    version="1.2"
)

# -------------------------------
# Blockchain-Validated AI Trust System
# -------------------------------
class Blockchain:
    """ Simulated Blockchain for recursive AI trust validation. """
    def __init__(self):
        self.chain = []
        self.pending_transactions = []
        self.create_block(previous_hash="1", proof=100)  # Genesis block

    def create_block(self, proof, previous_hash):
        """ Create a new block in the blockchain. """
        block = {
            'index': len(self.chain) + 1,
            'timestamp': time.time(),
            'transactions': self.pending_transactions,
            'proof': proof,
            'previous_hash': previous_hash,
        }
        self.pending_transactions = []
        self.chain.append(block)
        return block

    def add_transaction(self, sender, recipient, trust_score):
        """ Add AI trust node validation to the blockchain. """
        self.pending_transactions.append({
            'sender': sender,
            'recipient': recipient,
            'trust_score': float(trust_score)  # Ensure JSON serialization
        })
        return self.last_block['index'] + 1

    def proof_of_work(self, previous_proof):
        """ Proof of Recursion (PoR) consensus mechanism. """
        proof = 0
        while not self.valid_proof(previous_proof, proof):
            proof += 1
        return proof

    def valid_proof(self, previous_proof, proof):
        """ Validate proof for trust integrity. """
        guess = f"{previous_proof}{proof}".encode()
        guess_hash = hashlib.sha256(guess).hexdigest()
        return guess_hash[:4] == "0000"

    @property
    def last_block(self):
        return self.chain[-1]

# -------------------------------
# Recursive AI Model
# -------------------------------
class RecursiveAINetwork:
    """ Simulated Recursive AI intelligence with blockchain trust verification. """

    def __init__(self, num_nodes=100, trust_threshold=0.6):
        self.num_nodes = num_nodes
        self.trust_threshold = trust_threshold
        self.graph = nx.erdos_renyi_graph(num_nodes, 0.2, directed=True)
        self.blockchain = Blockchain()
        self.node_properties = {}

        # Initialize nodes with random trust scores and decoherence levels
        for node in self.graph.nodes():
            trust_score = np.random.uniform(0, 1)
            decoherence_level = np.random.uniform(0, 1)

            # Self-correcting mechanism: Nodes with decoherence > trust threshold attempt stabilization
            if decoherence_level > trust_threshold:
                corrected = np.random.choice([True, False], p=[0.7, 0.3])
                trust_score = trust_score * 1.1 if corrected else trust_score * 0.9
            else:
                corrected = True  

            self.node_properties[node] = {
                "Trust Score": float(trust_score),  # Convert NumPy float to Python float
                "Decoherence Level": float(decoherence_level),  # Convert NumPy float
                "Corrected": bool(corrected),  # Convert NumPy bool to Python bool
            }

            # Add blockchain validation if trust is above threshold
            if trust_score >= trust_threshold:
                self.blockchain.add_transaction("AI_Node", f"Node_{node}", trust_score)

        # Mine the blockchain after transactions
        self.blockchain.create_block(self.blockchain.proof_of_work(self.blockchain.last_block['proof']),
                                     previous_hash=self.blockchain.last_block['previous_hash'])

    def run_simulation(self, steps=10):
        """ Simulate recursive AI intelligence decision autonomy with blockchain validation. """
        results = []
        for step in range(steps):
            step_results = {"step": step + 1, "nodes": []}
            for node in self.graph.nodes():
                trust_score = self.node_properties[node]["Trust Score"]
                decoherence_level = self.node_properties[node]["Decoherence Level"]

                # Trust verification and correction
                if trust_score >= self.trust_threshold:
                    step_results["nodes"].append({"node": node, "status": "Trust Verified", "trust_score": float(trust_score)})
                else:
                    # Attempt self-correction using intrinsic field inference
                    self.node_properties[node]["Trust Score"] = min(1.0, trust_score + np.random.uniform(0.05, 0.15))
                    self.node_properties[node]["Decoherence Level"] = max(0.0, decoherence_level - np.random.uniform(0.05, 0.15))

                    if self.node_properties[node]["Trust Score"] >= self.trust_threshold:
                        step_results["nodes"].append({"node": node, "status": "Self-Correction Successful", "trust_score": float(self.node_properties[node]["Trust Score"])})
                        self.blockchain.add_transaction("AI_Node", f"Node_{node}", self.node_properties[node]["Trust Score"])
                    else:
                        step_results["nodes"].append({"node": node, "status": "Self-Correction Failed", "trust_score": float(self.node_properties[node]["Trust Score"])})

            results.append(step_results)

            # Mine the blockchain every few steps
            if step % 3 == 0:
                self.blockchain.create_block(self.blockchain.proof_of_work(self.blockchain.last_block['proof']),
                                             previous_hash=self.blockchain.last_block['previous_hash'])

        return results

# -------------------------------
# API Endpoints
# -------------------------------
ai_network = RecursiveAINetwork(num_nodes=100)

class SimulationRequest(BaseModel):
    steps: int = 10

@app.get("/")
def home():
    return {"message": "Welcome to the HFCTM-II Recursive AI API"}

@app.post("/run-simulation/")
def run_simulation(request: SimulationRequest):
    """ Run a recursive AI trust simulation. """
    if request.steps < 1:
        raise HTTPException(status_code=400, detail="Number of steps must be greater than 0.")

    result = ai_network.run_simulation(steps=request.steps)
    return {"status": "Simulation Completed", "data": result}

@app.get("/blockchain/")
def get_blockchain():
    """ Retrieve the AI blockchain ledger. """
    return {"status": "Blockchain Retrieved", "chain": ai_network.blockchain.chain}

@app.get("/nodes/")
def get_nodes():
    """ Retrieve AI node trust properties. """
    return {"status": "Node Data Retrieved", "nodes": ai_network.node_properties}

# -------------------------------
# Running the API
# -------------------------------
if __name__ == "__main__":
    print("🚀 HFCTM-II API is now running at http://127.0.0.1:8000/")
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)
