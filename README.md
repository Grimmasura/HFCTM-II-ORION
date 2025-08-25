# HFCTM-II-ORION

O.R.I.O.N. ∞ (Omniversal Recursive Intelligence and Ontological Network) is an advanced recursive AI framework optimized for multi-agent recursion, quantum stabilization, and self-improving inference.

## Features
- **Recursive AI API** with reinforcement learning for adaptive recursion depth.
- **Multi-Agent Task Distribution** for distributed recursive inference.
- **Quantum Recursive Synchronization** for stability enforcement.
- **Egregore Defense System** for adversarial resilience.
- **Live Monitoring with Prometheus & Grafana**.

## Setup & Deployment
1. Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```
2. Run the API:
    ```bash
    uvicorn orion_api.main:app --host 0.0.0.0 --port 8080
    ```
3. Deploy with Docker:
    ```bash
    docker build -t orion-api .
    docker run -p 8080:8080 orion-api
    ```
4. Deploy on Kubernetes:
    ```bash
    kubectl apply -f deployment/orion-deployment.yml
    ```

## Benchmarking
Run recursive AI performance tests:
```bash
python benchmarks/benchmark_recursive.py
```

## Validation Experiments
Run post-training validation experiments and build a KPI decision matrix:
```bash
python -m validation.run_validation --dataset S1=TruthfulQA --dataset default=HumanEval --output validation/results/decision_matrix.csv
```

## Testing
Install minimal test dependencies and run the unit tests:
```bash
pip install fastapi==0.115.11 httpx==0.27.0 pytest==8.0.0
pytest -q
```
The pinned `httpx` version has been tested with FastAPI's `TestClient` to ensure
API routes load correctly.

## Dependency Notes
The repository previously listed Hugging Face's `transformers` library in
`requirements.txt`, but no modules actually used it. The dependency has been
removed to keep installation lightweight.

## Automation Scripts
Scripts such as `commit_file.py` can automatically commit changes. To push to a
remote repository these scripts require a `GITHUB_TOKEN` environment variable.
Without the token they will create the commit locally and skip the push step.
