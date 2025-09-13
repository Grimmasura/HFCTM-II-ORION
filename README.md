# HFCTM-II-ORION

O.R.I.O.N. ∞ (Omniversal Recursive Intelligence for Ontological Navigation) is an advanced recursive AI framework optimized for multi-agent recursion, quantum stabilization, and self-improving inference.

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
    uvicorn orion_api.main:app --host $ORION_HOST --port $ORION_PORT
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

## Configuration
Runtime settings are managed via `orion_api/config.py`, which uses
Pydantic's `BaseSettings`.  Values can be supplied through environment
variables or a `.env` file.  The most common options are:

| Variable | Description | Default |
|----------|-------------|---------|
| `ORION_HOST` | Host interface for the API | `0.0.0.0` |
| `ORION_PORT` | Port the API listens on | `8080` |
| `ORION_MODEL_DIR` | Directory where models are stored | `models` |
| `ORION_RECURSIVE_MODEL_PATH` | Path to the recursive model file | `models/recursive_live_optimization_model.zip` |

Example `.env` file:

```env
ORION_HOST=127.0.0.1
ORION_PORT=8000
```

## Benchmarking
Run recursive AI performance tests:
```bash
python benchmarks/benchmark_recursive.py
```

### Enhanced ORION

Run with:
```bash
uvicorn orion_enhanced.orion_complete:create_complete_orion_app --reload --port 8080
```

Key endpoints:

* `GET /system/status`
* `POST /system/inference`
* `POST /system/test`
* `GET /system/health`

## Testing
Install minimal test dependencies and run the unit tests:
```bash
pip install fastapi==0.115.11 httpx==0.27.0 pytest==8.0.0 pydantic-settings==2.10.1
pytest -q
```
The pinned `httpx` version has been tested with FastAPI's `TestClient` to ensure
API routes load correctly.

## Dependency Notes
The repository previously listed Hugging Face's `transformers` library in
`requirements.txt`, but no modules actually used it. The dependency has been
removed to keep installation lightweight.

## Azure Quantum Integration
Running against real hardware requires Microsoft Azure Quantum credentials.  Set
the following environment variables to enable the `majorana1` backend used by
the `/quantum-sync/status` endpoint:

| Variable | Description |
|----------|-------------|
| `AZURE_QUANTUM_SUBSCRIPTION_ID` | Azure subscription identifier |
| `AZURE_QUANTUM_RESOURCE_GROUP` | Resource group containing the workspace |
| `AZURE_QUANTUM_WORKSPACE_NAME` | Azure Quantum workspace name |
| `AZURE_QUANTUM_LOCATION` | (Optional) workspace region, e.g. `eastus` |

Authentication relies on [Azure Identity's `DefaultAzureCredential`][dac].  In
most deployments this means providing the standard service principal secrets:
`AZURE_CLIENT_ID`, `AZURE_TENANT_ID`, and `AZURE_CLIENT_SECRET`.

[dac]: https://learn.microsoft.com/python/api/overview/azure/identity-readme

## Automation Scripts
Scripts such as `commit_file.py` can automatically commit changes. To push to a
remote repository these scripts require a `GITHUB_TOKEN` environment variable.
Without the token they will create the commit locally and skip the push step.

## Arch Linux (AUR-style)

```bash
sudo pacman -S --needed base-devel git
git clone https://github.com/Grimmasura/HFCTM-II-ORION.git
cd HFCTM-II-ORION
makepkg -si
```

See `README-arch.md` for systemd usage and optional dependencies.
