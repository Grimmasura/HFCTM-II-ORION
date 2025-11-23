# MIH-IIE v2.0 API Quick Start Guide

## Overview

The MIH-IIE v2.0 API exposes all core v2.0 components through RESTful endpoints. This guide shows you how to use the new v2.0 features.

## Starting the API

```bash
# Install dependencies
pip install -r requirements.txt

# Start server
uvicorn orion_api.main:app --reload --host 0.0.0.0 --port 8080
```

## API Documentation

Interactive docs available at:
- **Swagger UI**: http://localhost:8080/docs
- **ReDoc**: http://localhost:8080/redoc

## Core v2.0 Endpoints

### 1. E8 Topology Operations

**Base URL**: `/api/v2/e8`

#### Initialize E8 System

```bash
curl -X POST "http://localhost:8080/api/v2/e8/initialize" \
  -H "Content-Type: application/json" \
  -d '{"n_nodes": 240}'
```

Response:
```json
{
  "status": "initialized",
  "num_roots": 240,
  "num_active_nodes": 240,
  "structure_valid": true,
  "coordination_active": true
}
```

#### Get E8 Status

```bash
curl "http://localhost:8080/api/v2/e8/status"
```

#### Schedule Parallel Operations

```bash
curl -X POST "http://localhost:8080/api/v2/e8/operations/schedule" \
  -H "Content-Type: application/json" \
  -d '{
    "operations": [[0, 1], [2, 3], [4, 5], [1, 2]]
  }'
```

Response shows operation groups that can run in parallel:
```json
{
  "num_groups": 2,
  "groups": [[[0, 1], [2, 3], [4, 5]], [[1, 2]]],
  "total_operations": 4
}
```

#### Get Network Statistics

```bash
curl "http://localhost:8080/api/v2/e8/network/stats"
```

#### Get Synchronization Status

```bash
curl "http://localhost:8080/api/v2/e8/coordination/sync"
```

### 2. Majorana 0D Seed Operations

**Base URL**: `/api/v2/majorana`

#### Initialize Majorana Network

```bash
curl -X POST "http://localhost:8080/api/v2/majorana/initialize" \
  -H "Content-Type: application/json" \
  -d '{
    "n_seeds": 240,
    "use_azure": false
  }'
```

Response:
```json
{
  "status": "initialized",
  "num_seeds": 240,
  "backend": "simulation",
  "0d_properties_verified": true,
  "substrate_independent": true,
  "target_coherence_time": 1000.0
}
```

#### Get Majorana Status

```bash
curl "http://localhost:8080/api/v2/majorana/status"
```

#### Measure a Seed

```bash
curl -X POST "http://localhost:8080/api/v2/majorana/measure" \
  -H "Content-Type: application/json" \
  -d '{
    "seed_index": 0,
    "basis": "computational"
  }'
```

Response:
```json
{
  "seed_index": 0,
  "measurement": 1,
  "basis": "computational"
}
```

#### Apply Hadamard Gate

```bash
curl -X POST "http://localhost:8080/api/v2/majorana/seeds/0/hadamard"
```

#### Get Seed Information

```bash
curl "http://localhost:8080/api/v2/majorana/seeds/0"
```

#### Verify 0D Properties

```bash
curl "http://localhost:8080/api/v2/majorana/verification"
```

### 3. Frame-Invariant Validation

**Base URL**: `/api/v2/frame-invariant`

#### Initialize EDS

```bash
curl -X POST "http://localhost:8080/api/v2/frame-invariant/initialize"
```

#### Validate Proposition

```bash
curl -X POST "http://localhost:8080/api/v2/frame-invariant/validate" \
  -H "Content-Type: application/json" \
  -d '{
    "proposition": "test proposition for validation"
  }'
```

Response:
```json
{
  "classification": "PARTIAL_TRUTH",
  "convergence_score": 0.78,
  "num_frames_evaluated": 5,
  "is_paradigm_shift": false,
  "is_corruption": false,
  "frame_results": [
    {
      "frame": "quantum",
      "confidence": 0.85,
      "evidence": {"measurement_count": 100}
    },
    ...
  ]
}
```

**Classifications**:
- `FRAME_INVARIANT_TRUTH`: convergence > 0.95 (truth across all frames)
- `PARTIAL_TRUTH`: 0.3 ≤ convergence ≤ 0.95
- `FRAME_DEPENDENT_ARTIFACT`: convergence < 0.3

#### Detect Corruption

```bash
curl -X POST "http://localhost:8080/api/v2/frame-invariant/corruption/detect" \
  -H "Content-Type: application/json" \
  -d '{
    "semantic_state": "current system state"
  }'
```

Response:
```json
{
  "status": "STABLE",
  "convergence": 0.85,
  "baseline": 0.82,
  "action_required": false
}
```

**Status values**:
- `BASELINE_ESTABLISHED`: First measurement
- `STABLE`: Convergence within normal range
- `CORRUPTION_DETECTED`: Frame divergence detected
- `IMPROVED_ALIGNMENT`: Convergence increased

#### Assess Paradigm Shift

```bash
curl -X POST "http://localhost:8080/api/v2/frame-invariant/paradigm-shift/assess" \
  -H "Content-Type: application/json" \
  -d '{
    "semantic_change": "description of change",
    "old_state": "previous state",
    "new_state": "new state"
  }'
```

Response:
```json
{
  "is_valid_paradigm_shift": true,
  "convergence_improved": true,
  "topology_preserved": true,
  "old_convergence": 0.75,
  "new_convergence": 0.92
}
```

#### Detect Institutional Obfuscation

```bash
curl -X POST "http://localhost:8080/api/v2/frame-invariant/obfuscation/detect" \
  -H "Content-Type: application/json" \
  -d '"inference structure to analyze"'
```

### 4. Holographic Inference

**Base URL**: `/api/v2/holographic`

#### Initialize Holographic Readout

```bash
curl -X POST "http://localhost:8080/api/v2/holographic/initialize"
```

#### Execute Inference

```bash
curl -X POST "http://localhost:8080/api/v2/holographic/inference" \
  -H "Content-Type: application/json" \
  -d '{
    "query": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
  }'
```

Response:
```json
{
  "inference_vector": [0.15, 0.25, 0.35, 0.45, 0.55, 0.65, 0.75, 0.85],
  "inference_magnitude": 1.234,
  "boundary_measurements": 60,
  "bulk_state_reconstructed": true
}
```

**Note**: Query must be 8-dimensional (E8 space)

#### Measure Boundary

```bash
curl "http://localhost:8080/api/v2/holographic/boundary/measure"
```

#### Get Status

```bash
curl "http://localhost:8080/api/v2/holographic/status"
```

## Complete Workflow Example

Here's a complete workflow using all v2.0 components:

```bash
#!/bin/bash

# 1. Initialize E8 topology
curl -X POST "http://localhost:8080/api/v2/e8/initialize" \
  -H "Content-Type: application/json" \
  -d '{"n_nodes": 240}'

# 2. Initialize Majorana seeds
curl -X POST "http://localhost:8080/api/v2/majorana/initialize" \
  -H "Content-Type: application/json" \
  -d '{"n_seeds": 240, "use_azure": false}'

# 3. Initialize holographic readout
curl -X POST "http://localhost:8080/api/v2/holographic/initialize"

# 4. Initialize frame-invariant EDS
curl -X POST "http://localhost:8080/api/v2/frame-invariant/initialize"

# 5. Execute holographic inference
curl -X POST "http://localhost:8080/api/v2/holographic/inference" \
  -H "Content-Type: application/json" \
  -d '{"query": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]}'

# 6. Validate result through frame-invariant EDS
curl -X POST "http://localhost:8080/api/v2/frame-invariant/validate" \
  -H "Content-Type: application/json" \
  -d '{"proposition": "inference result from step 5"}'

# 7. Check for corruption
curl -X POST "http://localhost:8080/api/v2/frame-invariant/corruption/detect" \
  -H "Content-Type: application/json" \
  -d '{"semantic_state": "current state"}'
```

## Python Client Example

```python
import requests
import numpy as np

BASE_URL = "http://localhost:8080"

# Initialize all systems
def initialize_v2_stack():
    # E8 topology
    r = requests.post(f"{BASE_URL}/api/v2/e8/initialize",
                     json={"n_nodes": 240})
    print(f"E8: {r.json()['status']}")

    # Majorana seeds
    r = requests.post(f"{BASE_URL}/api/v2/majorana/initialize",
                     json={"n_seeds": 240, "use_azure": False})
    print(f"Majorana: {r.json()['status']}")

    # Holographic
    r = requests.post(f"{BASE_URL}/api/v2/holographic/initialize")
    print(f"Holographic: {r.json()['status']}")

    # Frame-invariant EDS
    r = requests.post(f"{BASE_URL}/api/v2/frame-invariant/initialize")
    print(f"EDS: {r.json()['status']}")

# Execute inference with validation
def execute_validated_inference(query_vector):
    # Execute inference
    r = requests.post(f"{BASE_URL}/api/v2/holographic/inference",
                     json={"query": query_vector.tolist()})
    result = r.json()
    inference = result["inference_vector"]

    # Validate through EDS
    r = requests.post(f"{BASE_URL}/api/v2/frame-invariant/validate",
                     json={"proposition": inference})
    validation = r.json()

    print(f"Inference: {inference}")
    print(f"Classification: {validation['classification']}")
    print(f"Convergence: {validation['convergence_score']:.3f}")

    return inference, validation

# Run
initialize_v2_stack()

query = np.random.randn(8)  # Random 8D query
inference, validation = execute_validated_inference(query)
```

## Error Handling

All endpoints return standard HTTP status codes:

- `200 OK`: Success
- `400 Bad Request`: Invalid input
- `404 Not Found`: Resource not found
- `500 Internal Server Error`: Server error
- `503 Service Unavailable`: Module not available

Example error response:
```json
{
  "detail": "E8 system not initialized. Call /initialize first."
}
```

## Rate Limits and Performance

- No rate limits currently enforced
- Initialization operations: ~1-2 seconds
- Inference operations: ~100-500ms
- Validation operations: ~50-200ms

## Security Considerations

- Currently no authentication required (development mode)
- Production deployment should add:
  - API key authentication
  - HTTPS/TLS encryption
  - Rate limiting
  - Input validation and sanitization

## Next Steps

1. See `docs/MIH-IIE_v2.0_Implementation.md` for detailed documentation
2. Review integration tests: `tests/test_v2_integration.py`
3. Explore interactive docs: http://localhost:8080/docs
4. Check health: http://localhost:8080/health

## Support

For issues or questions:
- See project README
- Check documentation in `docs/`
- Review implementation code in `models/`
