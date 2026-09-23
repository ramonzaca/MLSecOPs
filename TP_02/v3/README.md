# FastAPI ML Model Serving (v3)

This project serves the housing-price model from TP_01 as an HTTP API, using FastAPI and Docker.

Compared with a "just make it work" deployment, this version adds the basic safeguards any model in production should have:

| Safeguard | Why |
|---|---|
| Model loaded with **skops** by default, and pickle refused unless explicitly allowed | Loading a pickle can execute arbitrary code (see TP_01) |
| Optional **SHA-256 check** of the model file | Makes sure the API serves the model you trained, not a swapped one |
| **scikit-learn version check** at startup | A model loaded with another version can fail or return wrong results without any error |
| **Strict input schema** (types, ranges, known categories, batch size) | Bad inputs get a clear `422`, not a crash or a meaningless prediction |
| **Non-root** container, read-only code and model, health check | Limits what an attacker can do if the API is compromised |

## Prerequisites

- Docker
- The model exported at the end of the TP_01 notebook (v3 or later): `TP_01_model.skops`, plus the SHA-256 printed by the notebook

## Getting Started

1. Clone this repository:
   ```
   git clone https://github.com/ramonzaca/MLSecOPs.git
   cd MLSecOPs/TP_02/v3
   ```

2. Place your model in the **`app/models/`** directory:
   ```
   cp /path/to/TP_01_model.skops app/models/
   ```

3. Build the Docker image. `SKLEARN_VERSION` must be the version you trained with (`sklearn.__version__` in TP_01; the default is 1.9.1):
   ```
   docker build --build-arg SKLEARN_VERSION=1.9.1 -t fastapi-ml-model .
   ```

4. Run the Docker container:
   ```
   docker run -p 8000:8000 -e MODEL_SHA256=<hash printed by TP_01> fastapi-ml-model
   ```

5. The API is now accessible at `http://localhost:8000` (interactive docs at `http://localhost:8000/docs`)

If the API refuses to start, read the last line of the logs. It says which check failed and how to fix it.

### Configuration (environment variables)

| Variable | Default | Meaning |
|---|---|---|
| `MODEL_PATH` | `models/TP_01_model.skops` | Model file, relative to `app/`. `.skops` or `.pkl`. |
| `MODEL_SHA256` | *(unset)* | If set, the API refuses to start unless the model file has this hash |
| `ALLOW_PICKLE` | *(unset)* | Set to `1` to allow a `.pkl` model. Only do this for a file you produced yourself. |
| `MAX_BATCH_SIZE` | `1000` | Maximum number of districts per request |

Only have a `TP_01_model.pkl` (older TP_01 notebook)? Then:
`docker run -p 8000:8000 -e MODEL_PATH=models/TP_01_model.pkl -e ALLOW_PICKLE=1 fastapi-ml-model`

## API Endpoints

- `GET /`: Welcome message
- `GET /health`: `{"status": "ok"}` once the model is loaded (used by the Docker `HEALTHCHECK`)
- `POST /predict`: Get predictions from the model
  - Request body: `{"features": [[...], ...]}`, one list per district, with 9 values in this order:

    | # | Field | Rule |
    |---|---|---|
    | 1 | `longitude` | number |
    | 2 | `latitude` | number |
    | 3 | `housing_median_age` | number ≥ 0 |
    | 4 | `total_rooms` | number > 0 |
    | 5 | `total_bedrooms` | number ≥ 0, or `null` if unknown (the model imputes it) |
    | 6 | `population` | number ≥ 0 |
    | 7 | `households` | number > 0 |
    | 8 | `median_income` | number ≥ 0 |
    | 9 | `ocean_proximity` | one of `"<1H OCEAN"`, `"INLAND"`, `"ISLAND"`, `"NEAR BAY"`, `"NEAR OCEAN"` |

  - Response: `{"prediction": [float, ...]}`
  - Invalid input returns `422` with the location and reason of each error. The rejected values are not echoed back.

## Usage Example

```bash
curl -X POST http://localhost:8000/predict \
     -H "Content-Type: application/json" \
     -d @request_example.json
```

or, in Python (`pip install requests`):

```python
import json
import requests

with open("request_example.json", "r") as f:
    features = json.load(f)

resp = requests.post("http://localhost:8000/predict", json=features)
resp.json()

""" Output:
{'prediction': [85657.90192014378,
  305492.60737487697,
  152056.4612245569,
  186095.709460944,
  244550.67966088964]}
"""
```

The last digits can differ slightly between library versions.

## Tests

With the model in `app/models/`:

```
pip install -r requirements-dev.txt
pytest
```

The tests check that the API returns the predictions documented above, rejects invalid inputs with `422`, and that the loader refuses pickles, wrong checksums and version mismatches.

## Exercises

1. Start the API with `-e MODEL_SHA256=` set to a wrong value. What happens? Why is it better for the API to refuse to start than to start with a warning?
2. Send a request with `households = 0` to the **original** API (`TP_02/app`) and to this one. Explain the difference, and what happens inside the model.
3. Run `docker exec <container> id` and try to modify `/app/main.py` from inside the container. Why does it matter?
4. What is still missing before this API could be exposed on the Internet? (Hint: who is allowed to call it, and how often?)
