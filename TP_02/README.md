# FastAPI ML Model Serving

This project serves a pre-trained scikit-learn model using FastAPI and Docker.

## Prerequisites

- Docker

## Getting Started

1. Clone this repository:
   ```
   git clone https://github.com/ramonzaca/MLSecOPs.git
   cd TP_02
   ```

2. Place your pre-trained scikit-learn model (`TP_01_model.pkl`) in the `app` directory.

3. Build the Docker image:
   ```
   docker build -t fastapi-ml-model .
   ```

4. Run the Docker container:
   ```
   docker run -p 8000:8000 fastapi-ml-model
   ```

5. The API is now accessible at `http://localhost:8000`

## API Endpoints

- `GET /`: Welcome message
- `POST /predict`: Get predictions from the model
  - Request body: `{"features": [[float, float, ...]]}`
  - Response: `{"prediction": [float]}`

## Usage Example

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