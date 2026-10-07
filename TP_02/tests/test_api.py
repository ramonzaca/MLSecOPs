"""Run from TP_02/v3 with the model in app/models/:  pytest"""

import json
import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "app"))

if not (ROOT / "app" / "models" / "TP_01_model.skops").is_file():
    pytest.skip("app/models/TP_01_model.skops not found (export it from TP_01)", allow_module_level=True)

from main import app  # noqa: E402

EXAMPLE = json.loads((ROOT / "request_example.json").read_text())
# Documented in README.md: what the TP_01 model returns for request_example.json
EXPECTED = [85657.90192014378, 305492.60737487697, 152056.4612245569, 186095.709460944, 244550.67966088964]
VALID_ROW = EXAMPLE["features"][0]


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:  # runs the lifespan, i.e. loads the model
        yield c


def with_value(index, value):
    row = list(VALID_ROW)
    row[index] = value
    return {"features": [row]}


def test_health(client):
    assert client.get("/health").json() == {"status": "ok"}


def test_predict_matches_documented_output(client):
    resp = client.post("/predict", json=EXAMPLE)
    assert resp.status_code == 200
    assert resp.json()["prediction"] == pytest.approx(EXPECTED, rel=1e-9)


def test_missing_total_bedrooms_is_imputed(client):
    resp = client.post("/predict", json=with_value(4, None))
    assert resp.status_code == 200


@pytest.mark.parametrize(
    "payload",
    [
        {"features": [[1, 2, 3]]},  # wrong number of values
        with_value(8, "MARS"),  # unknown category
        with_value(6, 0),  # households = 0 → division by zero in the model
        with_value(3, -10),  # negative total_rooms
        with_value(7, "a lot"),  # not a number
        {"features": []},  # empty batch
        {"features": [VALID_ROW] * 1001},  # batch too large
    ],
)
def test_invalid_input_is_rejected_with_422(client, payload):
    assert client.post("/predict", json=payload).status_code == 422


def test_non_finite_json_is_rejected(client):
    body = json.dumps(with_value(0, float("nan")))  # Python writes NaN, which is not valid JSON
    resp = client.post("/predict", content=body, headers={"content-type": "application/json"})
    assert resp.status_code == 422
