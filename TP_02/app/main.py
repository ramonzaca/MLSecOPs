import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated, Literal, get_args

import numpy as np
from classes import CombinedAttributesAdder
from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from model import load_model, model_categories, predict
from pydantic import BaseModel, Field

import __main__

# The model was saved from a notebook, so it refers to `__main__.CombinedAttributesAdder`.
# Making the class available under that name lets the loader find it.
__main__.CombinedAttributesAdder = CombinedAttributesAdder

logging.basicConfig(
    level=logging.INFO, format="%(levelname)s:     %(name)s - %(message)s"
)
logger = logging.getLogger("tp02")

APP_DIR = Path(__file__).resolve().parent
MODEL_PATH = APP_DIR / os.environ.get("MODEL_PATH", "models/TP_01_model.skops")
MODEL_SHA256 = os.environ.get("MODEL_SHA256")  # optional: refuse any other file
ALLOW_PICKLE = os.environ.get("ALLOW_PICKLE") == "1"
MAX_BATCH_SIZE = int(os.environ.get("MAX_BATCH_SIZE", "1000"))

# ---- Input schema: one district = 9 values, in the order of model.MODELS_COLUMNS
Finite = Annotated[float, Field(allow_inf_nan=False)]
NonNegative = Annotated[float, Field(ge=0, allow_inf_nan=False)]
Positive = Annotated[
    float, Field(gt=0, allow_inf_nan=False)
]  # used as a divisor by the model
OceanProximity = Literal["<1H OCEAN", "INLAND", "ISLAND", "NEAR BAY", "NEAR OCEAN"]

District = tuple[
    Finite,  # longitude
    Finite,  # latitude
    NonNegative,  # housing_median_age
    Positive,  # total_rooms
    NonNegative | None,  # total_bedrooms (null = unknown, imputed by the model)
    NonNegative,  # population
    Positive,  # households
    NonNegative,  # median_income
    OceanProximity,  # ocean_proximity
]


class InputData(BaseModel):
    features: Annotated[list[District], Field(min_length=1, max_length=MAX_BATCH_SIZE)]


class Prediction(BaseModel):
    prediction: list[float]


model = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global model
    model = load_model(
        MODEL_PATH, expected_sha256=MODEL_SHA256, allow_pickle=ALLOW_PICKLE
    )
    # The API only accepts the categories listed in OceanProximity: they must match the model's
    if model_categories(model) != set(get_args(OceanProximity)):
        raise RuntimeError(
            f"model categories {model_categories(model)} don't match the API schema"
        )
    logger.info("model loaded, API ready")
    yield


app = FastAPI(title="TP_02 housing price API", lifespan=lifespan)


@app.exception_handler(RequestValidationError)
async def validation_error_handler(request: Request, exc: RequestValidationError):
    # FastAPI's default 422 echoes the rejected input back. Besides reflecting attacker-controlled
    # content, that crashes on NaN/inf (not valid JSON). Return only where and why it failed.
    errors = [{"loc": err["loc"], "msg": err["msg"]} for err in exc.errors()]
    return JSONResponse(status_code=422, content={"detail": errors})


@app.post("/predict")
def get_prediction(data: InputData) -> Prediction:
    prediction = predict(model, data.features)
    if not np.all(np.isfinite(prediction)):
        # Never return NaN/inf: it isn't valid JSON and hides a problem upstream
        raise HTTPException(
            status_code=500, detail="the model produced a non-finite prediction"
        )
    return Prediction(prediction=prediction.tolist())


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/")
def root():
    return {"message": "Welcome to the TP2 ML Model API"}
