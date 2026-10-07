import hashlib
import logging
import warnings
from pathlib import Path

import pandas as pd
import skops.io as sio
from sklearn.exceptions import InconsistentVersionWarning

logger = logging.getLogger(__name__)

MODELS_COLUMNS = [
    "longitude",
    "latitude",
    "housing_median_age",
    "total_rooms",
    "total_bedrooms",
    "population",
    "households",
    "median_income",
    "ocean_proximity",
]

# Types that skops may load without refusing: our own transformer, plus numpy.dtype,
# which skops lists as untrusted by default although it is harmless.
SKOPS_TRUSTED_TYPES = ["__main__.CombinedAttributesAdder", "numpy.dtype"]


class ModelLoadingError(RuntimeError):
    pass


def sha256sum(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_model(model_path, expected_sha256=None, allow_pickle=False):
    model_path = Path(model_path)
    if not model_path.is_file():
        raise ModelLoadingError(f"model file not found: {model_path}")

    # 1. Integrity: is this the file we expect?
    digest = sha256sum(model_path)
    logger.info("model %s sha256=%s", model_path.name, digest)
    if expected_sha256 and digest != expected_sha256.lower():
        raise ModelLoadingError(
            f"model checksum mismatch: expected {expected_sha256}, got {digest}"
        )

    # 2. Format: skops refuses unknown types; pickle executes whatever the file contains
    with warnings.catch_warnings():
        # 3. Compatibility: a model saved with another scikit-learn version may silently misbehave
        warnings.simplefilter("error", InconsistentVersionWarning)
        try:
            if model_path.suffix == ".skops":
                return sio.load(model_path, trusted=SKOPS_TRUSTED_TYPES)
            if model_path.suffix == ".pkl":
                if not allow_pickle:
                    raise ModelLoadingError(
                        "refusing to load a pickle: loading it can execute arbitrary code. "
                        "Use the .skops export from TP_01, or set ALLOW_PICKLE=1 if you trust this file."
                    )
                logger.warning(
                    "loading a pickle file (ALLOW_PICKLE=1): only do this with files you trust"
                )
                import joblib

                return joblib.load(model_path)
        except InconsistentVersionWarning as w:
            raise ModelLoadingError(
                f"model was trained with scikit-learn {w.original_sklearn_version} but this image has "
                f"{w.current_sklearn_version}. Rebuild with "
                f"`docker build --build-arg SKLEARN_VERSION={w.original_sklearn_version} ...`"
            ) from None
    raise ModelLoadingError(
        f"unsupported model format: {model_path.suffix} (expected .skops or .pkl)"
    )


def model_categories(model):
    """Return the ocean_proximity categories the model's one-hot encoder was fitted on."""
    encoder = model.named_steps["preparation"].named_transformers_["cat"]
    return set(encoder.categories_[0])


def predict(model, features):
    entry = pd.DataFrame(features, columns=MODELS_COLUMNS)
    return model.predict(entry)
