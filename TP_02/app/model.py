import joblib
import pandas as pd

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


def load_model(model_path):
    return joblib.load(model_path)


def predict(model, features):
    entry = pd.DataFrame(features, columns=MODELS_COLUMNS)
    return model.predict(entry)
