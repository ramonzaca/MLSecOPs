"""The loader's safeguards: format, integrity and version checks."""

import pickle
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "app"))

import main  # noqa: E402,F401  (registers __main__.CombinedAttributesAdder)
from model import ModelLoadingError, load_model, sha256sum  # noqa: E402

SKOPS_MODEL = ROOT / "app" / "models" / "TP_01_model.skops"
needs_model = pytest.mark.skipif(
    not SKOPS_MODEL.is_file(), reason="export TP_01_model.skops from TP_01 first"
)


def test_pickle_is_refused_by_default(tmp_path):
    path = tmp_path / "model.pkl"
    path.write_bytes(pickle.dumps({"not": "a model"}))
    with pytest.raises(ModelLoadingError, match="refusing to load a pickle"):
        load_model(path)


@needs_model
def test_checksum_mismatch_is_refused():
    with pytest.raises(ModelLoadingError, match="checksum mismatch"):
        load_model(SKOPS_MODEL, expected_sha256="0" * 64)


@needs_model
def test_matching_checksum_is_accepted():
    assert load_model(SKOPS_MODEL, expected_sha256=sha256sum(SKOPS_MODEL)) is not None


def test_version_mismatch_is_refused(tmp_path, monkeypatch):
    import sklearn.base
    from sklearn.linear_model import LinearRegression

    path = tmp_path / "model.pkl"
    with monkeypatch.context() as m:
        # scikit-learn stamps its version into every pickle: pretend it was another one
        m.setattr(sklearn.base, "__version__", "0.0.1")
        path.write_bytes(pickle.dumps(LinearRegression()))
    with pytest.raises(ModelLoadingError, match="SKLEARN_VERSION=0.0.1"):
        load_model(path, allow_pickle=True)
