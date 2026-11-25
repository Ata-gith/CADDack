"""Loading and using the trained affinity forest."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class AffinityModel:
    """A trained random forest, with the metadata needed to read its output.

    ``predict`` returns the mean over trees and the spread across them. That
    spread is an ensemble disagreement, not a calibrated confidence interval:
    on the held-out set it correlates only weakly with the error actually made,
    which is a limitation shared with the graph model it replaced.
    """

    forest: object
    feature_names: list[str]
    metrics: dict
    #: ECFP4 bits the model was trained with. Anything featurising for this
    #: model has to use the same width, or the columns do not line up.
    fingerprint_bits: int = 0

    def featurize(self, *args, **kwargs) -> np.ndarray | None:
        """Featurise for *this* model, with its own fingerprint width.

        Call this rather than ``features_from_atoms`` directly: the width is a
        property of the trained model, and getting it wrong shifts every contact
        count into a fingerprint column without raising anything until the
        length check below.
        """
        from caddack.affinity.features import features_from_atoms

        kwargs.setdefault("fp_bits", self.fingerprint_bits)
        return features_from_atoms(*args, **kwargs)

    def predict(self, x: np.ndarray) -> tuple[float, float]:
        """Predicted pK and the standard deviation across trees."""
        x = np.asarray(x, dtype=np.float64).reshape(1, -1)
        if x.shape[1] != len(self.feature_names):
            raise ValueError(
                f"expected {len(self.feature_names)} features, got {x.shape[1]}")
        per_tree = np.array([t.predict(x)[0] for t in self.forest.estimators_])
        return float(per_tree.mean()), float(per_tree.std(ddof=0))

    def predict_batch(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        X = np.asarray(X, dtype=np.float64)
        if X.ndim != 2 or X.shape[1] != len(self.feature_names):
            raise ValueError(
                f"expected (n, {len(self.feature_names)}) features, got {X.shape}")
        per_tree = np.stack([t.predict(X) for t in self.forest.estimators_])
        return per_tree.mean(0), per_tree.std(0, ddof=0)


def load_model(directory: str | Path) -> AffinityModel:
    """Load a model written by ``scripts/train_affinity_rf.py``."""
    directory = Path(directory)
    try:
        from joblib import load
    except Exception as exc:  # pragma: no cover - joblib is a core dependency
        raise ImportError("joblib is required to load the affinity model.") from exc

    meta = json.loads((directory / "metadata.json").read_text())
    return AffinityModel(
        forest=load(directory / "forest.joblib"),
        feature_names=meta["feature_names"],
        metrics=meta["metrics"],
        fingerprint_bits=int(meta.get("fingerprint_bits", 0)),
    )
