"""Applicability domain for QSAR models.

A trained model will happily extrapolate onto chemistry it has never seen and
report the prediction with no hint that it is guessing. An applicability domain
(AD) answers "is this query similar enough to the training set for the
prediction to mean anything?".

Two standard formulations are provided:

``SimilarityAD``
    Nearest-neighbour Tanimoto similarity to the training set. Works on
    fingerprints, so it applies to the full 2048-bit representation and is the
    right default for a fingerprint model.

``LeverageAD``
    The classical leverage / Williams-plot criterion, h_i = x_i (X'X)^-1 x_i',
    flagged when h_i exceeds h* = 3p/n. Only meaningful on the low-dimensional
    descriptor block -- with more features than compounds, X'X is singular and
    every point is formally out of domain.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


# --------------------------------------------------------------- similarity ---
def tanimoto_matrix(query: np.ndarray, reference: np.ndarray) -> np.ndarray:
    """Tanimoto similarity between every query and every reference fingerprint.

    Both arrays are 0/1 of shape ``(n, n_bits)``. Returns ``(n_query, n_ref)``.
    """
    q = np.asarray(query, dtype=np.float32)
    r = np.asarray(reference, dtype=np.float32)
    if q.ndim == 1:
        q = q[None, :]
    if r.ndim == 1:
        r = r[None, :]
    inter = q @ r.T
    q_sum = q.sum(axis=1)[:, None]
    r_sum = r.sum(axis=1)[None, :]
    union = q_sum + r_sum - inter
    with np.errstate(divide="ignore", invalid="ignore"):
        sim = np.where(union > 0, inter / union, 0.0)
    return sim


@dataclass
class SimilarityAD:
    """Nearest-neighbour Tanimoto applicability domain.

    The threshold is derived from the training set itself: each training
    compound's similarity to its nearest *other* training compound gives a
    distribution of "how close things normally are", and the ``percentile``
    quantile of that distribution becomes the cutoff. A query whose nearest
    training neighbour is less similar than that is flagged out of domain.
    """

    threshold: float
    percentile: float
    n_train: int
    _train: np.ndarray

    @classmethod
    def fit(cls, train_fps: np.ndarray, percentile: float = 5.0) -> SimilarityAD:
        fps = np.asarray(train_fps, dtype=np.float32)
        if fps.ndim != 2 or len(fps) < 2:
            raise ValueError("need at least 2 training fingerprints, shape (n, n_bits)")
        sim = tanimoto_matrix(fps, fps)
        np.fill_diagonal(sim, -1.0)          # exclude self-similarity
        nearest = sim.max(axis=1)
        return cls(threshold=float(np.percentile(nearest, percentile)),
                   percentile=float(percentile),
                   n_train=len(fps),
                   _train=fps)

    def similarity(self, query_fps: np.ndarray) -> np.ndarray:
        """Similarity of each query to its nearest training compound."""
        return tanimoto_matrix(query_fps, self._train).max(axis=1)

    def in_domain(self, query_fps: np.ndarray) -> np.ndarray:
        """Boolean mask: True where the prediction is inside the domain."""
        return self.similarity(query_fps) >= self.threshold

    def as_dict(self) -> dict:
        """Serialisable form, for persisting next to a model."""
        return {"kind": "similarity_tanimoto",
                "threshold": self.threshold,
                "percentile": self.percentile,
                "n_train": self.n_train}


# ----------------------------------------------------------------- leverage ---
@dataclass
class LeverageAD:
    """Leverage (Williams plot) applicability domain.

    ``h_star = 3p/n`` is the conventional warning level. Use the descriptor block
    only: with 2048 fingerprint bits and a few hundred compounds the design
    matrix is singular and the criterion is meaningless.
    """

    h_star: float
    n_train: int
    n_features: int
    _xtx_inv: np.ndarray
    _mean: np.ndarray

    @classmethod
    def fit(cls, train_X: np.ndarray) -> LeverageAD:
        X = np.asarray(train_X, dtype=np.float64)
        n, p = X.shape
        if n <= p:
            raise ValueError(
                f"leverage needs more compounds than features (got n={n}, p={p}); "
                "use SimilarityAD for fingerprints"
            )
        mean = X.mean(axis=0)
        Xc = X - mean
        # pinv rather than inv: descriptor columns are often near-collinear.
        return cls(h_star=float(3.0 * p / n), n_train=n, n_features=p,
                   _xtx_inv=np.linalg.pinv(Xc.T @ Xc), _mean=mean)

    def leverage(self, query_X: np.ndarray) -> np.ndarray:
        X = np.asarray(query_X, dtype=np.float64)
        if X.ndim == 1:
            X = X[None, :]
        Xc = X - self._mean
        return np.einsum("ij,jk,ik->i", Xc, self._xtx_inv, Xc)

    def in_domain(self, query_X: np.ndarray) -> np.ndarray:
        return self.leverage(query_X) <= self.h_star

    def as_dict(self) -> dict:
        return {"kind": "leverage",
                "h_star": self.h_star,
                "n_train": self.n_train,
                "n_features": self.n_features}
