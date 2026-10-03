"""Classical Growing Neural Gas (Fritzke, 1995)."""

from __future__ import annotations

import numpy as np

__all__ = ["GrowingNeuralGas"]


class GrowingNeuralGas:
    """Online Growing Neural Gas with a fixed unit cap.

    Units begin as two samples joined by an edge. Each new sample moves the
    nearest unit and its neighbors, refreshes the edge to the second-nearest
    unit, and drops edges older than ``max_edge_age``. Every ``lambda_steps``
    samples, a unit is inserted between the highest-error unit and its
    highest-error neighbor until ``max_neurons`` is reached.
    """

    def __init__(
        self,
        input_dim: int,
        max_neurons: int,
        lr_winner: float = 0.05,
        lr_neighbor: float = 0.0006,
        max_edge_age: int = 50,
        lambda_steps: int = 100,
        alpha: float = 0.5,
        beta: float = 0.995,
        seed: int | None = None,
    ) -> None:
        if input_dim < 1:
            raise ValueError("input_dim must be >= 1.")
        if max_neurons < 2:
            raise ValueError("max_neurons must be >= 2.")
        if lr_winner <= 0 or lr_neighbor < 0:
            raise ValueError("lr_winner must be > 0 and lr_neighbor must be >= 0.")
        if max_edge_age < 1:
            raise ValueError("max_edge_age must be >= 1.")
        if lambda_steps < 1:
            raise ValueError("lambda_steps must be >= 1.")
        if not 0 < alpha <= 1:
            raise ValueError("alpha must be in (0, 1].")
        if not 0 <= beta < 1:
            raise ValueError("beta must be in [0, 1).")

        self.input_dim = int(input_dim)
        self.max_neurons = int(max_neurons)
        self.lr_winner = float(lr_winner)
        self.lr_neighbor = float(lr_neighbor)
        self.max_edge_age = int(max_edge_age)
        self.lambda_steps = int(lambda_steps)
        self.alpha = float(alpha)
        self.beta = float(beta)

        self._rng = np.random.default_rng(seed)
        self._weights = np.zeros((self.max_neurons, self.input_dim), dtype=np.float64)
        self._errors = np.zeros(self.max_neurons, dtype=np.float64)
        self._ages = np.full(
            (self.max_neurons, self.max_neurons),
            -1,
            dtype=np.int32,
        )
        self._diff = np.empty((self.max_neurons, self.input_dim), dtype=np.float64)
        self._squared = np.empty((self.max_neurons, self.input_dim), dtype=np.float64)
        self._dist2 = np.empty(self.max_neurons, dtype=np.float64)
        self._n = 0
        self._steps = 0
        self._initialized = False

    @property
    def n_nodes(self) -> int:
        """Number of units currently in the graph."""
        return self._n

    @property
    def weights(self) -> np.ndarray:
        """Codebook of living units, shape ``[n_nodes, input_dim]``."""
        self._require_fit()
        return self._weights[: self._n].copy()

    def fit(
        self,
        data: np.ndarray,
        epochs: int = 1,
        shuffle: bool = True,
    ) -> None:
        """Present every row of ``data`` once per epoch.

        The first call places the initial two units. Later calls keep training
        the same graph.
        """
        samples = _as_2d(data)
        if isinstance(epochs, bool) or not isinstance(epochs, int) or epochs < 1:
            raise ValueError("epochs must be an integer >= 1.")
        if samples.shape[1] != self.input_dim:
            raise ValueError(
                f"data has dimension {samples.shape[1]}, expected {self.input_dim}."
            )
        if not self._initialized:
            if samples.shape[0] < 2:
                raise ValueError("fit requires at least 2 samples.")
            self._initialize(samples)

        for _ in range(epochs):
            order = (
                self._rng.permutation(samples.shape[0])
                if shuffle
                else np.arange(samples.shape[0])
            )
            for index in order:
                self._adapt(samples[index])

    def predict(self, data: np.ndarray) -> np.ndarray:
        """Index of the nearest living unit for each row of ``data``."""
        self._require_fit()
        samples = _as_2d(data)
        self._check_dim(samples)
        indices, _ = _nearest(samples, self._weights[: self._n])
        return indices

    def quantization_error(self, data: np.ndarray) -> float:
        """Mean Euclidean distance from each row to its nearest unit."""
        self._require_fit()
        samples = _as_2d(data)
        self._check_dim(samples)
        _, dist2 = _nearest(samples, self._weights[: self._n])
        return float(np.mean(np.sqrt(dist2)))

    def _initialize(self, samples: np.ndarray) -> None:
        first, second = self._rng.choice(samples.shape[0], size=2, replace=False)
        self._weights[:] = 0.0
        self._errors[:] = 0.0
        self._ages[:] = -1
        self._weights[0] = samples[first]
        self._weights[1] = samples[second]
        self._ages[0, 1] = 0
        self._ages[1, 0] = 0
        self._n = 2
        self._steps = 0
        self._initialized = True

    def _adapt(self, sample: np.ndarray) -> None:
        n = self._n
        weights = self._weights
        ages = self._ages
        diff = self._diff[:n]
        np.subtract(weights[:n], sample, out=diff)
        squared = self._squared[:n]
        np.multiply(diff, diff, out=squared)
        dist2 = self._dist2[:n]
        np.sum(squared, axis=1, out=dist2)

        if n == 2:
            s1, s2 = (0, 1) if dist2[0] <= dist2[1] else (1, 0)
        else:
            part = np.argpartition(dist2, 1)
            s1 = int(part[0])
            s2 = int(part[1])
            if dist2[s2] < dist2[s1]:
                s1, s2 = s2, s1

        neighbors = np.flatnonzero(ages[s1, :n] >= 0)
        if neighbors.size:
            ages[s1, neighbors] += 1
            ages[neighbors, s1] = ages[s1, neighbors]
            weights[neighbors] -= self.lr_neighbor * diff[neighbors]

        self._errors[s1] += dist2[s1]
        weights[s1] -= self.lr_winner * diff[s1]

        ages[s1, s2] = 0
        ages[s2, s1] = 0

        connected = np.flatnonzero(ages[s1, :n] >= 0)
        old = connected[ages[s1, connected] > self.max_edge_age]
        if old.size:
            ages[s1, old] = -1
            ages[old, s1] = -1
            self._remove_isolated(old)

        self._steps += 1
        if self._steps % self.lambda_steps == 0 and self._n < self.max_neurons:
            self._insert()
        self._errors[: self._n] *= self.beta

    def _insert(self) -> None:
        n = self._n
        errors = self._errors
        q = int(np.argmax(errors[:n]))
        neighbors = np.flatnonzero(self._ages[q, :n] >= 0)
        if neighbors.size == 0:
            return

        f = int(neighbors[np.argmax(errors[neighbors])])
        r = n
        self._ages[r, :] = -1
        self._ages[:, r] = -1
        self._weights[r] = 0.5 * (self._weights[q] + self._weights[f])
        self._ages[q, f] = -1
        self._ages[f, q] = -1
        self._ages[q, r] = 0
        self._ages[r, q] = 0
        self._ages[f, r] = 0
        self._ages[r, f] = 0
        errors[q] *= self.alpha
        errors[f] *= self.alpha
        errors[r] = errors[q]
        self._n = n + 1

    def _remove_isolated(self, candidates: np.ndarray) -> None:
        n = self._n
        isolated: list[int] = []
        seen: set[int] = set()
        for raw in candidates:
            index = int(raw)
            if index in seen or index >= n:
                continue
            seen.add(index)
            if not np.any(self._ages[index, :n] >= 0):
                isolated.append(index)

        extra = n - 2
        if extra <= 0:
            return
        for index in sorted(isolated, reverse=True)[:extra]:
            self._delete(index)

    def _delete(self, index: int) -> None:
        last = self._n - 1
        ages = self._ages
        if index != last:
            self._weights[index] = self._weights[last]
            self._errors[index] = self._errors[last]
            row = ages[index, : last + 1].copy()
            ages[index, : last + 1] = ages[last, : last + 1]
            ages[last, : last + 1] = row
            column = ages[: last + 1, index].copy()
            ages[: last + 1, index] = ages[: last + 1, last]
            ages[: last + 1, last] = column
        ages[last, :] = -1
        ages[:, last] = -1
        self._errors[last] = 0.0
        self._weights[last] = 0.0
        self._n = last

    def _require_fit(self) -> None:
        if not self._initialized:
            raise RuntimeError("Model is not initialized. Call fit() first.")

    def _check_dim(self, samples: np.ndarray) -> None:
        if samples.shape[1] != self.input_dim:
            raise ValueError(
                f"data has dimension {samples.shape[1]}, expected {self.input_dim}."
            )


def _as_2d(data: np.ndarray) -> np.ndarray:
    if hasattr(data, "detach"):
        data = data.detach().cpu().numpy()
    samples = np.asarray(data, dtype=np.float64)
    if samples.ndim != 2:
        raise ValueError("data must have shape [N, D].")
    return np.ascontiguousarray(samples)


def _nearest(samples: np.ndarray, weights: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Nearest-unit index and squared Euclidean distance for each sample."""
    count = samples.shape[0]
    indices = np.empty(count, dtype=np.int64)
    min_dist2 = np.empty(count, dtype=np.float64)
    weight_norm = np.einsum("md,md->m", weights, weights)
    chunk = 8192
    for start in range(0, count, chunk):
        stop = min(start + chunk, count)
        batch = samples[start:stop]
        dist2 = (
            np.einsum("nd,nd->n", batch, batch)[:, None]
            + weight_norm
            - 2.0 * (batch @ weights.T)
        )
        np.maximum(dist2, 0.0, out=dist2)
        nearest = np.argmin(dist2, axis=1)
        indices[start:stop] = nearest
        min_dist2[start:stop] = dist2[np.arange(stop - start), nearest]
    return indices, min_dist2
