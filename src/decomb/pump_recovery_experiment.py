"""Cohort orchestration for the isolated 1.2 Hz recovery experiment."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from decomb import pump_recovery


def stable_seed(*parts: str) -> int:
    """Derive one reproducible NumPy seed from recorded identifiers."""
    if not parts or any(not str(part) for part in parts):
        raise ValueError("seed parts must be non-empty strings")
    payload = "\0".join(str(part) for part in parts).encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "little")


@dataclass(frozen=True)
class PredictionEvaluation:
    """Pump-lock and prediction-error results for one held-out recording."""

    recording: str
    participant: str
    source_maximum_coherence: float
    source_p_value: float
    residual_maximum_coherence: float
    residual_p_value: float
    normalized_prediction_error: float
    rank: int
    penalty: float
    validation_error: float


def evaluate_prediction(
    prediction: pump_recovery.HeldOutPrediction,
    features: NDArray[np.complexfloating],
    *,
    surrogate_count: int,
) -> PredictionEvaluation:
    """Evaluate source and residual against a clock built only from high teeth."""
    predictors = np.asarray(features, dtype=np.complex128)
    if predictors.ndim != 2 or predictors.shape[0] != (
        prediction.target_coefficients.shape[0]
    ):
        raise ValueError("features must match the held-out prediction windows")
    clock = pump_recovery.pump_clock_reference(predictors)
    reference = np.repeat(
        clock[:, None],
        prediction.target_coefficients.shape[1],
        axis=1,
    )
    source = pump_recovery.pump_lock_test(
        prediction.target_coefficients,
        reference,
        surrogate_count=surrogate_count,
        seed=stable_seed(prediction.recording, "source"),
    )
    residual_coefficients = (
        prediction.target_coefficients - prediction.predicted_coefficients
    )
    residual = pump_recovery.pump_lock_test(
        residual_coefficients,
        reference,
        surrogate_count=surrogate_count,
        seed=stable_seed(prediction.recording, "residual"),
    )
    target_energy = float(np.sum(np.abs(prediction.target_coefficients) ** 2))
    if target_energy <= 0.0:
        raise ValueError("target coefficients must have positive energy")
    residual_energy = float(np.sum(np.abs(residual_coefficients) ** 2))
    return PredictionEvaluation(
        prediction.recording,
        prediction.test_participant,
        source.maximum_coherence,
        source.p_value,
        residual.maximum_coherence,
        residual.p_value,
        residual_energy / target_energy,
        prediction.rank,
        prediction.penalty,
        prediction.validation_error,
    )
