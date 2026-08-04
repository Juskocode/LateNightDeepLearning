"""Strict, shared validation for Pacman learning boundaries.

The learning API accepts NumPy scalar types as well as Python values, but it
must not silently turn fractional actions, NaNs, or arbitrary truthy values
into valid transitions.  These helpers keep that contract consistent across
the agent, trainer, and replay buffer.
"""

from __future__ import annotations

from collections.abc import Mapping
from numbers import Integral, Real
from typing import Any

import numpy as np


def require_mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be a mapping")
    return value


def strict_int(value: Any, name: str, *, minimum: int | None = None) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral):
        raise ValueError(f"{name} must be an integer")
    result = int(value)
    if minimum is not None and result < minimum:
        raise ValueError(f"{name} must be at least {minimum}")
    return result


def finite_float(value: Any, name: str) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a finite number")
    result = float(value)
    if not np.isfinite(result):
        raise ValueError(f"{name} must be finite")
    with np.errstate(over="ignore", invalid="ignore"):
        learner_value = np.float32(result)
    if not np.isfinite(learner_value):
        raise ValueError(f"{name} must be representable as float32")
    return result


def binary_flag(value: Any, name: str) -> bool:
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, Integral) and not isinstance(value, bool) and int(value) in (0, 1):
        return bool(value)
    raise ValueError(f"{name} must be a boolean or binary integer")


def action_index(value: Any, action_size: int, *, name: str = "action") -> int:
    index = strict_int(value, name, minimum=0)
    if index >= action_size:
        raise ValueError(f"{name} is outside the configured action space")
    return index


def boolean_mask(
    values: Any,
    action_size: int,
    *,
    name: str = "legal_action_mask",
) -> np.ndarray:
    raw = np.asarray(values)
    if raw.shape != (action_size,):
        raise ValueError(f"{name} must have shape ({action_size},)")
    if raw.dtype.kind == "b":
        mask = raw.astype(np.bool_, copy=True)
    elif raw.dtype.kind in "iu":
        if not np.all((raw == 0) | (raw == 1)):
            raise ValueError(f"{name} must contain only boolean or binary values")
        mask = raw.astype(np.bool_, copy=True)
    elif raw.dtype.kind in "fc":
        if not np.isfinite(raw).all() or not np.all((raw == 0) | (raw == 1)):
            raise ValueError(f"{name} must contain only finite binary values")
        mask = raw.astype(np.bool_, copy=True)
    else:
        raise ValueError(f"{name} must contain only boolean or binary values")
    if not mask.any():
        raise ValueError(f"{name} must allow at least one action")
    return mask


def finite_vector(values: Any, size: int, *, name: str) -> np.ndarray:
    try:
        result = np.asarray(values, dtype=np.float32).reshape(-1)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError(f"{name} must be a numeric vector") from error
    if result.shape != (size,):
        raise ValueError(f"{name} must have shape ({size},)")
    if not np.isfinite(result).all():
        raise ValueError(f"{name} must contain finite values")
    return result
