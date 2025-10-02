"""Tests for platform-agnostic math helpers."""

from types import SimpleNamespace
from typing import Any, Callable, cast

import importlib
import builtins

import numpy as np
import pytest

from hierarchical_naics_model.utils import _math


def _sigmoid_numpy(x: np.ndarray | float | int) -> np.ndarray | float:
    return 1.0 / (1.0 + np.exp(-x))


@pytest.fixture
def restore_math_module():
    """Reload `_math` after tests that mutate module-level state."""

    yield
    importlib.reload(_math)


def test_import_fallback_sets_flags(monkeypatch, restore_math_module):
    """If PyMC imports fail, the module should expose the NumPy-only defaults."""

    original_import = builtins.__import__

    def fake_import(name: str, *args, **kwargs):  # pragma: no cover - simple guard
        if name in {"pymc", "pytensor.tensor"}:
            raise ImportError("forced failure for test")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)

    module = cast(Any, importlib.reload(_math))

    assert module._has_pymc is False
    assert module.pm is None
    assert module.pt is None


@pytest.mark.parametrize(
    "func_name, expected",
    (
        ("exp", np.exp),
        ("sigmoid", _sigmoid_numpy),
    ),
)
@pytest.mark.parametrize("input_value", (0.0, np.array([-1.0, 2.0], dtype=float)))
def test_numpy_fallback(func_name: str, expected: Callable, input_value, monkeypatch):
    """When PyMC is unavailable, wrappers should delegate to NumPy implementations."""

    monkeypatch.setattr(_math, "_has_pymc", False)
    monkeypatch.setattr(_math, "pt", None)

    result = getattr(_math, func_name)(input_value)
    np.testing.assert_allclose(result, expected(input_value))


@pytest.mark.parametrize(
    "func_name, pt_attr",
    (
        ("exp", "exp"),
        ("sigmoid", "sigmoid"),
    ),
)
def test_tensor_variable_branch(func_name: str, pt_attr: str, monkeypatch):
    """If PyMC tensor types are detected, wrappers should call into PyTensor."""

    class FakeTensor:  # minimal stand-in for pt.TensorVariable
        pass

    called_with: list[FakeTensor] = []

    def fake_func(x: FakeTensor) -> str:
        called_with.append(x)
        return f"{pt_attr}-result"

    fake_pt = SimpleNamespace(TensorVariable=FakeTensor, **{pt_attr: fake_func})

    monkeypatch.setattr(_math, "_has_pymc", True)
    monkeypatch.setattr(_math, "pt", fake_pt)

    tensor_input = FakeTensor()
    result = getattr(_math, func_name)(tensor_input)

    assert result == f"{pt_attr}-result"
    assert called_with == [tensor_input]
