from __future__ import annotations

import pytest

from hierarchical_naics_model.modeling import sampling as sampling_module


def test_sample_functions_require_pymc(monkeypatch):
    monkeypatch.setattr(sampling_module, "pm", None)

    with pytest.raises(RuntimeError):
        sampling_module.sample_posterior(object())

    with pytest.raises(RuntimeError):
        sampling_module.sample_prior_predictive(object())


def test_sample_helpers_delegate_to_pm(monkeypatch):
    calls = {}

    class DummyPM:
        def sample(self, **kwargs):
            calls["posterior"] = kwargs
            return "posterior-idata"

        def sample_prior_predictive(self, **kwargs):
            calls["prior"] = kwargs
            return "prior-idata"

    class DummyModel:
        default_sampling_kwargs = {"draws": 10, "chains": 1}

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

    dummy_pm = DummyPM()
    monkeypatch.setattr(sampling_module, "pm", dummy_pm)

    model = DummyModel()

    out_post = sampling_module.sample_posterior(model, tune=5)
    assert out_post == "posterior-idata"
    assert calls["posterior"]["draws"] == 10
    assert calls["posterior"]["tune"] == 5

    out_prior = sampling_module.sample_prior_predictive(
        model, samples=7, random_seed=99
    )
    assert out_prior == "prior-idata"
    assert calls["prior"] == {"draws": 7, "random_seed": 99}
