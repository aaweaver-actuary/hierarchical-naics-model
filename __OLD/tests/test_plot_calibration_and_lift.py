import numpy as np
import pytest
import pandas as pd
from hierarchical_naics_model.plot_calibration_and_lift import (
    plot_reliability_curve,
    plot_lift_and_gain,
    plot_precision_at_k,
    plot_brier_logloss,
)


@pytest.fixture
def dummy_data():
    y_true = np.array([0, 1, 0, 1, 1, 0, 1, 0])
    y_pred = np.array([0.1, 0.9, 0.2, 0.8, 0.7, 0.3, 0.6, 0.4])
    return y_true, y_pred


def test_plot_reliability_curve_runs(dummy_data):
    y_true, y_pred = dummy_data
    # Create a reliability curve DataFrame as expected by the function
    df_rel = pd.DataFrame(
        {
            "p_avg": y_pred,
            "y_rate": y_true,
            "count": np.ones_like(y_true),
            "abs_gap": np.abs(y_pred - y_true),
        }
    )
    fig = plot_reliability_curve(df_rel)
    assert fig is not None
    fig.write_image("/tmp/test_plot_reliability_curve.png")  # smoke test for plotly


def test_plot_lift_and_gain_runs(dummy_data):
    y_true, y_pred = dummy_data
    # Create a DataFrame as expected by the function
    df_rank = pd.DataFrame(
        {
            "k_pct": np.linspace(0, 1, len(y_true)),
            "lift_at_k": y_pred,
            "cum_gain_pct": np.cumsum(y_true) / np.sum(y_true),
        }
    )
    fig = plot_lift_and_gain(df_rank)
    assert fig is not None
    fig.write_image("/tmp/test_plot_lift_and_gain.png")


def test_plot_precision_at_k_runs(dummy_data):
    y_true, y_pred = dummy_data
    # Create a DataFrame as expected by the function
    df_rank = pd.DataFrame(
        {
            "k_pct": np.linspace(0, 1, len(y_true)),
            "precision_at_k": y_pred,
        }
    )
    fig = plot_precision_at_k(df_rank)
    assert fig is not None
    fig.write_image("/tmp/test_plot_precision_at_k.png")


def test_plot_brier_logloss_runs(dummy_data):
    y_true, y_pred = dummy_data
    # Create a summary dict as expected by the function
    summary = {
        "brier": float(np.mean((y_pred - y_true) ** 2)),
        "log_loss": float(
            np.mean(
                -y_true * np.log(y_pred + 1e-8)
                - (1 - y_true) * np.log(1 - y_pred + 1e-8)
            )
        ),
    }
    fig = plot_brier_logloss(summary)
    assert fig is not None
    fig.write_image("/tmp/test_plot_brier_logloss.png")


def test_plot_brier_logloss_custom_layout_and_title():
    summary = {"brier": 0.1, "log_loss": 0.2}
    custom_layout = {"title": "Custom Title", "font": {"size": 18}}
    fig = plot_brier_logloss(summary, title="Custom Title", custom_layout=custom_layout)
    assert fig is not None
    assert fig.layout.title.text == "Custom Title"
    assert fig.layout.font.size == 18
    fig.write_image("/tmp/test_plot_brier_logloss_custom.png")
