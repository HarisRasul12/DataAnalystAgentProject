from __future__ import annotations

import pandas as pd

from oceanwatch.metrics import compute_ocean_metrics, compute_series_metrics


def test_compute_series_metrics_returns_slope_and_counts() -> None:
    frame = pd.DataFrame(
        {
            "timestamp": pd.date_range("2026-03-30", periods=10, freq="h", tz="UTC"),
            "wave_height_m": [0.5, 0.6, 0.7, 0.8, 0.6, 0.9, 1.0, 1.1, 1.0, 0.95],
        }
    )

    metrics = compute_series_metrics(frame, "wave_height_m")

    assert metrics.count == 10
    assert metrics.latest == 0.95
    assert metrics.slope_per_hour is not None


def test_compute_ocean_metrics_populates_cards() -> None:
    buoy = pd.DataFrame(
        {
            "timestamp": pd.date_range("2026-03-30", periods=6, freq="h", tz="UTC"),
            "wave_height_m": [0.5, 0.7, 0.8, 0.9, 1.0, 1.1],
            "wind_speed_mps": [4, 5, 5, 6, 6, 7],
            "gust_mps": [6, 7, 8, 8, 9, 10],
            "water_temp_c": [12, 12.1, 12.2, 12.2, 12.3, 12.4],
        }
    )
    water = pd.DataFrame(
        {
            "timestamp": pd.date_range("2026-03-30", periods=6, freq="h", tz="UTC"),
            "water_level_m": [0.2, 0.25, 0.21, 0.22, 0.3, 0.31],
        }
    )
    pred = pd.DataFrame(
        {
            "timestamp": pd.date_range("2026-03-30", periods=4, freq="6h", tz="UTC"),
            "predicted_tide_ft": [1.1, 5.0, 0.8, 5.2],
            "tide_type": ["L", "H", "L", "H"],
        }
    )

    report = compute_ocean_metrics(buoy, water, pred)

    assert report.metric_cards["latest_wave_height_m"] == 1.1
    assert report.tide_metrics["high_tide_count"] == 2
    assert report.confidence_note.startswith("High confidence")
