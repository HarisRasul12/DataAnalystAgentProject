from __future__ import annotations

from oceanwatch.noaa_clients import (
    parse_coops_predictions_json,
    parse_coops_water_level_json,
    parse_ndbc_realtime_text,
)


def test_parse_ndbc_realtime_text_handles_missing_and_timestamps() -> None:
    sample = """#YY MM DD hh mm WDIR WSPD GST WVHT DPD APD MWD PRES ATMP WTMP DEWP VIS PTDY TIDE
2026 03 30 00 30 310 7.0 8.0 MM MM MM MM 1013.4 MM 13.2 MM MM MM MM
2026 03 30 00 20 300 6.0 7.0 1.1 13 6.4 218 1013.5 MM 13.1 MM MM MM MM
"""

    frame = parse_ndbc_realtime_text(sample)

    assert not frame.empty
    assert "timestamp" in frame.columns
    assert frame["wave_height_m"].isna().sum() >= 1
    assert frame["water_temp_c"].notna().sum() == 2


def test_parse_coops_water_level_json() -> None:
    payload = {
        "metadata": {"id": "9414290", "name": "San Francisco"},
        "data": [
            {"t": "2026-03-30 00:00", "v": "0.621", "s": "0.062", "q": "p"},
            {"t": "2026-03-30 00:06", "v": "0.659", "s": "0.065", "q": "p"},
        ],
    }

    frame, metadata = parse_coops_water_level_json(payload)

    assert len(frame) == 2
    assert frame["water_level_m"].iloc[0] == 0.621
    assert metadata["id"] == "9414290"


def test_parse_coops_predictions_json() -> None:
    payload = {
        "predictions": [
            {"t": "2026-03-30 04:22", "v": "1.343", "type": "L"},
            {"t": "2026-03-30 10:25", "v": "5.302", "type": "H"},
        ]
    }

    frame = parse_coops_predictions_json(payload)

    assert len(frame) == 2
    assert set(frame["tide_type"].tolist()) == {"L", "H"}
