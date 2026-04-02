from __future__ import annotations

from datetime import datetime, timedelta, timezone
from io import StringIO
from typing import Any

import httpx
import numpy as np
import pandas as pd
from tenacity import Retrying, stop_after_attempt, wait_fixed

from oceanwatch.config import Settings
from oceanwatch.schemas import SourcePayload

NDBC_REALTIME_URL = "https://www.ndbc.noaa.gov/data/realtime2/{station_id}.txt"
COOPS_DATAGETTER_URL = "https://api.tidesandcurrents.noaa.gov/api/prod/datagetter"


class NOAAClient:
    def __init__(self, settings: Settings):
        self.settings = settings

    def _get_text(self, url: str, params: dict[str, Any] | None = None) -> str:
        for attempt in Retrying(
            stop=stop_after_attempt(self.settings.noaa_retry_attempts),
            wait=wait_fixed(self.settings.noaa_retry_wait_seconds),
            reraise=True,
        ):
            with attempt:
                response = httpx.get(url, params=params, timeout=self.settings.noaa_timeout_seconds)
                response.raise_for_status()
                return response.text
        raise RuntimeError("Unreachable retry state in _get_text")

    def _get_json(self, url: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
        for attempt in Retrying(
            stop=stop_after_attempt(self.settings.noaa_retry_attempts),
            wait=wait_fixed(self.settings.noaa_retry_wait_seconds),
            reraise=True,
        ):
            with attempt:
                response = httpx.get(url, params=params, timeout=self.settings.noaa_timeout_seconds)
                response.raise_for_status()
                payload = response.json()
                if not isinstance(payload, dict):
                    raise ValueError("Expected JSON object from NOAA endpoint")
                return payload
        raise RuntimeError("Unreachable retry state in _get_json")

    def fetch_ndbc_observations(self, station_id: str, hours_back: int) -> tuple[pd.DataFrame, SourcePayload]:
        text = self._get_text(NDBC_REALTIME_URL.format(station_id=station_id))
        frame = parse_ndbc_realtime_text(text)

        if not frame.empty:
            latest = frame["timestamp"].max()
            cutoff = latest - timedelta(hours=hours_back)
            frame = frame[frame["timestamp"] >= cutoff].copy()

        payload = SourcePayload(
            source_name="ndbc_buoy",
            station_id=station_id,
            available=not frame.empty,
            row_count=len(frame),
            records=_records_for_payload(frame),
            metadata={"units": "metric", "provider": "NOAA NDBC"},
            error=None if not frame.empty else "No buoy rows available for requested horizon.",
        )
        return frame, payload

    def fetch_coops_water_level(
        self,
        station_id: str,
        begin: datetime,
        end: datetime,
    ) -> tuple[pd.DataFrame, SourcePayload]:
        params = {
            "product": "water_level",
            "application": "OceanWatch",
            "begin_date": begin.strftime("%Y%m%d"),
            "end_date": end.strftime("%Y%m%d"),
            "datum": "MLLW",
            "station": station_id,
            "time_zone": "gmt",
            "units": "metric",
            "format": "json",
        }
        payload_json = self._get_json(COOPS_DATAGETTER_URL, params=params)
        frame, metadata = parse_coops_water_level_json(payload_json)

        payload = SourcePayload(
            source_name="coops_water_level",
            station_id=station_id,
            available=not frame.empty,
            row_count=len(frame),
            records=_records_for_payload(frame),
            metadata=metadata,
            error=None if not frame.empty else _coops_error_message(payload_json),
        )
        return frame, payload

    def fetch_coops_tide_predictions(
        self,
        station_id: str,
        begin: datetime,
        end: datetime,
    ) -> tuple[pd.DataFrame, SourcePayload]:
        params = {
            "product": "predictions",
            "application": "OceanWatch",
            "begin_date": begin.strftime("%Y%m%d"),
            "end_date": end.strftime("%Y%m%d"),
            "datum": "MLLW",
            "station": station_id,
            "time_zone": "gmt",
            "units": "english",
            "interval": "hilo",
            "format": "json",
        }
        payload_json = self._get_json(COOPS_DATAGETTER_URL, params=params)
        frame = parse_coops_predictions_json(payload_json)

        payload = SourcePayload(
            source_name="coops_tide_predictions",
            station_id=station_id,
            available=not frame.empty,
            row_count=len(frame),
            records=_records_for_payload(frame),
            metadata={"units": "english", "provider": "NOAA CO-OPS"},
            error=None if not frame.empty else _coops_error_message(payload_json),
        )
        return frame, payload


def parse_ndbc_realtime_text(text: str) -> pd.DataFrame:
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    header_line = next((line for line in lines if line.startswith("#YY")), None)
    if not header_line:
        return pd.DataFrame()

    columns = header_line.replace("#", "").split()
    data_lines = [line for line in lines if not line.startswith("#")]
    if not data_lines:
        return pd.DataFrame(columns=["timestamp"])  # pragma: no cover

    frame = pd.read_csv(StringIO("\n".join(data_lines)), sep=r"\s+", names=columns, engine="python")
    frame = frame.where(frame != "MM", np.nan)

    if {"YY", "MM", "DD", "hh", "mm"}.issubset(frame.columns):
        frame["timestamp"] = pd.to_datetime(
            frame[["YY", "MM", "DD", "hh", "mm"]].rename(
                columns={"YY": "year", "MM": "month", "DD": "day", "hh": "hour", "mm": "minute"}
            ),
            utc=True,
            errors="coerce",
        )
        frame = frame[frame["timestamp"].notna()].copy()

    numeric_fields = [
        "WDIR",
        "WSPD",
        "GST",
        "WVHT",
        "DPD",
        "APD",
        "MWD",
        "PRES",
        "ATMP",
        "WTMP",
        "DEWP",
        "VIS",
        "PTDY",
        "TIDE",
    ]
    for field in numeric_fields:
        if field in frame.columns:
            frame[field] = pd.to_numeric(frame[field], errors="coerce")

    renamed = {
        "WDIR": "wind_dir_deg",
        "WSPD": "wind_speed_mps",
        "GST": "gust_mps",
        "WVHT": "wave_height_m",
        "DPD": "dominant_period_s",
        "APD": "average_period_s",
        "MWD": "mean_wave_dir_deg",
        "PRES": "pressure_hpa",
        "ATMP": "air_temp_c",
        "WTMP": "water_temp_c",
        "DEWP": "dewpoint_c",
        "VIS": "visibility_nmi",
        "PTDY": "pressure_tendency_hpa",
        "TIDE": "tide_ft",
    }
    frame = frame.rename(columns=renamed)
    return frame.sort_values("timestamp").reset_index(drop=True)


def parse_coops_water_level_json(payload: dict[str, Any]) -> tuple[pd.DataFrame, dict[str, Any]]:
    rows = payload.get("data", [])
    if not isinstance(rows, list):
        rows = []

    frame = pd.DataFrame(rows)
    if frame.empty:
        return frame, payload.get("metadata", {})

    frame["timestamp"] = pd.to_datetime(frame.get("t"), utc=True, errors="coerce")
    frame["water_level_m"] = pd.to_numeric(frame.get("v"), errors="coerce")
    frame["sigma"] = pd.to_numeric(frame.get("s"), errors="coerce")
    frame["quality_flag"] = frame.get("q")

    keep_cols = ["timestamp", "water_level_m", "sigma", "quality_flag"]
    frame = frame[keep_cols].dropna(subset=["timestamp"]).sort_values("timestamp")
    return frame.reset_index(drop=True), payload.get("metadata", {})


def parse_coops_predictions_json(payload: dict[str, Any]) -> pd.DataFrame:
    rows = payload.get("predictions", [])
    if not isinstance(rows, list):
        rows = []

    frame = pd.DataFrame(rows)
    if frame.empty:
        return frame

    frame["timestamp"] = pd.to_datetime(frame.get("t"), utc=True, errors="coerce")
    frame["predicted_tide_ft"] = pd.to_numeric(frame.get("v"), errors="coerce")
    frame["tide_type"] = frame.get("type")

    keep_cols = ["timestamp", "predicted_tide_ft", "tide_type"]
    frame = frame[keep_cols].dropna(subset=["timestamp"]).sort_values("timestamp")
    return frame.reset_index(drop=True)


def _records_for_payload(frame: pd.DataFrame, max_rows: int = 240) -> list[dict[str, Any]]:
    if frame.empty:
        return []

    clipped = frame.tail(max_rows).copy()
    records = []
    for row in clipped.to_dict(orient="records"):
        record = {}
        for key, value in row.items():
            if isinstance(value, pd.Timestamp):
                record[key] = value.isoformat()
            elif isinstance(value, np.floating):
                record[key] = float(value)
            elif isinstance(value, np.integer):
                record[key] = int(value)
            else:
                record[key] = value
        records.append(record)
    return records


def _coops_error_message(payload_json: dict[str, Any]) -> str:
    error = payload_json.get("error")
    if isinstance(error, dict):
        message = error.get("message")
        if isinstance(message, str):
            return message
    return "CO-OPS did not return rows for the requested station/horizon."


def default_begin_end(hours_back: int) -> tuple[datetime, datetime]:
    end = datetime.now(timezone.utc)
    begin = end - timedelta(hours=hours_back)
    return begin, end
