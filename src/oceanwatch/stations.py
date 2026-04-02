from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class StationCatalogEntry:
    key: str
    display_name: str
    coast: str
    ndbc_station_id: str
    coops_station_id: str
    latitude: float
    longitude: float
    validation_status: str


STATION_CATALOG: tuple[StationCatalogEntry, ...] = (
    StationCatalogEntry("sf_bay", "San Francisco Bay", "West Coast", "46026", "9414290", 37.8063, -122.4659, "validated"),
    StationCatalogEntry("monterey", "Monterey Bay", "West Coast", "46042", "9413450", 36.6052, -121.8889, "validated"),
    StationCatalogEntry("la_long_beach", "Los Angeles / Long Beach", "West Coast", "46222", "9410660", 33.7200, -118.2700, "validated"),
    StationCatalogEntry("seattle", "Seattle", "West Coast", "46087", "9447130", 47.6026, -122.3393, "validated"),
    StationCatalogEntry("honolulu", "Honolulu", "Pacific", "51001", "1612340", 21.3069, -157.8679, "validated"),
    StationCatalogEntry("boston", "Boston Harbor", "East Coast", "44013", "8443970", 42.3550, -71.0500, "validated"),
    StationCatalogEntry("new_york", "New York Harbor", "East Coast", "44065", "8518750", 40.7000, -74.0142, "validated"),
    StationCatalogEntry("chesapeake", "Chesapeake Bay", "East Coast", "44014", "8637689", 37.2110, -76.4780, "validated"),
    StationCatalogEntry("charleston", "Charleston", "East Coast", "41004", "8665530", 32.7800, -79.9300, "validated"),
    StationCatalogEntry("miami", "Miami", "Atlantic", "41009", "8723214", 25.7617, -80.1918, "validated"),
    StationCatalogEntry("key_west", "Key West", "Atlantic", "42003", "8724580", 24.5551, -81.7800, "validated"),
    StationCatalogEntry("galveston", "Galveston", "Gulf", "42040", "8771450", 29.3020, -94.7970, "validated"),
)


def list_station_choices() -> list[str]:
    return [f"{entry.display_name} ({entry.coast})" for entry in STATION_CATALOG]


def get_station_by_key(key: str) -> StationCatalogEntry:
    for entry in STATION_CATALOG:
        if entry.key == key:
            return entry
    raise KeyError(f"Unknown station key: {key}")


def get_station_by_label(label: str) -> StationCatalogEntry:
    for entry in STATION_CATALOG:
        if f"{entry.display_name} ({entry.coast})" == label:
            return entry
    raise KeyError(f"Unknown station label: {label}")
