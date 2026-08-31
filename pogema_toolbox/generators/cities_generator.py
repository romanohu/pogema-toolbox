from __future__ import annotations

import hashlib
from importlib.resources import files

import yaml

CITY_NAMES = (
    "Berlin_1_256",
    "Boston_0_256",
    "London_2_256",
    "Milan_0_256",
    "Moscow_0_256",
    "NewYork_1_256",
    "Paris_1_256",
    "Paris_2_256",
)
MAP_NAMES = tuple(f"{city}_{tile:02d}" for city in CITY_NAMES for tile in range(16))
ASSET_SHA256 = "3d357b87c64bab6a08a8ec43cfe81295ed37fd4a14db36d563e4ae599785664f"


def load_cities_tiles() -> dict[str, str]:
    asset = files("pogema_toolbox").joinpath("maps/cities-tiles.yaml")
    raw = asset.read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    if digest != ASSET_SHA256:
        raise ValueError(f"cities-tiles asset hash mismatch: {digest}")
    document = yaml.safe_load(raw)
    if not isinstance(document, dict) or tuple(document) != MAP_NAMES:
        raise ValueError("cities-tiles asset does not contain the expected 128 maps")
    maps = {str(name): str(grid) for name, grid in document.items()}
    for name, grid in maps.items():
        rows = grid.splitlines()
        if len(rows) != 64 or any(len(row) != 64 for row in rows):
            raise ValueError(f"{name}: expected a 64 x 64 grid")
        if any(set(row) - {".", "#"} for row in rows):
            raise ValueError(f"{name}: grid contains unsupported symbols")
    return maps
