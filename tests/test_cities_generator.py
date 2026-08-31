import hashlib
import json
from collections import Counter

import numpy as np


def _sha256(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _instance_fingerprint(observations) -> str:
    payload = {
        "obstacles": np.asarray(
            observations[0]["global_obstacles"], dtype=np.int8
        ).astype(int).tolist(),
        "starts": [
            [int(value) for value in item["global_xy"]] for item in observations
        ],
        "targets": [
            [int(value) for value in item["global_target_xy"]]
            for item in observations
        ],
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def test_cities_tiles_asset_matches_the_public_benchmark():
    from pogema_toolbox.generators.cities_generator import (
        CITY_NAMES,
        MAP_NAMES,
        load_cities_tiles,
    )

    maps = load_cities_tiles()

    assert tuple(maps) == MAP_NAMES
    assert len(maps) == 128
    assert Counter(name.rsplit("_", 1)[0] for name in maps) == {
        city: 16 for city in CITY_NAMES
    }
    assert all(len(grid.splitlines()) == 64 for grid in maps.values())
    assert all(
        len(row) == 64 and set(row) <= {".", "#"}
        for grid in maps.values()
        for row in grid.splitlines()
    )
    assert _sha256(maps["Berlin_1_256_00"]) == (
        "b7cfb717a31e8d63d85f95f9c765bad13759abe4b8f317bcff1618a7149d5462"
    )
    assert _sha256(maps["Boston_0_256_15"]) == (
        "4f51f8efb641930a26a18b1d5f561f418eaa6f7e59676f49e90af5f392a4c1d2"
    )
    assert _sha256(maps["Paris_2_256_15"]) == (
        "ef397c2ec0c1d8c0dc741605d0916eddab102706ba8915c0e9adb20e97bcb586"
    )
