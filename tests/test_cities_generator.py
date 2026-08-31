import hashlib
import json
from collections import Counter

import numpy as np
import pytest
from pogema import pogema_v0


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


def test_cities_generator_imports_without_importlib_resources_files():
    import importlib
    import importlib.resources
    import sys

    resources_module = importlib.resources
    files = getattr(resources_module, "files", None)
    sys.modules.pop("pogema_toolbox.generators.cities_generator", None)
    if files is None:
        module = importlib.import_module("pogema_toolbox.generators.cities_generator")
    else:
        del resources_module.files
        try:
            module = importlib.import_module(
                "pogema_toolbox.generators.cities_generator"
            )
        finally:
            resources_module.files = files

    assert module.CITY_NAMES[0] == "Berlin_1_256"


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


def _key(instance):
    return instance.map_name, instance.num_agents, instance.scenario_seed


def test_official_mode_is_the_fixed_512_instance_protocol():
    from pogema_toolbox.generators import cities_generator

    instances = cities_generator.CitiesTilesGenerator.official().generate()
    assert len(instances) == 512
    assert len({_key(instance) for instance in instances}) == 512
    assert [_key(instance) for instance in instances[:4]] == [
        ("Berlin_1_256_00", 64, 0),
        ("Berlin_1_256_00", 128, 0),
        ("Berlin_1_256_00", 192, 0),
        ("Berlin_1_256_00", 256, 0),
    ]
    assert all(instance.grid_config.max_episode_steps == 256 for instance in instances)
    assert all(instance.grid_config.obs_radius == 5 for instance in instances)
    assert all(instance.grid_config.observation_type == "MAPF" for instance in instances)
    assert all(instance.grid_config.collision_system == "soft" for instance in instances)
    assert all(instance.grid_config.on_target == "nothing" for instance in instances)
    assert all(instance.grid_config.seed == 0 for instance in instances)


def test_single_mode_stays_on_one_map_and_is_deterministic():
    from pogema_toolbox.generators import cities_generator

    kwargs = dict(
        map_name="Berlin_1_256_07",
        num_agents=(64, 128),
        num_samples=12,
        seed=42,
    )
    left = cities_generator.CitiesTilesGenerator.single(**kwargs).generate()
    right = cities_generator.CitiesTilesGenerator.single(**kwargs).generate()
    assert [_key(item) for item in left] == [_key(item) for item in right]
    assert {item.map_name for item in left} == {"Berlin_1_256_07"}


def test_random_mode_samples_all_fields_deterministically():
    from pogema_toolbox.generators import cities_generator

    kwargs = dict(num_agents=(64, 128), num_samples=300, seed=42)
    left = cities_generator.CitiesTilesGenerator.random(**kwargs).generate()
    right = cities_generator.CitiesTilesGenerator.random(**kwargs).generate()
    assert [_key(item) for item in left] == [_key(item) for item in right]
    assert {item.map_name for item in left} <= set(cities_generator.MAP_NAMES)
    assert len({item.map_name for item in left}) > 1
    assert len({item.num_agents for item in left}) == 2
    assert len({item.map_name for item in left}) < len(left)
    for left_item, right_item in zip(left[:3], right[:3]):
        left_obs, _ = pogema_v0(grid_config=left_item.grid_config).reset()
        right_obs, _ = pogema_v0(grid_config=right_item.grid_config).reset()
        assert _instance_fingerprint(left_obs) == _instance_fingerprint(right_obs)


@pytest.mark.parametrize(
    "kwargs, message",
    [
        ({"mode": "bad"}, "mode"),
        ({"mode": "single", "map_name": None, "num_samples": 1}, "map_name"),
        ({"mode": "single", "map_name": "unknown", "num_samples": 1}, "map_name"),
        ({"mode": "random", "map_name": "Berlin_1_256_00", "num_samples": 1}, "map_name"),
        ({"mode": "random", "num_samples": 0}, "num_samples"),
        ({"mode": "official", "seed": 1}, "seed"),
        ({"mode": "official", "num_samples": 1}, "num_samples"),
        ({"mode": "random", "num_samples": 1, "num_agents": ()}, "num_agents"),
    ],
)
def test_invalid_generator_arguments_fail_before_rollout(kwargs, message):
    from pogema_toolbox.generators import cities_generator

    with pytest.raises(ValueError, match=message):
        cities_generator.CitiesTilesGenerator(**kwargs).generate()


def test_generated_grid_config_resets_a_real_environment():
    from pogema_toolbox.generators import cities_generator

    instance = cities_generator.CitiesTilesGenerator.single(
        map_name="Berlin_1_256_00",
        num_agents=(64,),
        num_samples=1,
        seed=7,
    ).generate()[0]
    env = pogema_v0(grid_config=instance.grid_config)
    observations, _ = env.reset()
    starts = [tuple(item["global_xy"]) for item in observations]
    goals = [tuple(item["global_target_xy"]) for item in observations]
    rows = instance.grid_config.map.splitlines()
    assert len(starts) == len(set(starts)) == 64
    assert len(goals) == len(set(goals)) == 64
    assert all(rows[row][col] == "." for row, col in starts)
    assert all(rows[row][col] == "." for row, col in goals)


@pytest.mark.parametrize(
    "map_name, num_agents, expected",
    [
        (
            "Berlin_1_256_00",
            64,
            "72b6f575d18799c52e3cebca1418e2203b1f3f1524dcc8d30e4fc0d4972a127d",
        ),
        (
            "Boston_0_256_15",
            128,
            "eedef4c3db44cdd57a4c9f1b25c04789cda171596cd11038caf78cd0b1305589",
        ),
        (
            "Paris_2_256_15",
            256,
            "bc007884c4febbc0ac93c0c37777c907421d8b3e2c43708d0186df6989341810",
        ),
    ],
)
def test_official_seed_zero_instances_match_public_fixtures(
    map_name, num_agents, expected
):
    from pogema_toolbox.generators import cities_generator

    item = next(
        instance
        for instance in cities_generator.CitiesTilesGenerator.official().generate()
        if instance.map_name == map_name and instance.num_agents == num_agents
    )
    observations, _ = pogema_v0(grid_config=item.grid_config).reset()
    assert item.scenario_seed == 0
    assert _instance_fingerprint(observations) == expected
