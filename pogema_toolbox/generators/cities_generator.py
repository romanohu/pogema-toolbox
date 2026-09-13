from __future__ import annotations

import hashlib
from collections.abc import Sequence
from dataclasses import dataclass
from functools import lru_cache
from importlib.resources import path

import numpy as np
import yaml
from pogema import GridConfig

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
OFFICIAL_AGENT_COUNTS = (64, 128, 192, 256)
MAX_EPISODE_STEPS = 256


@lru_cache(maxsize=None)
def _placement_capacity(grid: str) -> int:
    rows = grid.splitlines()
    seen = set()
    capacity = 0
    for row_index, row in enumerate(rows):
        for column_index, symbol in enumerate(row):
            start = (row_index, column_index)
            if symbol != "." or start in seen:
                continue
            seen.add(start)
            stack = [start]
            component_size = 0
            while stack:
                current_row, current_column = stack.pop()
                component_size += 1
                for next_row, next_column in (
                    (current_row - 1, current_column),
                    (current_row + 1, current_column),
                    (current_row, current_column - 1),
                    (current_row, current_column + 1),
                ):
                    next_cell = (next_row, next_column)
                    if (
                        0 <= next_row < len(rows)
                        and 0 <= next_column < len(rows[next_row])
                        and rows[next_row][next_column] == "."
                        and next_cell not in seen
                    ):
                        seen.add(next_cell)
                        stack.append(next_cell)
            capacity += component_size // 2
    return capacity


def load_cities_tiles() -> dict[str, str]:
    with path("pogema_toolbox.maps", "cities-tiles.yaml") as asset:
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


@dataclass(frozen=True)
class CitiesTilesInstance:
    grid_config: GridConfig
    city: str
    tile_index: int
    map_name: str
    num_agents: int
    scenario_seed: int


class CitiesTilesGenerator:
    def __init__(
        self,
        *,
        mode: str = "official",
        map_name: str | None = None,
        num_agents: Sequence[int] = OFFICIAL_AGENT_COUNTS,
        num_samples: int | None = None,
        seed: int = 0,
    ):
        self.mode = mode
        self.map_name = map_name
        self.num_agents = tuple(num_agents)
        self.num_samples = num_samples
        self.seed = seed

    @classmethod
    def official(cls):
        return cls(mode="official")

    @classmethod
    def single(cls, **kwargs):
        return cls(mode="single", **kwargs)

    @classmethod
    def random(cls, **kwargs):
        return cls(mode="random", **kwargs)

    def generate(self) -> list[CitiesTilesInstance]:
        maps = load_cities_tiles()
        self._validate(maps)
        if self.mode == "official":
            selections = [
                (name, agents, 0)
                for name in MAP_NAMES
                for agents in OFFICIAL_AGENT_COUNTS
            ]
        else:
            rng = np.random.default_rng(self.seed)
            selections = []
            for _ in range(self.num_samples):
                name = (
                    self.map_name
                    if self.mode == "single"
                    else MAP_NAMES[int(rng.integers(len(MAP_NAMES)))]
                )
                agents = self.num_agents[int(rng.integers(len(self.num_agents)))]
                scenario_seed = int(rng.integers(np.iinfo(np.int32).max))
                selections.append((name, agents, scenario_seed))
        records = [self._instance(maps, *selection) for selection in selections]
        keys = {(item.map_name, item.num_agents, item.scenario_seed) for item in records}
        if self.mode == "official" and (len(records) != 512 or len(keys) != 512):
            raise RuntimeError("official mode did not generate 512 unique instances")
        return records

    def _instance(self, maps, map_name, num_agents, scenario_seed):
        city, tile = map_name.rsplit("_", 1)
        capacity = _placement_capacity(maps[map_name])
        if capacity < num_agents:
            raise ValueError(
                f"mode={self.mode} map_name={map_name} num_agents={num_agents} "
                f"scenario_seed={scenario_seed}: placement capacity {capacity}"
            )
        config = GridConfig(
            map=maps[map_name],
            num_agents=int(num_agents),
            seed=int(scenario_seed),
            max_episode_steps=MAX_EPISODE_STEPS,
            obs_radius=5,
            observation_type="MAPF",
            collision_system="soft",
            on_target="nothing",
        )
        return CitiesTilesInstance(
            grid_config=config,
            city=city,
            tile_index=int(tile),
            map_name=map_name,
            num_agents=int(num_agents),
            scenario_seed=int(scenario_seed),
        )

    def _validate(self, maps: dict[str, str]) -> None:
        if self.mode not in {"official", "single", "random"}:
            raise ValueError(f"unsupported cities-tiles mode: {self.mode!r}")
        if (
            isinstance(self.seed, bool)
            or not isinstance(self.seed, int)
            or self.seed < 0
        ):
            raise ValueError(
                f"mode={self.mode} seed={self.seed!r}: "
                "seed must be a non-negative integer"
            )
        valid_agents = bool(self.num_agents) and all(
            not isinstance(value, bool)
            and isinstance(value, (int, np.integer))
            and int(value) > 0
            for value in self.num_agents
        )
        if not valid_agents:
            raise ValueError("num_agents must contain positive integers")
        if self.mode == "official":
            if self.map_name is not None:
                raise ValueError("official mode does not accept map_name")
            if self.num_samples is not None:
                raise ValueError("official mode does not accept num_samples")
            if self.seed != 0:
                raise ValueError("official mode requires seed=0")
            if self.num_agents != OFFICIAL_AGENT_COUNTS:
                raise ValueError("official mode requires the official num_agents")
            self._validate_capacities(maps, MAP_NAMES)
            return
        if (
            isinstance(self.num_samples, bool)
            or not isinstance(self.num_samples, int)
            or self.num_samples <= 0
        ):
            raise ValueError(f"{self.mode} mode requires positive num_samples")
        if self.mode == "single" and self.map_name not in maps:
            raise ValueError(f"single mode requires a known map_name: {self.map_name!r}")
        if self.mode == "random" and self.map_name is not None:
            raise ValueError("random mode does not accept map_name")
        names = (self.map_name,) if self.mode == "single" else MAP_NAMES
        self._validate_capacities(maps, names)

    def _validate_capacities(self, maps: dict[str, str], map_names) -> None:
        for map_name in map_names:
            capacity = _placement_capacity(maps[map_name])
            for num_agents in self.num_agents:
                if capacity < num_agents:
                    raise ValueError(
                        f"mode={self.mode} map_name={map_name} "
                        f"num_agents={num_agents}: placement capacity {capacity}"
                    )
