# Scenario parsing and sampling

These APIs read local data and produce explicit Pogema starts and targets.
Parsing, generation and replay do not download files. The strict MovingAI APIs
preserve every original map cell, border and dimension. They accept octile maps
containing only `.` (free), `@` and `T` (blocked), and reject inconsistent headers,
truncated/transposed grids and unsupported symbols. The legacy `map_to_grid`
importer retains its existing `remove_border=True` default.

```python
from pathlib import Path
from pogema_toolbox.moving_ai_ingestion import (
    parse_movingai_map, parse_movingai_scenarios, movingai_grid_config,
)

obstacles = parse_movingai_map(Path("fixture.map").read_text())
problems = parse_movingai_scenarios(
    Path("fixture.map.scen").read_text(), "fixture.map", obstacles,
)
config = movingai_grid_config(obstacles, problems, 32, max_episode_steps=256)
```

`MovingAIProblem` records retain `map_name`, `bucket`, `distance`, `start` and
`goal` in literal source order. Coordinates are always `(row, column)`:
MovingAI `(x=2, y=0)` becomes `(0, 2)`. Scenario dimensions and map names must
match, coordinates must be in-bounds free cells, distances must be finite and
nonnegative, and every pair must be connected through four-neighbor free cells.
`movingai_grid_config` takes exactly the first `num_agents` rows; it never
reorders or substitutes rows. Starts must be unique and goals must be unique
within the prefix. One agent's start may equal another agent's goal. Prefix
replay supports `on_target="nothing"` (default) or `"finish"`. A bounded cache
of four layouts reuses component labels across scenario files and prefix sizes.

```python
from pogema_toolbox.generators.scenario_generator import ScenarioSampler

sampler = ScenarioSampler(obstacles)
config = sampler.generate(32, 42, sampling="uniform", on_target="nothing")
lifelong = sampler.generate(
    32, 42, sampling="uniform", on_target="restart", max_episode_steps=256,
)
```

A sampler copies its binary grid and computes components once. Reusing it with
the same count, seed and options returns the same explicit positions and target
sequences. Counts and horizons must be positive integers; seeds must be
nonnegative integers smaller than `sys.maxsize`. Booleans and nonintegers are
rejected for counts, seeds, horizons and coordinates. Components label blocked
cells `-1` and connected free cells with nonnegative integers.

Uniform sampling selects free starts and targets within the same connected
component. Starts and initial goals are each unique, and every initial goal
differs from its own start. The sets may overlap, so a component of `n >= 2`
free cells can hold `n` agents. Singleton components cannot provide a nontrivial
pair and are excluded. Placement uses bounded sampling and repairs equal pairs;
full-capacity placement does not rely on retries. Uniform restart sequences
remain in their agent's component and never repeat the immediately preceding
target.

```python
sampler = ScenarioSampler(
    obstacles, pickup_cells=[(0, 0)], delivery_cells=[(0, 3)],
)
config = sampler.generate(1, 42, sampling="endpoints", on_target="nothing")
lifelong = sampler.generate(
    4, 42, sampling="endpoints", on_target="restart", max_episode_steps=256,
)
```

Endpoint lists must be supplied together, nonempty, free, in-bounds and unique
within each named role. The example requires a connected free grid containing
all four first-row cells. Endpoint one-shot sampling uses unique pickup starts
and unique delivery goals, with each own start/goal distinct and reachable.
Its capacity is the sum of `min(pickups, deliveries)` per eligible component.
Endpoint restart sampling places unique starts across all free cells of
components supporting both roles. Fleets may therefore exceed station counts.
Their explicit goals start with pickup and alternate delivery/pickup thereafter;
station goals may be shared by multiple agents. An initial start may equal its
first pickup goal. Successive goals always differ: an endpoint shared with a
singleton opposite role is excluded from that role's restart targets if it
would leave no valid next transition. Components without a distinct pickup and
delivery pair are excluded.

Every restart agent receives exactly `max_episode_steps + 1` targets. No target
needs to be generated during replay. Supported protocols are `nothing`,
`finish` and `restart`; invalid protocols or insufficient placement capacity
raise `ValueError`. Additional `GridConfig` options such as observation radius
and collision system may be supplied, but explicit positions, map dimensions,
binary encodings and counts cannot be overridden through `grid_options`.

The pinned Pogema custom-sequence implementation initializes its goal cursor at
zero although target zero is already active. Neural-MAPF's evaluator sets
`env.unwrapped.current_goal_indices = [1] * num_agents` after reset, so each
arrival advances to the next supplied target. Direct Pogema callers wanting
that same consumption order should apply this cursor adjustment after each
reset. Without it, Pogema repeats target zero on the first arrival. These APIs
return `GridConfig` objects and do not modify Pogema behavior.

## Local QD-MAPPER scenarios

```python
from pogema_toolbox.generators.qd_mapper import load_qd_map

qd_map = load_qd_map("research-map.json")
config = qd_map.grid_config(0, 32, max_episode_steps=256)
```

`load_qd_map` reads a local unweighted QD-MAPPER JSON file containing `name`,
`weight: false`, `n_row`, `n_col`, `layout`, `start` and `goal`. It preserves
all declared rows and columns, including nonsquare layouts. Layout symbols are
`.` (free) and `@` (blocked); weighted maps and other symbols are rejected.
Linear source indices use row-major order: `row = index // n_col` and
`column = index % n_col`. Dimensions and indices must be integers, excluding
booleans; all indices must refer to in-bounds free cells.

The returned `QDMap` exposes `obstacles`, `name`, `source_hash` (SHA-256 of the
original file bytes), and `problems`, an ordered list of stored scenarios.
Each scenario contains source-ordered `MovingAIProblem` records with `start`
and `goal` coordinates, `map_name` equal to the JSON name, and `bucket` equal
to the zero-based scenario index. `distance` is `None` because the source
provides no distances. Starts and goals must be unique within their own roles
in every stored scenario, and each pair must be four-neighbor reachable.
Cross-role overlap and an agent starting at its own goal are allowed.

`grid_config(scenario_index, num_agents, **options)` replays exactly the first
`num_agents` pairs of the selected scenario. Both arguments are integer indices
or counts, excluding booleans; the scenario index must be in range and the
positive agent count cannot exceed that scenario's stored pairs. Replay uses
`on_target="nothing"` by default and also supports `"finish"`. Map, positions,
dimensions and counts cannot be overridden through options. Loading and replay
use local JSON only, without upstream optimization imports or archive
unpickling. Research ZIPs are inputs for local verification and are not bundled
with the toolbox.
