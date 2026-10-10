# Delivery and Sortation layouts

These are reusable, procedurally generated task layouts, not exact reproductions
of unspecified paper maps. They do not download datasets or create evaluation
runs. Existing Warehouse and Cities generator APIs are unchanged.

```python
from pogema_toolbox.generators.delivery_generator import (
    DeliveryRangeSettings, generate_delivery,
)
from pogema_toolbox.generators.sortation_generator import (
    SortationRangeSettings, generate_sortation,
)
from pogema_toolbox.generators.scenario_generator import ScenarioSampler

layout = generate_delivery(
    width=17, height=17, block_width=3, block_height=3, aisle_width=2, seed=7,
)
sampled_layout = generate_sortation(**SortationRangeSettings().sample(seed=7))
sampler = ScenarioSampler(layout.obstacles, layout.pickup_cells, layout.delivery_cells)
one_shot = sampler.generate(2, 42, sampling="endpoints", on_target="nothing")
lifelong = sampler.generate(
    8, 42, sampling="endpoints", on_target="restart", max_episode_steps=256,
)
```

Both functions return `TaskLayout`, defined in `scenario_generator`, containing
`obstacles` (a binary NumPy array), `pickup_cells`, `delivery_cells` (lists of
`(row, column)` tuples), and `map_seed` (the supplied seed). The array shape is
exactly `(height, width)`, including nonsquare maps. Free cells are `0` and
obstacles are `1`. All free cells form one four-neighbor component; both role
sets are nonempty, unique within each role, disjoint, and reachable. Dimensions,
spacing and endpoint counts must be integers excluding booleans. Map seeds must
be nonnegative integers excluding booleans, or `None` for an unseeded draw.
Reusing a seed and parameters reproduces the layout and endpoint list order.

## Delivery

`generate_delivery(width=17, height=17, block_width=3, block_height=3,
aisle_width=2, seed=None)` places complete rectangular obstacle blocks starting
at `(aisle_width, aisle_width)`. Blocks repeat with strides
`block_height + aisle_width` and `block_width + aisle_width`. There are at least
`aisle_width` free cells on every outer side and between blocks. Any remainder
stays free at the bottom/right; dimensions are never enlarged or truncated.

All dimensions and widths must be positive. A dimension must accommodate one
block plus two aisles: `width >= block_width + 2 * aisle_width` and
`height >= block_height + 2 * aisle_width`. Otherwise generation raises
`ValueError`. There is one pickup on the leftmost column and one delivery on
the rightmost column for every block row. The seed independently selects each
endpoint's row within that block row's vertical span. Explicit block geometry
is fixed by parameters; only endpoint choices vary with the map seed. A
one-cell block height leaves no endpoint-row choice.

## Sortation

`generate_sortation(width=17, height=17, bin_spacing=2, pickup_count=4,
delivery_count=4, seed=None)` uses its own bin-and-perimeter geometry, without
Warehouse block parameters. Bins are one-cell obstacles at rows
`range(1, height - 2, bin_spacing + 1)` and columns
`range(1, width - 1, bin_spacing + 1)`. Thus `bin_spacing` counts free cells
between neighboring bins. The last two rows stay free so every bin has an
interior free delivery cell immediately below it.

Width must be at least `3`, height at least `4`, and bin spacing and endpoint
counts must be positive. Pickups are sampled without replacement from the
outer perimeter, which has capacity `2 * width + 2 * height - 4`. Deliveries
are sampled without replacement from the cells immediately below bins; their
capacity equals the number of bins. Counts exceeding these capacities raise
`ValueError`. Every bin is placed even if only a subset receives a named
delivery endpoint. The seed selects pickup and delivery subsets and list order;
it never scatters unconstrained obstacles.

## Range sampling and scenarios

`DeliveryRangeSettings` has inclusive `width_min`/`width_max`,
`height_min`/`height_max`, `block_width_min`/`block_width_max`,
`block_height_min`/`block_height_max`, and `aisle_width_min`/`aisle_width_max`.
`SortationRangeSettings` instead has inclusive dimension ranges,
`bin_spacing_min`/`bin_spacing_max`, `pickup_count_min`/`pickup_count_max`, and
`delivery_count_min`/`delivery_count_max`. Defaults use dimensions `17..33`,
Delivery block dimensions `3..5` and aisles `2..3`, and Sortation spacing
`1..3` with pickup/delivery counts `2..8`.

`sample(seed)` returns function keyword arguments, including `seed`, with
ordinary Python integer values. Ranges must have positive integer bounds and
`min <= max`. Values are drawn independently, then the resulting geometry and
capacities are validated. An infeasible draw raises `ValueError`; the sampler
does not retry, resize dimensions, clamp spacing or reduce counts. Choose
ranges whose combinations are valid if every seed must succeed. An impossible
fixed range is always rejected. `sample(None)` is nondeterministic, and its
returned seed remains `None`.

Use `ScenarioSampler` for uniform free-cell or named-endpoint placement and
explicit lifelong targets. Endpoint one-shot capacity is the smaller role-set
size; lifelong starts may use all free cells, with shared station goals.
For direct Pogema restart replay, set
`env.unwrapped.current_goal_indices = [1] * num_agents` after each reset so the
already-active first target is consumed once. See
[scenario parsing and sampling](scenarios.md) for full scenario validation and
replay rules. Map and scenario seeds are separate inputs.
