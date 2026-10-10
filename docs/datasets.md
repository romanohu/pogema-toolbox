# Pinned dataset preparation

`pogema_toolbox.datasets` prepares the pinned MovingAI and QD-MAPPER catalogs
without Neural-MAPF, Hydra or OmegaConf. Raw maps and scenarios remain external
caller-owned artifacts. The wheel and source distribution contain catalogs,
notices and the existing Cities Tiles asset; they do not contain the acquired
MovingAI or QD-MAPPER files.

Preparation is explicit. Parsers, scenario samplers and generators never acquire
missing data. A successful import verifies every selected archive and member
before atomically replacing the data directory. Failed verification preserves
existing data and its receipt. The returned `receipt` is also written to
`receipt.json`, including source identities, attribution and rights.

```python
from pathlib import Path
from pogema_toolbox.datasets import prepare_maps
from pogema_toolbox.moving_ai_ingestion import (
    parse_movingai_map, parse_movingai_scenarios, movingai_grid_config,
)
from pogema_toolbox.generators.scenario_generator import ScenarioSampler

prepared = prepare_maps("movingai", artifact_root="./artifacts")
data = Path(prepared["data_dir"])
map_name = "random-32-32-10"
obstacles = parse_movingai_map((data / (map_name + ".map")).read_text())
problems = parse_movingai_scenarios(
    (data / "scen-random" / (map_name + "-random-1.scen")).read_text(),
    map_name + ".map", obstacles,
)
replay = movingai_grid_config(obstacles, problems, 32, max_episode_steps=256)
sampled = ScenarioSampler(obstacles).generate(32, 42, sampling="uniform")
```

MovingAI may download the catalog's pinned URLs, or import caller-supplied ZIPs
with `staged_dir="/path/to/archives"`. Use `offline=True` to require local inputs.
ZIP filenames, byte counts, archive SHA-256 values and member hashes must match
the catalog. Unsafe paths, symlinks, duplicate destinations, extra members and
missing members are rejected. Original map cells, borders and source bytes are
preserved.

QD-MAPPER dataset permission remains `permission_unverified`. Automatic QD
acquisition is rejected before filesystem or network side effects. Supply
explicit local collection ZIPs or an extracted directory:

```python
from pogema_toolbox.generators.qd_mapper import load_qd_map

qd = prepare_maps(
    "qd_mapper", artifact_root="./artifacts",
    local_dir="/absolute/path/to/extracted-collections",
    collections=["CBS"], offline=True,
)
# Alternatively use staged_dir="/absolute/path/to/collection-zips".
qd_map = load_qd_map(Path(qd["data_dir"]) / "CBS/0_30.json")
replay = qd_map.grid_config(0, 32, max_episode_steps=256)
sampled = ScenarioSampler(qd_map.obstacles).generate(32, 42, sampling="uniform")
```

Extracted inputs must retain the catalog's member paths, such as
`CBS/0_30.json`; local ZIPs use catalog filenames, such as `CBS.zip`. Omitting
`collections` imports all QD collections in catalog order. Explicit selections
retain their supplied order and must be nonempty, unique known names. Local
receipts retain the original directory or ZIP path and `permission_unverified`;
verification does not establish redistribution permission.

The public API also includes `dataset_manifest(dataset, *, artifact_root=".",
manifest_path=None)`, `data_directory(dataset, *, artifact_root=".")` and
`selected_archives(manifest, collections=None)`. `prepare_maps` accepts these
keyword arguments: `artifact_root="."`, `collections=None`, `staged_dir=None`,
`local_dir=None`, `manifest_path=None`, `offline=False`, `timeout_seconds=30`.
Relative manifest and import paths are relative to `artifact_root`. Prepared
files use `<artifact_root>/maps/<dataset>/data`; explicit custom manifests must
use schema version 1 and the requested dataset name. `staged_dir` and
`local_dir` cannot be combined.

`prepare_maps("cities_tiles")` verifies the existing packaged asset and returns
its map count and SHA-256, without creating a raw data directory. Use the
[Cities Tiles generators](../README.md#cities-tiles) to generate scenarios.
See [scenario parsing and sampling](scenarios.md) for parser and sampler details.

Preserve the packaged [MovingAI attribution](../pogema_toolbox/maps/movingai/notices/ATTRIBUTION.md),
[Cities Tiles notices](../pogema_toolbox/maps/cities_tiles/notices/ATTRIBUTION.md)
and [QD rights notice](../pogema_toolbox/maps/qd_mapper/notices/RIGHTS.md).
Copied preparation code retains the exact
[Neural-MAPF MIT notice](../pogema_toolbox/licenses/Neural-MAPF-LICENSE).
These repository code licenses do not replace third-party dataset terms.
