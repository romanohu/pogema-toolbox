from dataclasses import dataclass
from typing import Optional

import numpy as np

from pogema_toolbox.generators.generator_utils import maps_dict_to_yaml


@dataclass
class WarehouseRangeSettings:
    size_min: int = 5
    size_max: int = 9

    wall_width_min: int = 5
    wall_width_max: int = 5

    wall_height_min: int = 2
    wall_height_max: int = 2

    side_pad: int = 1
    side_pad_min: Optional[int] = None
    side_pad_max: Optional[int] = None
    horizontal_gap: int = 1
    horizontal_gap_min: Optional[int] = None
    horizontal_gap_max: Optional[int] = None

    vertical_gap: int = 3
    vertical_gap_min: Optional[int] = None
    vertical_gap_max: Optional[int] = None

    num_wall_rows_min: Optional[int] = None
    num_wall_rows_max: Optional[int] = None

    num_wall_cols_min: Optional[int] = None
    num_wall_cols_max: Optional[int] = None

    wfi_instance: bool = False
    block_extra_space: bool = False

    def sample(self, seed=None):
        rng = np.random.default_rng(seed)

        wall_width = rng.integers(self.wall_width_min, self.wall_width_max + 1)
        wall_height = rng.integers(self.wall_height_min, self.wall_height_max + 1)

        side_pad = self.side_pad
        if self.side_pad_min is not None and self.side_pad_max is not None:
            side_pad = rng.integers(self.side_pad_min, self.side_pad_max + 1)

        horizontal_gap = self.horizontal_gap
        if self.horizontal_gap_min is not None and self.horizontal_gap_max is not None:
            horizontal_gap = rng.integers(
                self.horizontal_gap_min, self.horizontal_gap_max + 1
            )

        vertical_gap = self.vertical_gap
        if self.vertical_gap_min is not None and self.vertical_gap_max is not None:
            vertical_gap = rng.integers(self.vertical_gap_min, self.vertical_gap_max + 1)

        if self.num_wall_rows_min is not None:
            num_wall_rows = rng.integers(self.num_wall_rows_min, self.num_wall_rows_max + 1)
            num_wall_cols = rng.integers(self.num_wall_cols_min, self.num_wall_cols_max + 1)

            height = vertical_gap * (num_wall_rows + 1) + wall_height * num_wall_rows
            width = (
                side_pad * 2
                + wall_width * num_wall_cols
                + horizontal_gap * (num_wall_cols - 1)
            )
            size = max(width, height)
        else:
            size = rng.integers(self.size_min, self.size_max + 1)
            num_wall_rows = (size - vertical_gap) // (wall_height + vertical_gap)
            num_wall_cols = (size - side_pad * 2 + horizontal_gap) // (
                wall_width + horizontal_gap
            )

        return {
            "width": size,
            "height": size,
            "num_wall_rows": num_wall_rows,
            "num_wall_cols": num_wall_cols,
            "wall_width": wall_width,
            "wall_height": wall_height,
            "side_pad": side_pad,
            "horizontal_gap": horizontal_gap,
            "vertical_gap": vertical_gap,
            "wfi_instance": self.wfi_instance,
            "block_extra_space": self.block_extra_space,
            "seed": seed,
        }


def generate_warehouse(
    width,
    height,
    num_wall_rows,
    num_wall_cols,
    wall_width,
    wall_height,
    side_pad,
    horizontal_gap,
    vertical_gap,
    wfi_instance=False,
    block_extra_space=False,
    seed=None,
):
    grid = np.zeros((height, width), dtype=int)

    if not block_extra_space:
        max_wall_rows = (height - vertical_gap) // (wall_height + vertical_gap)
        max_wall_cols = (width - side_pad * 2 + horizontal_gap) // (wall_width + horizontal_gap)

        max_wall_rows = max(0, max_wall_rows)
        max_wall_cols = max(0, max_wall_cols)

        rows_to_place = max_wall_rows
        cols_to_place = max_wall_cols
    else:
        rows_to_place = num_wall_rows
        cols_to_place = num_wall_cols

    layout_height = vertical_gap * (rows_to_place + 1) + wall_height * rows_to_place
    layout_width = side_pad * 2
    if cols_to_place > 0:
        layout_width += wall_width * cols_to_place + horizontal_gap * (cols_to_place - 1)

    offset_y = max(0, (height - layout_height) // 2)
    offset_x = max(0, (width - layout_width) // 2)

    for row in range(rows_to_place):
        row_start = offset_y + vertical_gap * (row + 1) + wall_height * row
        for col in range(cols_to_place):
            col_start = offset_x + side_pad + col * (wall_width + horizontal_gap)
            grid[row_start:row_start + wall_height, col_start:col_start + wall_width] = 1

    if block_extra_space:
        blocked_grid = np.ones_like(grid)
        y_end = min(height, offset_y + layout_height)
        x_end = min(width, offset_x + layout_width)
        blocked_grid[offset_y:y_end, offset_x:x_end] = grid[offset_y:y_end, offset_x:x_end]
        grid = blocked_grid

    return grid


def generate_and_save_warehouses(name_prefix, seed_range, settings_generator=None):
    test_maps = {}
    max_digits = len(str(max(seed_range)))
    settings_generator = settings_generator or WarehouseRangeSettings()

    for seed in seed_range:
        settings = settings_generator.sample(seed)
        map_data = generate_warehouse(**settings)
        map_name = f"{name_prefix}-seed-{str(seed).zfill(max_digits)}"
        test_maps[map_name] = "\n".join(
            "".join("." if cell == 0 else "#" for cell in row) for row in map_data
        )

    maps_dict_to_yaml(f"{name_prefix}.yaml", test_maps)


def main():
    generate_and_save_warehouses("validation-warehouse", range(0, 128))
    generate_and_save_warehouses("training-warehouse", range(128, 128 + 512))


if __name__ == "__main__":
    main()
