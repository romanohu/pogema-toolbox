from dataclasses import dataclass

import numpy as np

from pogema_toolbox.generators.generator_utils import maps_dict_to_yaml


@dataclass
class WarehouseRangeSettings:
    wall_width_min: int = 3
    wall_width_max: int = 8
    wall_height_min: int = 1
    wall_height_max: int = 3
    walls_in_row_min: int = 4
    walls_in_row_max: int = 6
    walls_rows_min: int = 4
    walls_rows_max: int = 6
    bottom_gap_min: int = 3
    bottom_gap_max: int = 7
    horizontal_gap_min: int = 1
    horizontal_gap_max: int = 3
    vertical_gap_min: int = 2
    vertical_gap_max: int = 4

    def sample(self, seed=None):
        rng = np.random.default_rng(seed)
        return {
            "wall_width": int(rng.integers(self.wall_width_min, self.wall_width_max + 1)),
            "wall_height": int(rng.integers(self.wall_height_min, self.wall_height_max + 1)),
            "walls_in_row": int(rng.integers(self.walls_in_row_min, self.walls_in_row_max + 1)),
            "walls_rows": int(rng.integers(self.walls_rows_min, self.walls_rows_max + 1)),
            "bottom_gap": int(rng.integers(self.bottom_gap_min, self.bottom_gap_max + 1)),
            "horizontal_gap": int(rng.integers(self.horizontal_gap_min, self.horizontal_gap_max + 1)),
            "vertical_gap": int(rng.integers(self.vertical_gap_min, self.vertical_gap_max + 1)),
        }


def generate_warehouse(
    wall_width=5,
    wall_height=2,
    walls_in_row=5,
    walls_rows=5,
    bottom_gap=5,
    horizontal_gap=1,
    vertical_gap=3,
):
    if isinstance(wall_width, dict):
        return generate_warehouse(**wall_width)

    height = vertical_gap * (walls_rows + 1) + wall_height * walls_rows
    width = bottom_gap * 2 + wall_width * walls_in_row + horizontal_gap * (walls_in_row - 1)

    grid = np.zeros((height, width), dtype=int)

    for row in range(walls_rows):
        row_start = vertical_gap * (row + 1) + wall_height * row
        for col in range(walls_in_row):
            col_start = bottom_gap + col * (wall_width + horizontal_gap)
            grid[row_start:row_start + wall_height, col_start:col_start + wall_width] = 1

    return '\n'.join(''.join('!' if cell == 0 else '#' for cell in row) for row in grid)


def generate_wfi_positions(grid_str, bottom_gap, vertical_gap):
    if vertical_gap == 1:
        raise ValueError("Cannot generate WFI instance with vertical_gap of 1.")

    grid = [list(row) for row in grid_str.strip().split('\n')]
    height = len(grid)
    width = len(grid[0])

    start_locations = []
    goal_locations = []

    for row in range(1, height - 1):
        if row % 3 == 0:
            continue
        for col in range(bottom_gap - 1):
            if grid[row][col] == '!':
                start_locations.append((row, col))
        for col in range(width - bottom_gap + 1, width):
            if grid[row][col] == '!':
                start_locations.append((row, col))

    if vertical_gap == 2:
        for row in range(1, height):
            for col in range(width):
                if grid[row][col] == '!' and grid[row - 1][col] == '#':
                    goal_locations.append((row, col))
    else:
        for row in range(height):
            for col in range(width):
                if grid[row][col] == '!':
                    if (row > 0 and grid[row - 1][col] == '#') or (row < height - 1 and grid[row + 1][col] == '#'):
                        goal_locations.append((row, col))

    return start_locations, goal_locations


def generate_wfi_warehouse(
    wall_width=5,
    wall_height=2,
    walls_in_row=5,
    walls_rows=5,
    bottom_gap=5,
    horizontal_gap=1,
    vertical_gap=3,
):
    if isinstance(wall_width, dict):
        return generate_wfi_warehouse(**wall_width)

    grid = generate_warehouse(
        wall_width=wall_width,
        wall_height=wall_height,
        walls_in_row=walls_in_row,
        walls_rows=walls_rows,
        bottom_gap=bottom_gap,
        horizontal_gap=horizontal_gap,
        vertical_gap=vertical_gap,
    )
    start_locations, goal_locations = generate_wfi_positions(grid, bottom_gap, vertical_gap)
    grid_list = [list(row) for row in grid.split('\n')]

    for s in start_locations:
        grid_list[s[0]][s[1]] = '@'
    for s in goal_locations:
        if grid_list[s[0]][s[1]] == '$':
            grid_list[s[0]][s[1]] = '!'
        else:
            grid_list[s[0]][s[1]] = '$'
    str_grid = '\n'.join([''.join(row) for row in grid_list])

    return str_grid


def generate_and_save_warehouses(name_prefix, seed_range, settings_generator=None, wfi=True):
    test_maps = {}
    max_digits = len(str(max(seed_range)))
    settings_generator = settings_generator or WarehouseRangeSettings()

    for seed in seed_range:
        settings = settings_generator.sample(seed)
        map_data = generate_wfi_warehouse(**settings) if wfi else generate_warehouse(**settings)
        map_name = f"{name_prefix}-seed-{str(seed).zfill(max_digits)}"
        test_maps[map_name] = map_data

    maps_dict_to_yaml(f'{name_prefix}.yaml', test_maps)


def main():
    generate_and_save_warehouses("validation-warehouse", range(0, 128))
    generate_and_save_warehouses("training-warehouse", range(128, 128 + 512))


if __name__ == '__main__':
    main()
