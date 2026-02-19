from dataclasses import dataclass

import numpy as np

from pogema_toolbox.generators.generator_utils import maps_dict_to_yaml


@dataclass
class RoomRangeSettings:
    room_width_min: int = 5
    room_width_max: int = 9

    room_height_min: int = 5
    room_height_max: int = 9

    num_rows_min: int = 3
    num_rows_max: int = 5

    num_cols_min: int = 3
    num_cols_max: int = 5

    obstacle_density_min: float = 0.0
    obstacle_density_max: float = 0.4

    uniform: bool = True
    only_centre_obstacles: bool = False

    def sample(self, seed=None):
        rng = np.random.default_rng(seed)

        room_height = rng.integers(self.room_height_min, self.room_height_max + 1)
        num_rows = rng.integers(self.num_rows_min, self.num_rows_max + 1)

        if self.uniform:
            room_width = room_height
            num_cols = num_rows
        else:
            room_width = rng.integers(self.room_width_min, self.room_width_max + 1)
            num_cols = rng.integers(self.num_cols_min, self.num_cols_max + 1)

        obstacle_density = rng.uniform(self.obstacle_density_min, self.obstacle_density_max)

        return {
            "room_width": room_width,
            "room_height": room_height,
            "num_rows": num_rows,
            "num_cols": num_cols,
            "obstacle_density": obstacle_density,
            "only_centre_obstacles": self.only_centre_obstacles,
            "seed": seed,
        }


def generate_room(
    room_width,
    room_height,
    num_rows,
    num_cols,
    obstacle_density,
    only_centre_obstacles=False,
    seed=None,
):
    rng = np.random.default_rng(seed)

    room = np.zeros(
        (room_height * num_rows + num_rows - 1, room_width * num_cols + num_cols - 1),
        dtype="int",
    )

    obs_prob = rng.uniform(0, 1, size=room.shape)
    room[obs_prob < obstacle_density] = 1

    if only_centre_obstacles:
        room[0:: room_height + 1, :] = 0
        room[:, 0:: room_width + 1] = 0
        room[room_height - 1:: room_height + 1, :] = 0
        room[:, room_width - 1:: room_width + 1] = 0

    room[room_height:: room_height + 1, :] = 1
    room[:, room_width:: room_width + 1] = 1

    row_doors = rng.integers(low=0, high=room_width, size=(num_rows - 1, num_cols))
    offset = np.arange(num_cols) * (room_width + 1)
    offset = np.expand_dims(offset, axis=0)
    row_doors = offset + row_doors
    np.put_along_axis(room[room_height:: room_height + 1, :], row_doors, 0, axis=1)

    col_doors = rng.integers(low=0, high=room_height, size=(num_rows, num_cols - 1))
    offset = np.arange(num_rows) * (room_height + 1)
    offset = np.expand_dims(offset, axis=1)
    col_doors = offset + col_doors
    np.put_along_axis(room[:, room_width:: room_width + 1], col_doors, 0, axis=0)

    return room


class RoomGenerator:
    @staticmethod
    def generate(**kwargs):
        room = generate_room(**kwargs)
        return "\n".join("".join("." if cell == 0 else "#" for cell in row) for row in room)


def generate_and_save_room_maps(name_prefix, seed_range, settings_generator=None):
    test_maps = {}
    max_digits = len(str(max(seed_range)))
    settings_generator = settings_generator or RoomRangeSettings()

    for seed in seed_range:
        settings = settings_generator.sample(seed)
        map_data = RoomGenerator.generate(**settings)
        map_name = f"{name_prefix}-seed-{str(seed).zfill(max_digits)}"
        test_maps[map_name] = map_data

    maps_dict_to_yaml(f"{name_prefix}.yaml", test_maps)


def main():
    generate_and_save_room_maps("validation-room", range(0, 128))
    generate_and_save_room_maps("training-room", range(128, 128 + 512))


if __name__ == "__main__":
    main()
