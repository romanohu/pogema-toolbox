import numpy as np

from dataclasses import dataclass

from pogema_toolbox.generators.generator_utils import maps_dict_to_yaml

@dataclass
class RoomRangeSettings:
    room_size_min: int = 5
    room_size_max: int = 10
    rooms_x_min: int = 2
    rooms_x_max: int = 4
    rooms_y_min: int = 2
    rooms_y_max: int = 4
    door_size: int = 1

    def sample(self, seed=None):
        rng = np.random.default_rng(seed)
        return {
            "room_size": int(rng.integers(self.room_size_min, self.room_size_max + 1)),
            "rooms_x": int(rng.integers(self.rooms_x_min, self.rooms_x_max + 1)),
            "rooms_y": int(rng.integers(self.rooms_y_min, self.rooms_y_max + 1)),
            "door_size": self.door_size,
            "seed": seed,
        }
    
class RoomGenerator:
    @staticmethod
    def generate(room_size, rooms_x, rooms_y, door_size, seed=None):
        rng = np.random.default_rng(seed)
        WALL_THICK = 1

        height = (rooms_y * room_size) + ((rooms_y - 1) * WALL_THICK)
        width = (rooms_x * room_size) + ((rooms_x - 1) * WALL_THICK)
        grid = np.zeros((height, width), dtype=int)

        for c in range(rooms_x - 1):
            x_wall = (c + 1) * room_size + c * WALL_THICK
            grid[:, x_wall : x_wall + WALL_THICK] = 1
            
        for r in range(rooms_y - 1):
            y_wall = (r + 1) * room_size + r * WALL_THICK
            grid[y_wall : y_wall + WALL_THICK, :] = 1

        for c in range(rooms_x - 1):
            x_wall = (c + 1) * room_size + c * WALL_THICK
            for r in range(rooms_y):
                y_room_start = r * (room_size + WALL_THICK)
                possible_y = rng.integers(y_room_start, y_room_start + room_size - door_size + 1)
                grid[possible_y : possible_y + door_size, x_wall : x_wall + WALL_THICK] = 0

        for r in range(rooms_y - 1):
            y_wall = (r + 1) * room_size + r * WALL_THICK
            for c in range(rooms_x):
                x_room_start = c * (room_size + WALL_THICK)
                possible_x = rng.integers(x_room_start, x_room_start + room_size - door_size + 1)
                grid[y_wall : y_wall + WALL_THICK, possible_x : possible_x + door_size] = 0

        return '\n'.join(''.join('.' if cell == 0 else '#' for cell in row) for row in grid)

def generate_and_save_room_maps(name_prefix, seed_range):
    test_maps = {}
    max_digits = len(str(max(seed_range)))
    settings_generator = RoomRangeSettings()

    for seed in seed_range:
        settings = settings_generator.sample(seed)
        map_data = RoomGenerator.generate(**settings)
        map_name = f"{name_prefix}-seed-{str(seed).zfill(max_digits)}"
        test_maps[map_name] = map_data
    maps_dict_to_yaml(f'{name_prefix}.yaml', test_maps)
    print(f"Saved {len(test_maps)} maps to {name_prefix}.yaml")

def main():
    generate_and_save_room_maps("validation-room", range(0, 128))
    generate_and_save_room_maps("training-room", range(128, 128 + 512))

if __name__ == "__main__":
    main()