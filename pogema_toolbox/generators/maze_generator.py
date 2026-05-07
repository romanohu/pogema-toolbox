import numpy as np
from dataclasses import dataclass

from .generator_utils import maps_dict_to_yaml


@dataclass
class MazeRangeSettings:
    width_min: int = 5
    width_max: int = 9

    height_min: int = 5
    height_max: int = 9

    obstacle_density_min: float = 0.0
    obstacle_density_max: float = 1.0

    wall_components_min: int = 1
    wall_components_max: int = 8

    go_straight_min: float = 0.75
    go_straight_max: float = 0.85

    def sample(self, seed=None):
        rng = np.random.default_rng(seed)
        width = rng.integers(self.width_min, self.width_max + 1)
        height = rng.integers(self.height_min, self.height_max + 1)
        obstacle_density = rng.uniform(self.obstacle_density_min, self.obstacle_density_max)
        wall_components = rng.integers(
            self.wall_components_min, self.wall_components_max + 1
        )
        go_straight = rng.uniform(self.go_straight_min, self.go_straight_max)
        return {
            "width": width,
            "height": height,
            "obstacle_density": obstacle_density,
            "wall_components": wall_components,
            "go_straight": go_straight,
            "seed": seed,
        }


class MazeGenerator:
    @classmethod
    def array_to_string(cls, array_maze):
        result = []
        for line in array_maze:
            result.append("".join(["#" if x == 1 else "." for x in line]))
        return "\n".join(result)

    @classmethod
    def string_to_array(cls, string_maze):
        lines = string_maze.split("\n")
        return np.array([[1 if char == "#" else 0 for char in line] for line in lines])

    @staticmethod
    def select_random_neighbor(
        x, y, maze_grid, maze_shape, rng, last_direction, go_straight
    ):
        neighbor_coords = []
        probabilities = []
        if x > 1:
            neighbor_coords.append((y, x - 2))
            probabilities.append(
                go_straight
                if (y, x - 2) == (y + last_direction[0], x + last_direction[1])
                else (1 - go_straight)
            )
        if x < maze_shape[1] - 2:
            neighbor_coords.append((y, x + 2))
            probabilities.append(
                go_straight
                if (y, x + 2) == (y + last_direction[0], x + last_direction[1])
                else (1 - go_straight)
            )
        if y > 1:
            neighbor_coords.append((y - 2, x))
            probabilities.append(
                go_straight
                if (y - 2, x) == (y + last_direction[0], x + last_direction[1])
                else (1 - go_straight)
            )
        if y < maze_shape[0] - 2:
            neighbor_coords.append((y + 2, x))
            probabilities.append(
                go_straight
                if (y + 2, x) == (y + last_direction[0], x + last_direction[1])
                else (1 - go_straight)
            )

        if not neighbor_coords:
            return None, None, last_direction

        if all(prob == go_straight for prob in probabilities):
            probabilities = [1 / len(probabilities)] * len(probabilities)
        else:
            total = sum(probabilities)
            probabilities = [prob / total for prob in probabilities]

        chosen_index = rng.choice(range(len(neighbor_coords)), p=probabilities)
        next_y, next_x = neighbor_coords[chosen_index]
        new_direction = (next_y - y, next_x - x)
        return next_x, next_y, new_direction

    @classmethod
    def generate_maze(
        cls, width, height, obstacle_density, wall_components, go_straight, seed=None
    ):
        rng = np.random.default_rng(seed)
        assert width > 0 and height > 0
        maze_shape = ((height // 2) * 2 + 3, (width // 2) * 2 + 3)
        density = (
            int(maze_shape[0] * maze_shape[1] * obstacle_density // wall_components)
            if wall_components != 0
            else 0
        )

        maze_grid = np.zeros(maze_shape, dtype="int")
        maze_grid[0, :] = maze_grid[-1, :] = 1
        maze_grid[:, 0] = maze_grid[:, -1] = 1

        for _ in range(density):
            x = rng.integers(0, maze_shape[1] // 2) * 2
            y = rng.integers(0, maze_shape[0] // 2) * 2
            maze_grid[y, x] = 1
            last_direction = (0, 0)
            for _ in range(wall_components):
                next_x, next_y, last_direction = cls.select_random_neighbor(
                    x, y, maze_grid, maze_shape, rng, last_direction, go_straight
                )
                if next_x is not None and maze_grid[next_y, next_x] == 0:
                    maze_grid[next_y, next_x] = 1
                    maze_grid[next_y + (y - next_y) // 2, next_x + (x - next_x) // 2] = 1
                    x, y = next_x, next_y

        return cls.array_to_string(maze_grid[1:-1, 1:-1])

    def generate_maze_from_ranges(self, seed=None):
        pass
