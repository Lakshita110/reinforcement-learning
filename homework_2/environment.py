import numpy as np
import gymnasium as gym
import pygame

START = 0.8
END = 0.4


class RaceTrackEnv(gym.Env):
    metadata = {"render.modes": ["human"], "render_fps": 15}

    def __init__(self, map: str, render_mode: str = "human", size: int = 20):
        self.size = size

        assert map in ["a", "b"]
        assert render_mode in self.metadata["render.modes"] or render_mode is None

        self.render_mode = render_mode

        filename = "track_" + map + ".npy"

        self.track = np.load(filename)

        self.window_size = self.track.shape
        self.window_size = (
            self.window_size[1] * self.size,
            self.window_size[0] * self.size,
        )
        self.window = None
        self.clock = None
        self.truncated = False

        self.start_states = np.dstack(np.where(self.track == START))[0]

        self.finish_states = np.where(self.track == END)
        self.state = None  # [y, x] (row, col) is the position

        self.speed = None  # [y, x] (row, col) is the velocity

        self.nA = 9

        # (curr_row, curr_col, row_speed, col_speed)
        self.nS = (self.track.shape[0], self.track.shape[1], 5, 9)

        self._actions = {
            0: (-1, -1),
            1: (-1, 0),
            2: (-1, 1),
            3: (0, -1),
            4: (0, 0),
            5: (0, 1),
            6: (1, -1),
            7: (1, 0),
            8: (1, 1),
        }

    def _get_obs(self):
        return (*self.state, *self.speed)

    def _get_info(self):
        return None

    def _check_finish(self):
        rows = self.finish_states[0]
        col = self.finish_states[1][0]
        return self.state[0] in rows and self.state[1] >= col

    def _check_out_of_bounds(self, next_state):
        row, col = next_state
        H, W = self.track.shape

        # off the map
        if row < 0 or row >= H or col < 0 or col >= W:
            return True
        # off the track
        if self.track[next_state[0], next_state[1]] == 0:
            return True

        # check if the whole path is on the track
        for row_step in range(self.state[0], row, -1):
            if self.track[row_step, self.state[1]] == 0:
                return True
        for col_step in range(self.state[1], col, 1 if col > self.state[1] else -1):
            if self.track[row, col_step] == 0:
                return True

        return False

    def reset(self):
        start_state_idx = np.random.choice(self.start_states.shape[0])
        self.state = self.start_states[start_state_idx]

        self.speed = (0, 0)

        if self.render_mode == "human":
            self.render(self.render_mode)

        return self._get_obs(), self._get_info()

    def step(self, action: int, noise: bool = False):
        new_state = np.copy(self.state)
        y_action, x_action = self._actions[action]

        temp_y_speed = self.speed[0] + y_action
        temp_x_speed = self.speed[1] + x_action

        # velocity must be nonnegative and less than 5, and they cannot both be zero except at the starting line.
        if temp_y_speed > 0:  # means that the car ,is trying to go down
            temp_y_speed = 0
        if temp_y_speed < -4:
            temp_y_speed = -4
        if temp_x_speed < -4:
            temp_x_speed = -4
        if temp_x_speed > 4:
            temp_x_speed = 4
        if (temp_y_speed == 0 and temp_x_speed == 0 and
                self.state not in self.start_states):
            temp_y_speed = self.speed[0]
            temp_x_speed = self.speed[1]

        # If noise is True, with probability 0.1, the velocity changes to (0, 0)
        if noise and np.random.rand() < 0.1:
            temp_y_speed = 0
            temp_x_speed = 0

        new_state[0] += temp_y_speed
        new_state[1] += temp_x_speed

        terminated = False
        reset = False

        if self._check_finish():
            terminated = True
        elif self._check_out_of_bounds(new_state):
            self.reset()
            reset = True
        else:
            self.state = new_state
            self.speed = (temp_y_speed, temp_x_speed)

        if self.render_mode == "human":
            self.render(self.render_mode)

        return self._get_obs(), -1, terminated, reset

    def render(self, mode="human"):
        if self.window is None:
            pygame.init()
            pygame.display.set_caption("Race Track")
            if mode == "human":
                self.window = pygame.display.set_mode(self.window_size)

        if self.clock is None:
            self.clock = pygame.time.Clock()

        rows, cols = self.track.shape
        self.window.fill((255, 255, 255))

        # Draw the map
        for row in range(rows):
            for col in range(cols):
                cell_val = self.track[row, col]
                # Draw finishing cells
                if cell_val == END:
                    fill = (235, 52, 52)
                    pygame.draw.rect(
                        self.window,
                        fill,
                        (col * self.size, row * self.size, self.size, self.size),
                        0,
                    )
                # Draw starting cells
                elif cell_val == START:
                    fill = (61, 227, 144)
                    pygame.draw.rect(
                        self.window,
                        fill,
                        (col * self.size, row * self.size, self.size, self.size),
                        0,
                    )

                color = (120, 120, 120)
                # Draw gravels
                if cell_val == 0:
                    color = (255, 255, 255)
                # Draw race track
                elif cell_val == 1:
                    color = (160, 160, 160)

                pygame.draw.rect(
                    self.window,
                    color,
                    (col * self.size, row * self.size, self.size, self.size),
                    1,
                )

        # Draw the car
        pygame.draw.rect(
            self.window,
            (86, 61, 227),
            (
                self.state[1] * self.size,
                self.state[0] * self.size,
                self.size,
                self.size,
            ),
            0,
        )

        if mode == "human":
            pygame.display.update()
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.window = None
                    pygame.quit()
                    self.truncated = True
            self.clock.tick(self.metadata["render_fps"])


__all__ = ["RaceTrackEnv"]
