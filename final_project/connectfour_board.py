import numpy as np
import gymnasium as gym
import pygame
import random


class ConnectFourBoard(gym.Env):
    meta_data = {"render_modes": ["human"], "render_fps": 15}

    def __init__(self, render_mode: str = None):

        assert (
            render_mode in self.meta_data["render_modes"] or render_mode is None)

        super(ConnectFourBoard, self).__init__()
        self.rows = 6
        self.columns = 7
        self.board = np.zeros((self.rows, self.columns), dtype=int)
        self.action_space = gym.spaces.Discrete(self.columns)
        self.observation_space = gym.spaces.Box(
            low=-1, high=1, shape=(self.rows, self.columns), dtype=int)
        self.render_mode = render_mode

        self.current_player = 1
        self.done = False
        self.winner = None

        if render_mode == "human":
            pygame.init()
            self.window_size = (self.columns * 100, self.rows * 100)
            self.window = pygame.display.set_mode(self.window_size)
            pygame.display.set_caption("Connect 4")
            self.clock = pygame.time.Clock()
            self.font = pygame.font.Font(None, 36)

            self.render_board()

    def render_board(self):
        if self.render_mode != "human":
            return

        # Fill background with blue for the board
        self.window.fill((0, 0, 255))

        for r in range(self.rows):
            for c in range(self.columns):
                # Draw the board holes
                pygame.draw.rect(self.window, (0, 0, 255),
                                 (c * 100, r * 100, 100, 100))
                color = (0, 0, 0)  # Default to black (empty)
                if self.board[r][c] == 1:
                    color = (255, 0, 0)  # Player 1 (red)
                elif self.board[r][c] == -1:
                    color = (255, 255, 0)  # Player -1 (yellow)
                pygame.draw.circle(self.window, color,
                                   (c * 100 + 50, r * 100 + 50), 40)

        # If game is done, show message
        if self.done:
            if self.winner == 0:
                text = self.font.render("Draw!", True, (255, 255, 255))
            else:
                color = "Red" if self.winner == 1 else "Yellow"
                text = self.font.render(
                    f"{color} wins!", True, (255, 255, 255))
            self.window.blit(text, (10, 10))

        pygame.display.flip()
        self.clock.tick(self.meta_data["render_fps"])

    def reset(self):
        self.board = np.zeros((self.rows, self.columns), dtype=int)
        self.current_player = 1
        self.done = False
        self.winner = None

        if self.render_mode == "human":
            self.render_board()

        return (self.board.copy(), {"winner": None})

    def get_available_actions(self):
        return [c for c in range(self.columns) if self.board[0][c] == 0]

    def check_game_done(self):
        # Check horizontal, vertical, and diagonal for a win
        for r in range(self.rows):
            for c in range(self.columns - 3):
                if self.board[r, c] == self.current_player and \
                   self.board[r, c + 1] == self.current_player and \
                   self.board[r, c + 2] == self.current_player and \
                   self.board[r, c + 3] == self.current_player:
                    self.done = True
                    self.winner = self.current_player
                    return

        for r in range(self.rows - 3):
            for c in range(self.columns):
                if self.board[r, c] == self.current_player and \
                   self.board[r + 1, c] == self.current_player and \
                   self.board[r + 2, c] == self.current_player and \
                   self.board[r + 3, c] == self.current_player:
                    self.done = True
                    self.winner = self.current_player
                    return

        for r in range(self.rows - 3):
            for c in range(self.columns - 3):
                if self.board[r, c] == self.current_player and \
                   self.board[r + 1, c + 1] == self.current_player and \
                   self.board[r + 2, c + 2] == self.current_player and \
                   self.board[r + 3, c + 3] == self.current_player:
                    self.done = True
                    self.winner = self.current_player
                    return

        for r in range(3, self.rows):
            for c in range(self.columns - 3):
                if self.board[r, c] == self.current_player and \
                   self.board[r - 1, c + 1] == self.current_player and \
                   self.board[r - 2, c + 2] == self.current_player and \
                   self.board[r - 3, c + 3] == self.current_player:
                    self.done = True
                    self.winner = self.current_player
                    return

        # Check for a draw
        if all(self.board[0, c] != 0 for c in range(self.columns)):
            self.done = True
            self.winner = 0  # Indicate a draw

    def step(self, action):
        if self.done:
            raise Exception("Game is already done. Please reset.")

        if action not in self.get_available_actions():
            raise ValueError("Invalid action.")

        row = self.get_next_open_row(action)
        self.board[row][action] = self.current_player

        self.check_game_done()

        reward = 1 if self.winner == self.current_player else 0
        info = {"winner": self.winner}
        self.render_board()
        self.current_player *= -1
        return (self.board.copy(), reward, self.done, info)

    def get_next_open_row(self, column):
        for r in range(self.rows - 1, -1, -1):
            if self.board[r][column] == 0:
                return r
        return None
