"""
Coin Game environment.
"""
import gym
import numpy as np
from numba import njit, prange
from numba.typed import List

from gym.spaces import Discrete, Tuple
from gym.spaces import prng

@njit
def _same_pos(x, y):
    return (x == y).all()

@njit(parallel=True)
def move_players(batch_size, actions, red_pos, blue_pos, moves, grid_size):
    for j in prange(batch_size):
        # a0, a1 = actions[j]

        # assert a0 in {0, 1, 2, 3} and a1 in {0, 1, 2, 3}

        red_pos[j] = \
            (red_pos[j] + moves[actions[j, 0]]) % grid_size
        blue_pos[j] = \
            (blue_pos[j] + moves[actions[j, 1]]) % grid_size
    return red_pos, blue_pos

@njit(parallel=True)
def compute_reward(batch_size, red_pos, blue_pos, coin_pos, red_coin):
    reward_red = np.zeros(batch_size)
    reward_blue = np.zeros(batch_size)
    generate = np.zeros(batch_size, dtype=np.bool_)
    for i in prange(batch_size):
        if red_coin[i]:
            if _same_pos(red_pos[i], coin_pos[i]):
                generate[i] = True
                reward_red[i] += 2 # Set to 1 to make it symmetric again
            if _same_pos(blue_pos[i], coin_pos[i]):
                generate[i] = True
                reward_red[i] += -1 # Set to -2 to make it symmetric again
                reward_blue[i] += 1
        else:
            if _same_pos(red_pos[i], coin_pos[i]):
                generate[i] = True
                reward_red[i] += 1
                reward_blue[i] += -2
            if _same_pos(blue_pos[i], coin_pos[i]):
                generate[i] = True
                reward_blue[i] += 1
    reward = [reward_red, reward_blue]
    return reward, generate

@njit
def _flatten_index(pos, grid_size):
    y_pos, x_pos = pos
    idx = grid_size * y_pos
    idx += x_pos
    return idx

@njit
def _unflatten_index(pos, grid_size):
    x_idx = pos % grid_size
    y_idx = pos // grid_size
    return np.array([y_idx, x_idx])

@njit
def place_coin(red_pos_i, blue_pos_i, grid_size):
    red_pos_flat = _flatten_index(red_pos_i, grid_size)
    blue_pos_flat = _flatten_index(blue_pos_i, grid_size)
    possible_coin_pos = np.array([x for x in range(9) if ((x!= blue_pos_flat) and (x!=red_pos_flat))])
    flat_coin_pos = np.random.choice(possible_coin_pos)
    return _unflatten_index(flat_coin_pos, grid_size)


@njit(parallel=True)
def generate_coin(batch_size, generate, red_coin, red_pos, blue_pos, coin_pos, grid_size):
    red_coin[generate] = 1 - red_coin[generate]
    for i in prange(batch_size):
        if generate[i]:
            coin_pos[i] = place_coin(red_pos[i], blue_pos[i], grid_size)
    return coin_pos

@njit(parallel=True)
def generate_state(batch_size, red_pos, blue_pos, coin_pos, red_coin):
    state = np.zeros((batch_size, 4, 3, 3)) #TODO: Avoid hard coding this
    for i in prange(batch_size):
        state[i, 0, red_pos[i][0], red_pos[i][1]] = 1
        state[i, 1, blue_pos[i][0], blue_pos[i][1]] = 1
        if red_coin[i]:
            state[i, 2, coin_pos[i][0], coin_pos[i][1]] = 1
        else:
            state[i, 3, coin_pos[i][0], coin_pos[i][1]] = 1
    return state

@njit
def step(actions, batch_size, red_pos, blue_pos, coin_pos, red_coin, moves, grid_size):
    red_pos, blue_pos = move_players(batch_size, actions, red_pos, blue_pos, moves, grid_size)
    reward, generate = compute_reward(batch_size, red_pos, blue_pos, coin_pos, red_coin)
    coin_pos = generate_coin(batch_size, generate, red_coin, red_pos, blue_pos, coin_pos, grid_size)
    state = generate_state(batch_size, red_pos, blue_pos, coin_pos, red_coin)
    return red_pos, blue_pos, reward, coin_pos, state, red_coin


class CoinGameVec:
    """
    Vectorized Coin Game environment.
    Note: slightly deviates from the Gym API.
    """
    NUM_AGENTS = 2
    NUM_ACTIONS = 4
    MOVES = List([
        np.array([0,  1]),
        np.array([0, -1]),
        np.array([1,  0]),
        np.array([-1, 0]),
    ])

    def __init__(self, max_steps, batch_size, grid_size=3):
        self.max_steps = max_steps
        self.grid_size = grid_size
        self.batch_size = batch_size
        # The 4 channels stand for 2 players and 2 coin positions
        self.ob_space_shape = [4, grid_size, grid_size]

        self.step_count = None

    def reset(self):
        self.step_count = 0
        self.red_coin = np.random.randint(2, size=self.batch_size)
        # Agent and coin positions
        self.red_pos  = np.random.randint(
            self.grid_size, size=(self.batch_size, 2))
        self.blue_pos = np.random.randint(
            self.grid_size, size=(self.batch_size, 2))
        self.coin_pos = np.zeros((self.batch_size, 2), dtype=np.int8)
        for i in range(self.batch_size):
            # Make sure players don't overlap
            while _same_pos(self.red_pos[i], self.blue_pos[i]):
                self.blue_pos[i] = np.random.randint(self.grid_size, size=2)

        generate = np.ones(self.batch_size, dtype=bool)
        self.coin_pos = generate_coin(
            self.batch_size, generate, self.red_coin, self.red_pos, self.blue_pos, self.coin_pos, self.grid_size)
        return generate_state(self.batch_size, self.red_pos, self.blue_pos, self.coin_pos, self.red_coin)

    def step(self, actions):
        self.step_count += 1
        self.red_pos, self.blue_pos, reward, self.coin_pos, state, self.red_coin = step(
            actions, self.batch_size, self.red_pos, self.blue_pos, self.coin_pos, self.red_coin, self.MOVES, self.grid_size)
        done = (self.step_count == self.max_steps)
        return state, reward, done
