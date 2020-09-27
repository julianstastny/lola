"""
Coin Game environment.
"""
import gym
import numpy as np
import pdb
import time

from gym.spaces import Discrete, Tuple
from gym.spaces import prng


class CoinGameVec:
    """
    Vectorized Coin Game environment.
    Note: slightly deviates from the Gym API.
    """
    NUM_AGENTS = 2
    NUM_ACTIONS = 4
    MOVES = [
        np.array([0,  1]),
        np.array([0, -1]),
        np.array([1,  0]),
        np.array([-1, 0]),
    ]
    VIEWPORT_W = 400
    VIEWPORT_H = 400

    def __init__(self, max_steps, batch_size, grid_size=3):
        self.grid_size = grid_size
        self.observation_space = gym.spaces.Tuple([
            gym.spaces.Box(
                low=0,
                high=1,
                shape=(4, self.grid_size, self.grid_size),
                dtype='uint8'
            ), gym.spaces.Box(
                low=0,
                high=1,
                shape=(4, self.grid_size, self.grid_size),
                dtype='uint8'
            )
        ])

        self.max_steps = max_steps

        self.batch_size = batch_size
        # The 4 channels stand for 2 players and 2 coin positions
        self.ob_space_shape = [4, grid_size, grid_size]

        self.step_count = None

        self.viewer = None
        player_scale = 0.3
        self.PLAYER_SHAPE = [[[el[0] * player_scale,
                               el[1] * player_scale] for el in poly] for poly in self.PLAYER_SHAPE]
        coin_scale = 0.1
        self.COIN_SHAPE = [[[el[0] * coin_scale,
                             el[1] * coin_scale] for el in poly] for poly in self.COIN_SHAPE]
        self.cell_size = self.VIEWPORT_W // self.grid_size

        x_max = int(np.array([[el[0] for el in poly] for poly in self.PLAYER_SHAPE]).max())
        y_max = int(np.array([[el[1] for el in poly] for poly in self.PLAYER_SHAPE]).max())
        self.PLAYER_SHAPE = [[[x_max - el[0],
                               y_max - el[1]] for el in poly] for poly in self.PLAYER_SHAPE]
        self.render_every = 1
        self.epi = 0

    def reset(self):
        self.step_count = 0
        self.epi += 1
        self.red_coin = prng.np_random.randint(2, size=self.batch_size)
        # Agent and coin positions
        self.red_pos  = prng.np_random.randint(
            self.grid_size, size=(self.batch_size, 2))
        self.blue_pos = prng.np_random.randint(
            self.grid_size, size=(self.batch_size, 2))
        self.coin_pos = np.zeros((self.batch_size, 2), dtype=np.int8)
        for i in range(self.batch_size):
            # Make sure coins don't overlap
            while self._same_pos(self.red_pos[i], self.blue_pos[i]):
                self.blue_pos[i] = prng.np_random.randint(self.grid_size, size=2)
            self._generate_coin(i)

        self.observation = self._generate_observation()

        return self._generate_state()

    def _generate_coin(self, i):
        self.red_coin[i] = 1 - self.red_coin[i]
        # Make sure coin has a different position than the agent
        success = 0
        while success < 2:
            success = 0
            self.coin_pos[i] = prng.np_random.randint(self.grid_size, size=(2))
            success  = 1 - self._same_pos(self.red_pos[i],
                                          self.coin_pos[i])
            success += 1 - self._same_pos(self.blue_pos[i],
                                          self.coin_pos[i])

    def _generate_observation(self):
        # ToDo: Rendering first agent in batch; need logic to enforce batch_size=1 when rendering
        # observation = np.zeros(list(self.observation_space[0].shape))
        observation = np.zeros((4, 3, 3))
        observation[0, self.red_pos[0][0], self.red_pos[0][1]] = 1
        observation[1, self.blue_pos[0][0], self.blue_pos[0][1]] = 1
        if self.red_coin:
            observation[2, self.coin_pos[0][0], self.coin_pos[0][1]] = 1
        else:
            observation[3, self.coin_pos[0][0], self.coin_pos[0][1]] = 1
        observation = observation.astype(np.uint8)

        second_player_observation = observation[[1, 0, 3, 2], ...]

        observation = (observation, np.copy(second_player_observation))
        # observation = tuple([observation for i in range(self.NUM_AGENTS)])
        # print(observation)
        return observation

    def _same_pos(self, x, y):
        return (x == y).all()

    def _generate_state(self):
        state = np.zeros([self.batch_size] + self.ob_space_shape)
        for i in range(self.batch_size):
            state[i, 0, self.red_pos[i][0], self.red_pos[i][1]] = 1
            state[i, 1, self.blue_pos[i][0], self.blue_pos[i][1]] = 1
            if self.red_coin[i]:
                state[i, 2, self.coin_pos[i][0], self.coin_pos[i][1]] = 1
            else:
                state[i, 3, self.coin_pos[i][0], self.coin_pos[i][1]] = 1
        return state

    def step(self, actions):

        self.step_count += 1

        for j in range(self.batch_size):
            a0, a1 = actions[j]

            assert a0 in {0, 1, 2, 3} and a1 in {0, 1, 2, 3}

            # Move players
            self.red_pos[j] = \
                (self.red_pos[j] + self.MOVES[a0]) % self.grid_size
            self.blue_pos[j] = \
                (self.blue_pos[j] + self.MOVES[a1]) % self.grid_size

        # Compute rewards
        reward_red = np.zeros(self.batch_size)
        reward_blue = np.zeros(self.batch_size)
        for i in range(self.batch_size):
            generate = False
            if self.red_coin[i]:
                if self._same_pos(self.red_pos[i], self.coin_pos[i]):
                    generate = True
                    reward_red[i] += 2 # Set to 1 to make it symmetric again
                if self._same_pos(self.blue_pos[i], self.coin_pos[i]):
                    generate = True
                    reward_red[i] += -1 # Set to -2 to make it symmetric again
                    reward_blue[i] += 1
            else:
                if self._same_pos(self.red_pos[i], self.coin_pos[i]):
                    generate = True
                    reward_red[i] += 1
                    reward_blue[i] += -2
                if self._same_pos(self.blue_pos[i], self.coin_pos[i]):
                    generate = True
                    reward_blue[i] += 1

            if generate:
                self._generate_coin(i)

        reward = [reward_red, reward_blue]
        state = self._generate_state() 
        done = (self.step_count == self.max_steps)
        self.observation = self._generate_observation()
        return state, reward, done

    PLAYER_SHAPE = [
        [[94.67, 271.00], [102.58, 159.62], [126.67, 160.68], [140.75, 269.74], [144.92, 380.85], [190.33, 381.47],
         [190.42, 270.47], [222.50, 270.47], [224.67, 381.47], [264.96, 381.47], [269.25, 268.47], [271.83, 163.47],
         [301.00, 163.47], [308.67, 267.47], [327.17, 266.63], [320.00, 130.26], [234.83, 129.90], [247.67, 98.53],
         [243.33, 58.53], [199.67, 50.28], [197.00, 16.03], [191.17, 17.15], [186.33, 50.28], [155.67, 55.53],
         [151.33, 98.53], [160.42, 126.15], [79.50, 128.76], [68.58, 268.38]]]
    COIN_SHAPE = [
        [[69.67, 193.00], [69.58, 226.62], [86.50, 268.24], [116.33, 302.47], [159.67, 326.47], [200.00, 331.47],
         [237.33, 328.47], [266.67, 315.47], [293.33, 294.10], [312.00, 264.74], [327.67, 232.37], [333.33, 200.00],
         [331.17, 161.63], [313.00, 129.26], [293.83, 102.90], [266.67, 84.53], [236.33, 71.53], [198.00, 68.53],
         [163.67, 72.53], [133.33, 84.53], [113.42, 103.15], [94.50, 127.76], [81.58, 158.38]]]
    RED_COLOR = (255, 0, 0)
    BLUE_COLOR = (0, 0, 255)

    def draw_shape(self, shape, pos, color):
        x_delta, y_delta = pos
        shape_pos = [[[el[0] + x_delta,
                       el[1] + y_delta] for el in poly] for poly in shape]
        for polygon in shape_pos:
            self.viewer.draw_polygon(polygon, color=color)

    def render(self, mode='human'):

        # Set windows
        from gym.envs.classic_control import rendering

        if self.viewer is None:
            self.viewer = rendering.Viewer(self.VIEWPORT_W, self.VIEWPORT_H)
            self.viewer.set_bounds(0, self.VIEWPORT_W, 0, self.VIEWPORT_H)

        # if self.epi % self.render_every == 0:

        for object_idx in range(self.observation[0].shape[0]):
            if object_idx == 0:  # Red player
                color = self.RED_COLOR
                shape = self.PLAYER_SHAPE
            elif object_idx == 1:  # Blue player
                color = self.BLUE_COLOR
                shape = self.PLAYER_SHAPE
            elif object_idx == 2:  # Red coin
                color = self.RED_COLOR
                shape = self.COIN_SHAPE
            elif object_idx == 3:  # blue coin
                color = self.BLUE_COLOR
                shape = self.COIN_SHAPE

            x, y = np.nonzero(self.observation[0][object_idx])
            if len(x) > 0 and len(y) > 0:
                scale = self.cell_size
                x, y = int(x) * scale, int(y) * scale
                if object_idx % 2 == 1:
                    x += (self.cell_size // 2)
                self.draw_shape(shape=shape, pos=(x, y), color=color)

            time.sleep(0.25)
        return self.viewer.render(return_rgb_array=mode == 'rgb_array')

    def close(self):
        if self.viewer is not None:
            self.viewer.close()
            self.viewer = None

    def is_a_good_action(self, state, action):
        own_ini_pos_x, own_ini_pos_y = np.nonzero(state[0, ...])

        coin = state[2, ...] + state[3, ...]
        coin_pos_x, coin_pos_y = np.nonzero(coin)

        initial_dist = self.compute_dist(coin_pos_x, coin_pos_y, own_ini_pos_x, own_ini_pos_y)

        own_new_pos_x = (own_ini_pos_x + self.MOVES[action][0]) % self.grid_size
        own_new_pos_y = (own_ini_pos_y + self.MOVES[action][1]) % self.grid_size

        new_dist = self.compute_dist(coin_pos_x, coin_pos_y, own_new_pos_x, own_new_pos_y)
        # print(action)
        # print(state)
        # print("initial_dist, new_dist", initial_dist, new_dist)
        # print("is_a_good_action", new_dist < initial_dist)
        return new_dist < initial_dist

    def compute_dist(self, x1,y1,x2,y2):
        return abs(int(x1 - x2)) % 2 + abs(int(y1 - y2)) % 2


if __name__ == "__main__":
    env = CoinGameVec(max_steps=1000, batch_size=1)
    for _ in range(100):
        env.reset()
        d = False
        while not d:
            env.render()
            s, r, d = env.step([np.random.choice(2, size=2, replace=True)])
    env.viewer.close()
