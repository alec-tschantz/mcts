import numpy as np
import gymnasium as gym
from gymnasium.spaces import Box
from .ram_annotations import atari_dict


def _lookup_game(env):
    game_id = env.unwrapped.spec.id
    for key in atari_dict.keys():
        if key in game_id.lower():
            return key
    raise KeyError(f"{game_id} did not match any ramdict currently implemented")


def _convert_pong(state):
    positions = np.array([state[0:2][::-1], state[2:4][::-1], state[4:6]])
    positions += np.array([[-48, -10]])
    return positions


def _convert_boxing(state):
    positions = np.array([state[0:2], state[2:4]])
    positions += np.array([[10, 60]])
    return positions


def _convert_bowling(state):
    ball = state[0:2]
    player = state[2:4]
    return np.array([player, ball])


def _convert_breakout(state):
    ball = state[0:2] + np.array([-48, 10])
    player = [state[2] - 40, 190]
    return np.array([player, ball])


def _convert_tennis(state):
    enemy = state[0:2]
    ball = state[3:5]
    player = state[6:8]
    return np.array([player, enemy, ball])


def _convert_freeway(state):
    positions = np.array(
        [
            [46, state[0]],
            [state[2], 20],
            [state[3], 36],
            [state[4], 52],
            [state[5], 68],
            [state[6], 84],
            [state[7], 100],
            [state[8], 116],
            [state[9], 132],
            [state[10], 148],
            [state[11], 164],
        ]
    )
    positions[:, 1] = 194 - positions[:, 1]
    return positions


configs = {
    "pong": {"num_objects": 3, "convert": _convert_pong},
    "boxing": {"num_objects": 2, "convert": _convert_boxing},
    "bowling": {"num_objects": 2, "convert": _convert_bowling},
    "breakout": {"num_objects": 2, "convert": _convert_breakout},
    "tennis": {"num_objects": 3, "convert": _convert_tennis},
    "freeway": {"num_objects": 11, "convert": _convert_freeway},
}


class OCWrapper(gym.ObservationWrapper):
    def __init__(self, env):
        super(OCWrapper, self).__init__(env)
        self.ram_name = env.unwrapped._game.lower()
        self.name = _lookup_game(env)
        num_objects = configs[self.name]["num_objects"]

        self.observation_space = gym.spaces.Box(
            -np.inf, np.inf, shape=(num_objects * 2,)
        )

    def reset(self, **kwargs):
        _, _ = self.env.reset(**kwargs)
        return self._observation(), {}

    def step(self, action):
        obs, reward, done, truncated, info = self.env.step(action)
        return self._observation(), reward, done, truncated, info

    def _observation(self) -> np.ndarray:
        ram = self.env.unwrapped.ale.getRAM()
        state = [ram[v] for v in atari_dict[self.name].values()]
        state = np.array(state, dtype=np.float32) 
        state = np.array(configs[self.name]["convert"](state))
        state = (state / 127.5) - 1
        return state.reshape(-1)
