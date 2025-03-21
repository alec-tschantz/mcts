import equinox as eqx

from jax import Array
from jax import numpy as jnp
from jax import random as jr
from jax import vmap, lax, nn

import mcts


class Test(eqx.Module):
    model: eqx.nn.MLP
    action_dim: int
    obs_shape: tuple[int, int]

    def __init__(self, obs_shape, action_dim, key):
        dim = obs_shape[0] * obs_shape[1]
        self.model = eqx.nn.MLP(
            in_size=dim + action_dim,
            out_size=dim,
            width_size=64,
            depth=2,
            key=key,
        )
        self.action_dim = action_dim
        self.obs_shape = obs_shape

    def __call__(self, obs, action):
        obs = obs.reshape(-1)
        action = nn.one_hot(action, num_classes=self.action_dim)
        inputs = jnp.concatenate([obs, action], axis=-1)
        return self.model(inputs)


def loss_fn(model, batch, key):
    B, T = batch.action.shape

    def step(carry, t):
        total_loss, rng = carry

        obs = batch.obs[:, t]
        action = batch.action[:, t]
        next_obs = batch.obs[:, t + 1]

        pred = vmap(model)(obs, action)
        pred = pred.reshape(B, *model.obs_shape)
        loss = jnp.mean((pred - next_obs) ** 2)

        return (total_loss + loss, rng), loss

    init = (0.0, key)
    (total_loss, _), losses = lax.scan(step, init, jnp.arange(T - 1))
    return total_loss / T, {"train_loss": total_loss / T}


@eqx.filter_jit
def rollout_fn(key, model, batch):
    init_obs = batch.obs[:, 0]
    B, T = batch.action.shape
    actions = batch.action.swapaxes(0, 1)

    def step(carry, action):
        obs, rng = carry
        pred_obs = vmap(model)(obs, action)
        pred_obs = pred_obs.reshape(B, *model.obs_shape)
        return (pred_obs, rng), pred_obs

    _, pred_obs = lax.scan(step, (init_obs, key), actions)
    return pred_obs.swapaxes(0, 1)
