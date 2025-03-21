# models/phys.py

import equinox as eqx
import jax.nn as nn
import jax.numpy as jnp
import jax.random as jr

from jax import lax, vmap


class Phys(eqx.Module):
    """
    Physics-inspired model with velocity scaling to help learning.
    """
    mlp: eqx.nn.MLP
    action_dim: int
    obs_shape: tuple[int, int]
    vel_scale: float

    def __init__(self, obs_shape, action_dim, key, vel_scale=20.0):
        """
        Args:
            obs_shape: e.g. (3, 4)
            action_dim: number of actions
            key: PRNGKey
            vel_scale: velocity scaling factor (default 20.0)
        """
        dim = obs_shape[0] * obs_shape[1]
        self.mlp = eqx.nn.MLP(
            in_size=dim + action_dim,
            out_size=dim,
            width_size=64,
            depth=2,
            key=key,
        )
        self.action_dim = action_dim
        self.obs_shape = obs_shape
        self.vel_scale = vel_scale

    def __call__(self, obs, action):
        """
        Forward pass:
        - Scales velocity
        - MLP predicts delta
        - Unscale velocity correction
        """
        obs_flat = obs.reshape(-1)
        action_onehot = nn.one_hot(action, self.action_dim)

        # Separate pos & vel
        pos = obs[..., 0:2]
        vel = obs[..., 2:4]

        vel_scaled = vel * self.vel_scale

        # Combine pos + scaled vel
        obs_scaled = jnp.concatenate([pos.reshape(-1), vel_scaled.reshape(-1)])
        inputs = jnp.concatenate([obs_scaled, action_onehot], axis=-1)

        # Predict delta
        delta = self.mlp(inputs).reshape(*self.obs_shape)

        dpos = delta[..., 0:2]
        dvel_scaled = delta[..., 2:4]
        dvel = dvel_scaled / self.vel_scale

        # Physics update
        new_pos = pos + vel + dpos
        new_pos = jnp.clip(new_pos, 0.0, 1.0)  # keep within bounds
        new_vel = vel + dvel

        return jnp.concatenate([new_pos, new_vel], axis=-1)


def loss_fn(model, batch, key):
    B, T = batch.action.shape

    def step(carry, t):
        total_loss, rng = carry

        obs = batch.obs[:, t]
        action = batch.action[:, t]
        next_obs = batch.obs[:, t + 1]

        pred = vmap(model)(obs, action)
        loss = jnp.mean((pred - next_obs) ** 2)

        return (total_loss + loss, rng), loss

    init = (0.0, key)
    (total_loss, _), _ = lax.scan(step, init, jnp.arange(T - 1))
    avg_loss = total_loss / T
    return avg_loss, {"train_loss": avg_loss}

@eqx.filter_jit
def rollout_fn(key, model, batch):
    init_obs = batch.obs[:, 0]
    B, T = batch.action.shape
    actions = batch.action.swapaxes(0, 1)

    def step(carry, action):
        obs, rng = carry
        pred_obs = vmap(model)(obs, action)
        return (pred_obs, rng), pred_obs

    _, pred_obs = lax.scan(step, (init_obs, key), actions)
    return pred_obs.swapaxes(0, 1)
