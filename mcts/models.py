import optax
import equinox as eqx

from jax import Array
from jax import numpy as jnp
from jax import random as jr
from jax import vmap, lax

from . import utils
from .buffer import Transition


class Policy(eqx.Module):
    value_head: eqx.nn.MLP
    policy_head: eqx.nn.MLP
    value_dim: int

    def __init__(
        self,
        input_dim: int,
        policy_dim: int,
        value_dim: int,
        width: int,
        depth: int,
        key: jr.PRNGKey,
    ):
        k1, k2 = jr.split(key)
        full_value_dim = 2 * value_dim + 1
        self.value_head = eqx.nn.MLP(input_dim, full_value_dim, width, depth, key=k1)
        self.policy_head = eqx.nn.MLP(input_dim, policy_dim, width, depth, key=k2)

    def __call__(self, inp: Array) -> tuple[Array, Array]:
        inp = inp.reshape(-1)
        return self.value_head(inp), self.policy_head(inp)


def loss_fn(model: eqx.Module, batch: Transition, env: eqx.Module, key: jr.PRNGKey):
    batch_size, traj_length = batch.action.shape
    initial_obs = batch.obs[:, 0]

    target_returns = utils.to_discrete(batch.returns, model.value_dim)

    def step(carry, t):
        total_loss, value_loss, policy_loss, obs, rng = carry

        value_logits, policy_logits = vmap(model)(obs)

        actions = batch.action[:, t]
        target_value = target_returns[:, t]
        target_policy = batch.action_probs[:, t]

        v_loss = jnp.mean(optax.softmax_cross_entropy(value_logits, target_value))
        pi_loss = jnp.mean(optax.softmax_cross_entropy(policy_logits, target_policy))

        rng, sk = jr.split(rng)
        keys = jr.split(sk, batch_size)
        next_obs, reward, done = vmap(env.step)(obs, actions, keys)

        new_carry = (
            total_loss + v_loss + pi_loss,
            value_loss + v_loss,
            policy_loss + pi_loss,
            next_obs,
            rng,
        )
        return new_carry, None

    (total_loss, value_loss, policy_loss, _, _), _ = lax.scan(
        step,
        (0.0, 0.0, 0.0, initial_obs, key),
        jnp.arange(traj_length),
    )

    l2_penalty = l2_loss(model)
    total_loss = total_loss / traj_length + 1e-4 * l2_penalty

    metrics = {
        "total_loss": total_loss,
        "value_loss": value_loss / traj_length,
        "policy_loss": policy_loss / traj_length,
        "l2_loss": 1e-4 * l2_penalty,
    }
    return total_loss, metrics


def l2_loss(model: eqx.Module) -> jnp.ndarray:
    return 0.5 * sum(
        jnp.sum(jnp.square(p))
        for p in jax.tree_util.tree_leaves(eqx.filter(model, eqx.is_array))
    )
