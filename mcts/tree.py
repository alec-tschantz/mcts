import mctx
import optax
import equinox as eqx

from jax import Array
from jax import numpy as jnp
from jax import random as jr
from jax import vmap, lax, nn

from . import utils
from .buffer import Transition


def root_fn(
    model: eqx.Module, env: eqx.Module, obs: Array, key: jr.PRNGKey
) -> mctx.RootFnOutput:
    value_logits, policy_logits = vmap(model)(obs)
    value = value_logits.squeeze(-1)
    return mctx.RootFnOutput(prior_logits=policy_logits, value=value, embedding=obs)


def recurrent_fn(
    params: tuple[eqx.Module, eqx.Module],
    key: jr.PRNGKey,
    action: Array,
    obs: Array,
) -> tuple[mctx.RecurrentFnOutput, Array]:
    model, env = params
    batch_size = obs.shape[0]

    next_obs, reward, done = vmap(env.step)(
        obs, action.astype(jnp.int32), jr.split(key, batch_size)
    )
    value_logits, policy_logits = vmap(model)(next_obs)
    value = value_logits.squeeze(-1)
    discount = jnp.where(done, 0.0, 1.0)
    return (
        mctx.RecurrentFnOutput(
            reward=reward,
            discount=discount,
            prior_logits=policy_logits,
            value=value,
        ),
        next_obs,
    )


def plan_fn(
    obs: Array,
    model: eqx.Module,
    env: eqx.Module,
    key: jr.PRNGKey,
    max_depth: int = 20,
    gumbel_scale: float = 1.0,
    num_simulations: int = 500,
) -> tuple[int, Array, float]:
    root = root_fn(model, env, obs[None], key)
    out = mctx.gumbel_muzero_policy(
        params=(model, env),
        rng_key=key,
        root=root,
        recurrent_fn=recurrent_fn,
        num_simulations=num_simulations,
        max_depth=max_depth,
        gumbel_scale=gumbel_scale,
    )
    return out.action[0], out.action_weights[0], root.value[0]
