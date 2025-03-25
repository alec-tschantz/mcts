import optax
import equinox as eqx

from jax import Array
from jax import numpy as jnp
from jax import random as jr
from jax import tree_util as jtu
from jax import vmap, lax

from . import utils
from .buffer import Transition


class Policy(eqx.Module):
    value_head: eqx.nn.MLP
    policy_head: eqx.nn.MLP


def forward(policy: Policy, inp: Array) -> tuple[Array, Array]:
    inp = inp.reshape(-1)
    return policy.value_head(inp), policy.policy_head(inp)


def init_policy(
    feature_dim: int,
    action_dim: int,
    width: int,
    depth: int,
    key: jr.PRNGKey,
) -> Policy:
    return Policy(
        value_head=eqx.nn.MLP(
            in_size=feature_dim,
            out_size=1,
            width_size=width,
            depth=depth,
            key=key,
        ),
        policy_head=eqx.nn.MLP(
            in_size=feature_dim,
            out_size=action_dim,
            width_size=width,
            depth=depth,
            key=key,
        ),
    )
