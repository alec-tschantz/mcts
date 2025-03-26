import optax
import equinox as eqx
from jax import Array, numpy as jnp, random as jr, nn


class Policy(eqx.Module):
    value_head: eqx.nn.MLP
    policy_head: eqx.nn.MLP
    value_dim: int
    value_min: float
    value_max: float


def forward(policy: Policy, inp: Array) -> tuple[Array, Array]:
    inp = inp.reshape(-1)
    value_logits = policy.value_head(inp)
    policy_logits = policy.policy_head(inp)
    return value_logits, policy_logits


def init_policy(
    feature_dim: int,
    action_dim: int,
    width: int,
    depth: int,
    key: jr.PRNGKey,
    value_dim: int = 8,
    value_min: float = -1.0,
    value_max: float = 1.0,
) -> Policy:
    k_val, k_pol = jr.split(key, 2)
    return Policy(
        value_head=eqx.nn.MLP(
            in_size=feature_dim,
            out_size=value_dim,
            width_size=width,
            depth=depth,
            key=k_val,
        ),
        policy_head=eqx.nn.MLP(
            in_size=feature_dim,
            out_size=action_dim,
            width_size=width,
            depth=depth,
            key=k_pol,
        ),
        value_dim=value_dim,
        value_min=value_min,
        value_max=value_max,
    )
