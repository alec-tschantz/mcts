import optax
import equinox as eqx
from jax import Array, numpy as jnp, random as jr, nn


class Policy(eqx.Module):
    value_head: eqx.nn.MLP
    policy_head: eqx.nn.MLP


def forward(policy: Policy, inp: Array) -> tuple[Array, Array]:
    value_logits = policy.value_head(inp)
    policy_logits = policy.policy_head(inp)
    return value_logits, policy_logits


def init_policy(
    feature_dim: int,
    action_dim: int,
    value_dim: int,
    policy_width: int,
    policy_depth: int,
    value_width: int,
    value_depth: int,
    key: jr.PRNGKey,
) -> Policy:
    k_val, k_pol = jr.split(key, 2)
    return Policy(
        value_head=eqx.nn.MLP(
            in_size=feature_dim,
            out_size=value_dim,
            width_size=value_width,
            depth=value_depth,
            key=k_val,
        ),
        policy_head=eqx.nn.MLP(
            in_size=feature_dim,
            out_size=action_dim,
            width_size=policy_width,
            depth=policy_depth,
            key=k_pol,
        ),
    )
