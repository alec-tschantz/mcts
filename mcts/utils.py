from jax import numpy as jnp, nn

from .buffer import Transition


def entropy(probs):
    return float(-jnp.mean(jnp.sum(probs * jnp.log(probs + 1e-12), axis=-1)))


def compute_returns(transition: Transition, steps=10, gamma=0.99, alpha=0.5):
    T = transition.reward.shape[0]
    out_returns = jnp.zeros_like(transition.returns)
    out_weight = jnp.zeros_like(transition.weight)

    def compute(i):
        sum_r, discount, val_next, ended = 0.0, 1.0, 0.0, False
        for k in range(steps):
            idx = i + k
            if idx >= T:
                break
            sum_r += discount * transition.reward[idx]
            discount *= gamma
            if transition.done[idx]:
                ended = True
                break
        if not ended and (i + steps < T):
            val_next = transition.value[i + steps]
        nsr = sum_r + discount * val_next
        w = (
            abs(transition.value[i] - nsr) ** alpha
            if alpha is not None
            else transition.weight[i]
        )
        return nsr, w

    results = [compute(i) for i in range(T)]
    nsr_vals, weights = zip(*results)
    out_returns = jnp.array(nsr_vals)
    out_weight = jnp.array(weights)

    return Transition(
        obs=transition.obs,
        action=transition.action,
        reward=transition.reward,
        done=transition.done,
        returns=out_returns,
        value=transition.value,
        action_probs=transition.action_probs,
        weight=out_weight,
    )


def map_class_to_value(
    bin_idx: int, value_dim: int, value_min: float, value_max: float
) -> float:
    bin_width = (value_max - value_min) / float(value_dim)
    return value_min + (bin_idx + 0.5) * bin_width


def map_value_to_class(
    value: float, value_dim: int, value_min: float, value_max: float
) -> int:
    v_clamped = jnp.clip(value, value_min, value_max)
    bin_width = (value_max - value_min) / float(value_dim)
    idx = (v_clamped - value_min) // bin_width
    return jnp.minimum(idx.astype(int), value_dim - 1)


def map_reward_to_class(r, offset, dim):
    idx = jnp.clip(r + offset, 0, dim - 1)
    return idx.astype(int)
