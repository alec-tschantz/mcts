from jax import numpy as jnp, nn

from .buffers import Transition


def compute_returns(transition: Transition, steps=10, gamma=0.99, alpha=0.5):
    T = transition.reward.shape[0]
    out_returns = jnp.zeros_like(transition.reward)
    out_weight = jnp.zeros_like(transition.reward)

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


def entropy(probs):
    return float(-jnp.mean(jnp.sum(probs * jnp.log(probs + 1e-12), axis=-1)))


def scalar_to_support(x, support_size):
    x = _scaling(x)
    x = jnp.clip(x, -support_size, support_size)
    low = jnp.floor(x).astype(jnp.int32)
    high = jnp.ceil(x).astype(jnp.int32)
    prob_high = x - low
    prob_low = 1.0 - prob_high
    idx_low = low + support_size
    idx_high = high + support_size
    support_low = nn.one_hot(idx_low, 2 * support_size + 1) * prob_low[..., None]
    support_high = nn.one_hot(idx_high, 2 * support_size + 1) * prob_high[..., None]
    return support_low + support_high


def support_to_scalar(probs, support_size):
    x = jnp.sum((jnp.arange(2 * support_size + 1) - support_size) * probs, axis=-1)
    x = _inv_scaling(x)
    return x


def _scaling(x, eps: float = 1e-3):
    return jnp.sign(x) * (jnp.sqrt(jnp.abs(x) + 1) - 1) + eps * x


def _inv_scaling(x, eps: float = 1e-3):
    return jnp.sign(x) * (
        ((jnp.sqrt(1 + 4 * eps * (jnp.abs(x) + 1 + eps)) - 1) / (2 * eps)) ** 2 - 1
    )
