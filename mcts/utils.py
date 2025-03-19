
from jax import numpy as jnp 

from .buffer import Transition

def compute_returns(transition: Transition, n=10, gamma=0.99, alpha=0.5):
    T = transition.reward.shape[0]
    out_returns = jnp.zeros_like(transition.returns)
    out_weight = jnp.zeros_like(transition.weight)

    def compute(i):
        sum_r, discount, val_next, ended = 0.0, 1.0, 0.0, False
        for k in range(n):
            idx = i + k
            if idx >= T:
                break
            sum_r += discount * transition.reward[idx]
            discount *= gamma
            if transition.done[idx]:
                ended = True
                break
        if not ended and (i + n < T):
            val_next = transition.value[i + n]
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
