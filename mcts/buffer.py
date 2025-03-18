from typing import List
import equinox as eqx
from jax import Array, numpy as jnp, random as jr


class Transition(eqx.Module):
    obs: Array
    action: Array
    reward: Array
    done: Array
    returns: Array
    value: Array
    action_probs: Array
    weight: Array


class Buffer(eqx.Module):
    storage: List[Transition] = eqx.field(default_factory=list)

    def add(self, transition: Transition):
        self.storage.append(transition)

    def sample(self, rng_key, batch_size, steps):
        rng_key, subkey = jr.split(rng_key)
        indices = jr.choice(subkey, len(self.storage), (batch_size,), replace=True)

        batch = []
        for idx in indices:
            traj = self.storage[idx]
            T = traj.reward.shape[0]
            if T < steps:
                continue
            rng_key, subkey = jr.split(rng_key)
            start = jr.randint(subkey, (), 0, T - steps + 1)
            sliced = {
                f: getattr(traj, f)[start : start + steps] for f in traj.__annotations__
            }
            batch.append(Transition(**sliced))

        stacked = {
            f: jnp.stack([getattr(t, f) for t in batch])
            for f in batch[0].__annotations__
        }
        return Transition(**stacked)

    def clear(self):
        self.storage.clear()

    def __len__(self):
        return len(self.storage)


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
