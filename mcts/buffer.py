from typing import List
import equinox as eqx
from jax import Array, numpy as jnp, random as jr


class Transition(eqx.Module):
    obs: Array
    action: Array
    reward: Array
    done: Array
    value: Array
    action_probs: Array
    returns: Array
    weight: Array


class Buffer(eqx.Module):
    storage: List[Transition] = eqx.field(default_factory=list)
    weights: List[float] = eqx.field(default_factory=list)

    def add(self, transition: Transition, weight: float = 1.0):
        self.storage.append(transition)
        self.weights.append(weight)

    def sample(self, rng_key, batch_size, steps):
        weights = jnp.array(self.weights)
        norm_weights = weights / jnp.sum(weights)

        rng_key, subkey = jr.split(rng_key)
        indices = jr.choice(
            subkey, len(self.storage), (batch_size,), replace=True, p=norm_weights
        )

        batch = []
        for idx in indices:
            traj = self.storage[idx]
            T = traj.reward.shape[0]
            if T < steps:
                continue

            valid_range = T - steps + 1

            trans_weights = traj.weight[:valid_range]
            norm_trans_weights = trans_weights / (jnp.sum(trans_weights) + 1e-6)

            rng_key, subkey = jr.split(rng_key)
            start_idx = jr.choice(subkey, valid_range, (), p=norm_trans_weights)

            sliced = {
                f: getattr(traj, f)[start_idx : start_idx + steps]
                for f in traj.__annotations__
            }
            batch.append(Transition(**sliced))

        stacked = {
            f: jnp.stack([getattr(t, f) for t in batch])
            for f in batch[0].__annotations__
        }
        return Transition(**stacked)

    def clear(self):
        self.storage.clear()
        self.weights.clear()

    def __len__(self):
        return len(self.storage)


def train_test_split(buffer: Buffer, rng_key: jr.PRNGKey, ratio: float = 0.2):
    total = len(buffer.storage)
    indices = jnp.arange(total)
    shuffled_indices = jr.permutation(rng_key, indices)

    split_idx = int(total * (1 - ratio))
    train_indices = shuffled_indices[:split_idx]
    test_indices = shuffled_indices[split_idx:]

    train_buffer = Buffer()
    test_buffer = Buffer()

    for idx in train_indices:
        idx = int(idx)
        train_buffer.add(buffer.storage[idx], buffer.weights[idx])

    for idx in test_indices:
        idx = int(idx)
        test_buffer.add(buffer.storage[idx], buffer.weights[idx])

    return train_buffer, test_buffer
