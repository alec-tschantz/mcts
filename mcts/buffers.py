from typing import Any, List, Optional, Dict
import equinox as eqx
from jax import Array, numpy as jnp, random as jr

import equinox as eqx
import jax.numpy as jnp
from typing import Optional, Dict, Any
from jaxtyping import Array


class Transition(eqx.Module):
    obs: Optional[Array] = None
    action: Optional[Array] = None
    reward: Optional[Array] = None
    done: Optional[Array] = None
    value: Optional[Array] = None
    action_probs: Optional[Array] = None
    returns: Optional[Array] = None
    weight: Optional[Array] = None

    def append(self, data: Dict[str, Any]):
        def _append_field(field_name: str, x: jnp.ndarray):
            arr = getattr(self, field_name)
            if arr is None:
                return jnp.expand_dims(x, axis=0)
            else:
                return jnp.concatenate([arr, jnp.expand_dims(x, 0)], axis=0)

        new_fields = {}
        new_fields["obs"] = _append_field("obs", data["obs"])
        new_fields["action"] = _append_field("action", data["action"][0])
        new_fields["reward"] = _append_field(
            "reward", jnp.array(data["reward"], dtype=jnp.float32)
        )
        new_fields["done"] = _append_field(
            "done", jnp.array(data["done"], dtype=jnp.bool_)
        )
        new_fields["value"] = _append_field("value", data["value"].squeeze())
        new_fields["action_probs"] = _append_field(
            "action_probs", data["probs"].squeeze()
        )

        return eqx.tree_at(
            where=lambda t: (
                t.obs,
                t.action,
                t.reward,
                t.done,
                t.value,
                t.action_probs,
            ),
            pytree=self,
            replace=(
                new_fields["obs"],
                new_fields["action"],
                new_fields["reward"],
                new_fields["done"],
                new_fields["value"],
                new_fields["action_probs"],
            ),
            is_leaf=lambda x: x is None,
        )


class Buffer(eqx.Module):
    storage: List[Transition] = eqx.field(default_factory=list)
    weights: List[float] = eqx.field(default_factory=list)

    def add(self, transition: Transition, weight: float = 1.0):
        self.storage.append(transition)
        self.weights.append(weight)

    def sample(self, rng_key, batch_size, steps, weighted: bool = True):
        if weighted and len(self.weights) > 0:
            weights = jnp.array(self.weights)
            norm_weights = weights / jnp.sum(weights)
        else:
            norm_weights = jnp.ones(len(self.storage)) / len(self.storage)

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

            if weighted:
                trans_weights = traj.weight[:valid_range]
                norm_trans_weights = trans_weights / (jnp.sum(trans_weights) + 1e-6)
            else:
                norm_trans_weights = jnp.ones(valid_range) / valid_range

            rng_key, subkey = jr.split(rng_key)
            start_idx = jr.choice(subkey, valid_range, (), p=norm_trans_weights)

            sliced = {
                f: getattr(traj, f)[start_idx : start_idx + steps]
                for f in traj.__annotations__
            }
            batch.append(Transition(**sliced))

        if len(batch) == 0:
            raise ValueError("No valid trajectories long enough for sampling.")

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
