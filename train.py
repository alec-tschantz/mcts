import pickle
import wandb
import argparse
import numpy as np
from functools import partial
from matplotlib import pyplot as plt

import optax
import equinox as eqx


from jax import Array
from jax import numpy as jnp
from jax import random as jr
from jax import vmap, lax

import mcts
from models import test, phys

store = {
    "test": {
        "model": test.Test,
        "loss_fn": test.loss_fn,
        "rollout_fn": test.rollout_fn,
    },
    "phys": {
        "model": phys.Phys,            
        "loss_fn": phys.loss_fn,
        "rollout_fn": phys.rollout_fn,
    },
}

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--name", type=str, default="phys")
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--steps", type=int, default=10)
    p.add_argument("--rollout_steps", type=int, default=30)
    p.add_argument("--learning_rate", type=float, default=3e-4)
    p.add_argument("--num_train_epochs", type=int, default=100)
    p.add_argument("--num_test_batches", type=int, default=20)
    p.add_argument("--buffer_path", type=str, default="./data/buffer.pkl")
    return p.parse_args()


def log_metrics(train_aux, test_aux, train_fig, test_fig):
    wandb.log(
        {
            **{f"{k}": float(v) for k, v in train_aux.items()},
            **{f"{k}": float(v) for k, v in test_aux.items()},
            "train_rollout": wandb.Image(train_fig),
            "test_rollout": wandb.Image(test_fig),
        }
    )
    plt.close(train_fig)
    plt.close(test_fig)


def create_figure(true_obs, pred_obs):
    B, T, O, D = true_obs.shape
    b = 0

    fig, axes = plt.subplots(O, D, figsize=(D * 2, O * 2), constrained_layout=True)

    for o in range(O):
        for d in range(D):
            ax = axes[o, d]
            ax.plot(true_obs[b, :, o, d], color="black", linewidth=1, label="True")
            ax.plot(
                pred_obs[b, :, o, d],
                color="red",
                linestyle="--",
                linewidth=1,
                label="Pred",
            )
    return fig


def sample_test_data(test_buffer, key, batch_size, steps, num_batches):
    batches = []
    for _ in range(num_batches):
        key, subkey = jr.split(key)
        batch = test_buffer.sample(subkey, batch_size, steps)
        batches.append(batch)
    return batches


@eqx.filter_jit
def train_step(key, model, loss_fn, batch, optim, opt_state):
    value_and_grad = eqx.filter_value_and_grad(loss_fn, has_aux=True)
    (loss, aux), grads = value_and_grad(model, batch, key)
    updates, opt_state = optim.update(grads, opt_state, model)
    model = eqx.apply_updates(model, updates)
    return model, opt_state, aux


@eqx.filter_jit
def test_step(key, model, loss_fn, test_batches):
    total_loss = 0.0
    for batch in test_batches:
        key, subkey = jr.split(key)
        loss, _ = loss_fn(model, batch, subkey)
        total_loss = total_loss + loss
    return {"test_loss": total_loss / len(test_batches)}


def main():
    args = parse_args()
    wandb.init(project="muzero-project", name=args.name)
    wandb.config.update(vars(args))

    key = jr.PRNGKey(args.seed)

    with open(args.buffer_path, "rb") as f:
        buffer = pickle.load(f)

    key, subkey = jr.split(key)
    train_buffer, test_buffer = mcts.train_test_split(buffer, subkey)

    key, subkey = jr.split(key)
    loss_fn = store[args.name]["loss_fn"]
    rollout_fn = store[args.name]["rollout_fn"]
    dynamics = store[args.name]["model"]((3, 4), 3, subkey)

    optim = optax.adam(args.learning_rate)
    opt_state = optim.init(eqx.filter(dynamics, eqx.is_array))

    for epoch in range(args.num_train_epochs):
        key, subkey = jr.split(key)
        batch = train_buffer.sample(subkey, args.batch_size, args.steps)

        key, subkey = jr.split(key)
        dynamics, opt_state, train_aux = train_step(
            subkey, dynamics, loss_fn, batch, optim, opt_state
        )

        key, subkey = jr.split(key)
        test_batches = sample_test_data(
            test_buffer, subkey, args.batch_size, args.steps, args.num_test_batches
        )

        key, subkey = jr.split(key)
        test_aux = test_step(
            subkey,
            dynamics,
            loss_fn,
            test_batches,
        )

        key, subkey = jr.split(key)
        rollout_test_batches = sample_test_data(
            test_buffer,
            subkey,
            args.batch_size,
            args.rollout_steps,
            args.num_test_batches,
        )

        key, subkey = jr.split(key)
        rollout_train_batches = sample_test_data(
            train_buffer,
            subkey,
            args.batch_size,
            args.rollout_steps,
            args.num_test_batches,
        )

        rollout_batch = rollout_test_batches[0]
        pred_test_obs = rollout_fn(subkey, dynamics, rollout_batch)
        test_fig = create_figure(rollout_batch.obs, pred_test_obs)

        rollout_batch = rollout_train_batches[0]
        pred_train_obs = rollout_fn(subkey, dynamics, rollout_batch)
        train_fig = create_figure(rollout_batch.obs, pred_train_obs)

        log_metrics(train_aux, test_aux, train_fig, test_fig)
        eqx.tree_serialise_leaves(f"data/{args.name}.eqx", dynamics)


if __name__ == "__main__":
    main()
