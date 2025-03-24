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
from jax import tree_util as jtu
from jax import vmap, lax, nn

import mcts
from mcts import qwen


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--name", type=str, default="qwen")
    p.add_argument("--batch_size", type=int, default=50)
    p.add_argument("--steps", type=int, default=50)
    p.add_argument("--rollout_steps", type=int, default=50)
    p.add_argument("--learning_rate", type=float, default=1e-4)
    p.add_argument("--num_train_epochs", type=int, default=200)
    p.add_argument("--num_test_batches", type=int, default=30)
    p.add_argument("--buffer_path", type=str, default="./data/buffer.pkl")
    return p.parse_args()


def log_metrics(train_aux, test_aux):
    wandb.log(
        {
            **{f"{k}": float(v) for k, v in train_aux.items()},
            **{f"{k}": float(v) for k, v in test_aux.items()},
            # **{f"train_{k}": wandb.Image(v) for k, v in train_figs.items()},
            # **{f"test_{k}": wandb.Image(v) for k, v in test_figs.items()},
        }
    )
    # plt.close("all")


def sample_test_data(test_buffer, key, batch_size, steps, num_batches):
    batches = []
    for _ in range(num_batches):
        key, subkey = jr.split(key)
        batch = test_buffer.sample(subkey, batch_size, steps, weighted=False)
        batches.append(batch)
    return batches


def process_data(obs_seq, action_seq, reward_seq):
    obs_seq = obs_seq[..., :2]
    prev_obs, next_obs = obs_seq[:-1], obs_seq[1:]
    prev_action, next_reward = action_seq[:-1], reward_seq[1:]
    next_reward = next_reward + 1
    prev_action = nn.one_hot(prev_action, 3)
    next_reward = nn.one_hot(next_reward, 3)
    return prev_obs, prev_action, next_obs, next_reward


@eqx.filter_jit
def test_step(key, model, test_batches):
    total_loss = 0.0
    for batch in test_batches:
        key, subkey = jr.split(key)
        prev_obs, prev_action, next_obs, next_reward = process_data(
            batch.obs, batch.action, batch.reward
        )
        loss, _ = qwen.loss_fn(
            model, prev_obs, prev_action, next_obs, next_reward, subkey
        )
        total_loss = total_loss + loss
    return {"test_loss": total_loss / len(test_batches)}


@eqx.filter_jit
def train_step(key, model, batch, optim, opt_state):
    prev_obs, prev_action, next_obs, next_reward = process_data(
        batch.obs, batch.action, batch.reward
    )

    value_and_grad = eqx.filter_value_and_grad(qwen.loss_fn, has_aux=True)
    (loss, aux), grads = value_and_grad(
        model, prev_obs, prev_action, next_obs, next_reward, key
    )
    updates, opt_state = optim.update(grads, opt_state, model)
    model = eqx.apply_updates(model, updates)
    return model, opt_state, aux


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
    model = qwen.init(
        key,
        obs_dim=6,
        action_dim=3,
        reward_dim=3,
        hidden_size=256,
        num_layers=4,
        num_heads=4,
        num_key_value_heads=4,
        rope_theta=5000.0,
        rms_norm_eps=1e-5,
        dropout=0.05,
    )

    adamw = optax.adamw(
        learning_rate=args.learning_rate, weight_decay=1e-3, b1=0.9, b2=0.999, eps=1e-8
    )
    optim = optax.chain(optax.clip_by_global_norm(1.0), adamw)
    opt_state = optim.init(eqx.filter(model, eqx.is_array))

    for epoch in range(args.num_train_epochs):
        key, subkey = jr.split(key)
        batch = train_buffer.sample(subkey, args.batch_size, args.steps, weighted=False)

        key, subkey = jr.split(key)
        model, opt_state, train_aux = train_step(subkey, model, batch, optim, opt_state)

        if epoch % 10 == 0:
            key, subkey = jr.split(key)
            test_batches = sample_test_data(
                test_buffer, subkey, args.batch_size, args.steps, args.num_test_batches
            )
            key, subkey = jr.split(key)
            test_aux = test_step(subkey, model, test_batches)

            log_metrics(train_aux, test_aux)
            eqx.tree_serialise_leaves(f"data/{args.name}.eqx", model)

        #     key, subkey = jr.split(key)
        #     rollout_train_batches = sample_test_data(
        #         train_buffer,
        #         subkey,
        #         args.batch_size,
        #         2 * args.rollout_steps + 1,
        #         args.num_test_batches,
        #     )

        #     key, subkey = jr.split(key)
        #     rollout_test_batches = sample_test_data(
        #         test_buffer,
        #         subkey,
        #         args.batch_size,
        #         2 * args.rollout_steps + 1,
        #         args.num_test_batches,
        #     )

        #     train_figs = evaluate(
        #         subkey, model, rollout_train_batches[0], args.rollout_steps
        #     )
        #     test_figs = evaluate(
        #         subkey, model, rollout_test_batches[0], args.rollout_steps
        #     )


if __name__ == "__main__":
    main()
