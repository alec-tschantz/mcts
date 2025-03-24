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
from mcts import rssm


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--name", type=str, default="rssm")
    p.add_argument("--batch_size", type=int, default=50)
    p.add_argument("--steps", type=int, default=50)
    p.add_argument("--rollout_steps", type=int, default=50)
    p.add_argument("--learning_rate", type=float, default=1e-4)
    p.add_argument("--num_train_epochs", type=int, default=200)
    p.add_argument("--num_test_batches", type=int, default=30)
    p.add_argument("--buffer_path", type=str, default="./data/buffer.pkl")
    return p.parse_args()


def log_metrics(train_aux, test_aux, train_figs, test_figs):
    wandb.log(
        {
            **{f"{k}": float(v) for k, v in train_aux.items()},
            **{f"{k}": float(v) for k, v in test_aux.items()},
            **{f"train_{k}": wandb.Image(v) for k, v in train_figs.items()},
            **{f"test_{k}": wandb.Image(v) for k, v in test_figs.items()},
        }
    )
    plt.close("all")


def sample_test_data(test_buffer, key, batch_size, steps, num_batches):
    batches = []
    for _ in range(num_batches):
        key, subkey = jr.split(key)
        batch = test_buffer.sample(subkey, batch_size, steps)
        batches.append(batch)
    return batches


def process_data(obs_seq, action_seq):
    obs_seq = obs_seq[..., :2]
    B, T, K, D = obs_seq.shape
    action_seq = nn.one_hot(action_seq, 3)
    return obs_seq.reshape(B, T, K * D), action_seq


@eqx.filter_jit
def test_step(key, model, test_batches):
    total_loss = 0.0
    for batch in test_batches:
        key, subkey = jr.split(key)
        obs_seq, action_seq = process_data(batch.obs, batch.action)
        loss, _ = rssm.loss_fn(model, obs_seq, action_seq, subkey)
        total_loss = total_loss + loss
    return {"test_loss": total_loss / len(test_batches)}


@eqx.filter_jit
def train_step(key, model, batch, optim, opt_state):
    obs_seq, action_seq = process_data(batch.obs, batch.action)
    value_and_grad = eqx.filter_value_and_grad(rssm.loss_fn, has_aux=True)
    (loss, aux), grads = value_and_grad(model, obs_seq, action_seq, key)
    updates, opt_state = optim.update(grads, opt_state, model)
    model = eqx.apply_updates(model, updates)
    return model, opt_state, aux


def evaluate(key, model, batch, rollout_len, num_samples=5):
    K, D = 3, 2  # TODO
    obs_seq, act_seq = process_data(batch.obs, batch.action)
    obs_seq, act_seq = obs_seq[0], act_seq[0]

    total_len = 2 * rollout_len
    start_idx = np.random.randint(0, obs_seq.shape[0] - total_len)
    
    obs_slice = obs_seq[start_idx : start_idx + total_len]
    act_slice = act_seq[start_idx : start_idx + total_len]
    warm_obs, warm_act = obs_slice[:rollout_len], act_slice[:rollout_len]
    remain_obs, remain_act = obs_slice[rollout_len:], act_slice[rollout_len:]

    vmap_encoder = lambda x: rssm.forward_encoder(model.encoder, x)
    vmap_decoder = lambda x: rssm.forward_decoder(model.decoder, x)
    vmap_rollout = lambda p, a, k: rssm.rollout_prior(model.prior, p, a, k)
    replicate = lambda x: jnp.repeat(x[None], num_samples, axis=0)

    warm_obs = warm_obs.reshape((rollout_len, K * D))
    init_post = rssm.init_post_state(model)
    key, rollout_key = jr.split(key)

    post_seq, _ = rssm.rollout(
        model.prior,
        model.posterior,
        vmap(vmap_encoder)(warm_obs),
        init_post,
        warm_act,
        rollout_key,
    )

    final_post = jtu.tree_map(lambda x: x[-1], post_seq)
    final_post_b = jtu.tree_map(replicate, final_post)
    remain_act_b = replicate(remain_act)

    key, subkey = jr.split(key)
    keys = jr.split(subkey, num_samples)

    rollout_outputs = vmap(vmap_rollout)(final_post_b, remain_act_b, keys)
    decoded_rollout = vmap(vmap(vmap_decoder))(rollout_outputs)
    decoded_rollout = decoded_rollout.reshape((num_samples, rollout_len, K, D))

    decoded_obs = vmap(vmap_decoder)(post_seq)
    decoded_obs = decoded_obs.reshape((rollout_len, K, D))

    fig, axes = plt.subplots(K, D, figsize=(2 * K, 2 * D), squeeze=False)
    obs_slice = obs_slice.reshape((-1, K, D))
    for i in range(K * D):
        r, c = i // D, i % D
        axes[r, c].plot(
            jnp.arange(rollout_len),
            decoded_obs[:, r, c],
            alpha=0.5,
            color="orange",
            marker="x",
        )

        for s in range(num_samples):
            axes[r, c].plot(
                jnp.arange(rollout_len, rollout_len * 2),
                decoded_rollout[s, :, r, c],
                alpha=0.5,
                color="orange",
                marker="x",
            )
        axes[r, c].plot(
            jnp.arange(rollout_len * 2),
            obs_slice[:, r, c],
            alpha=1.0,
            color="green",
            marker="o",
        )

    return {"rollout": fig}


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
    model = rssm.init_model(
        obs_dim=6,
        action_dim=3,
        embed_dim=512,
        state_dim=200,
        num_discrete=32,
        discrete_dim=32,
        hidden_dim=200,
        key=subkey,
    )

    optim = optax.chain(optax.clip_by_global_norm(1.0), optax.adam(args.learning_rate))
    opt_state = optim.init(eqx.filter(model, eqx.is_array))

    for epoch in range(args.num_train_epochs):
        key, subkey = jr.split(key)
        batch = train_buffer.sample(subkey, args.batch_size, args.steps)

        key, subkey = jr.split(key)
        model, opt_state, train_aux = train_step(subkey, model, batch, optim, opt_state)

        key, subkey = jr.split(key)
        test_batches = sample_test_data(
            test_buffer, subkey, args.batch_size, args.steps, args.num_test_batches
        )

        key, subkey = jr.split(key)
        test_aux = test_step(subkey, model, test_batches)

        key, subkey = jr.split(key)
        rollout_train_batches = sample_test_data(
            train_buffer,
            subkey,
            args.batch_size,
            2 * args.rollout_steps + 1,
            args.num_test_batches,
        )

        key, subkey = jr.split(key)
        rollout_test_batches = sample_test_data(
            test_buffer,
            subkey,
            args.batch_size,
            2 * args.rollout_steps + 1,
            args.num_test_batches,
        )

        train_figs = evaluate(
            subkey, model, rollout_train_batches[0], args.rollout_steps
        )
        test_figs = evaluate(subkey, model, rollout_test_batches[0], args.rollout_steps)

        log_metrics(train_aux, test_aux, train_figs, test_figs)
        eqx.tree_serialise_leaves(f"data/{args.name}.eqx", model)


if __name__ == "__main__":
    main()
