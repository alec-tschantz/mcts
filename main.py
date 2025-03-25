import pickle
import wandb
import argparse
import numpy as np
from functools import partial

import optax
import equinox as eqx

from jax import numpy as jnp
from jax import random as jr
from jax import tree_util as jtu
from jax import vmap, lax

import mcts
from mcts import models, rssm, losses, utils


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--network_depth", type=int, default=2)
    p.add_argument("--network_width", type=int, default=64)
    p.add_argument("--num_episodes", type=int, default=20)
    p.add_argument("--num_warmup_episodes", type=int, default=5)
    p.add_argument("--num_episode_steps", type=int, default=500)
    p.add_argument("--num_train_epochs", type=int, default=50)
    p.add_argument("--num_simulations", type=int, default=100)
    p.add_argument("--max_depth", type=int, default=10)
    p.add_argument("--gumbel_scale", type=float, default=1.0)
    p.add_argument("--learning_rate", type=float, default=3e-4)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--seq_size", type=int, default=32)
    p.add_argument("--return_steps", type=int, default=10)
    p.add_argument("--rssm_embed_dim", type=int, default=128)
    p.add_argument("--rssm_state_dim", type=int, default=200)
    p.add_argument("--rssm_num_discrete", type=int, default=8)
    p.add_argument("--rssm_discrete_dim", type=int, default=8)
    p.add_argument("--rssm_hidden_dim", type=int, default=200)
    p.add_argument("--policy_hidden_dim", type=int, default=200)
    p.add_argument("--name", type=str, default="muzero-run")
    return p.parse_args()


def log_episode_metrics(env, traj, env_steps, episode_idx):
    frames = jnp.stack([env.render(s) for s in traj.obs], axis=0)
    frames = (frames * 255).clip(0, 255).astype(jnp.uint8)
    frames = jnp.transpose(frames, (0, 3, 1, 2))
    wandb.log(
        {
            "video": wandb.Video(np.array(frames), fps=30),
            "episode/episode_idx": episode_idx,
            "episode/episode_reward": float(jnp.sum(traj.reward)),
            "episode/env_steps": env_steps,
            "episode/policy_entropy": utils.entropy(traj.action_probs),
        }
    )


def log_train_metrics(aux, train_steps):
    wandb.log(
        {
            **{f"train/{k}": float(v) for k, v in aux.items()},
            "train/train_steps": train_steps,
        }
    )


@eqx.filter_jit
def rollout_episode(obs, post, env, model, rng_key, num_steps, **kwargs):
    _batch = lambda x: jtu.tree_map(lambda x: x[None], x)
    _process_obs = lambda o: o[..., :2].reshape(-1)

    def scan_step(carry, _):
        obs, post, key = carry
        key, k1, k2, k3 = jr.split(key, 4)
        action, probs, value = models.action_fn(k1, model, _batch(post))
        new_post = models.compute_posterior(
            model, post, _process_obs(obs), action[0], k2
        )
        next_obs, reward, done = env.step(obs, action[0], k3)
        return (next_obs, new_post, key), (obs, action, reward, done, value, probs)

    (final_obs, final_post, _), data = lax.scan(
        scan_step, (obs, post, rng_key), None, length=num_steps
    )
    obs_seq, action_seq, reward_seq, done_seq, value_seq, probs_seq = data
    transition = mcts.Transition(
        obs=obs_seq,
        action=action_seq.squeeze(),
        reward=reward_seq,
        done=done_seq,
        value=value_seq.squeeze(),
        action_probs=probs_seq.squeeze(),
        returns=jnp.zeros_like(reward_seq),
        weight=jnp.ones_like(reward_seq),
    )
    return transition, final_obs, final_post


@eqx.filter_jit
def train_step(model, batch, optim, opt_state, rng_key):
    (loss, aux), grads = eqx.filter_value_and_grad(losses.loss_fn, has_aux=True)(
        model, batch, rng_key
    )
    updates, opt_state = optim.update(grads, opt_state, model)
    model = eqx.apply_updates(model, updates)
    return model, opt_state, aux


def main():
    args = parse_args()
    wandb.init(project="muzero", name=args.name)
    wandb.config.update(vars(args))
    key = jr.PRNGKey(args.seed)
    kwargs = {
        "max_depth": args.max_depth,
        "gumbel_scale": args.gumbel_scale,
        "num_simulations": args.num_simulations,
    }
    env = mcts.Pong()
    buffer = mcts.Buffer()
    key, subkey = jr.split(key)
    model = models.Model(
        obs_dim=6,
        action_dim=env.action_shape,
        rssm_embed_dim=args.rssm_embed_dim,
        rssm_state_dim=args.rssm_state_dim,
        rssm_num_discrete=args.rssm_num_discrete,
        rssm_discrete_dim=args.rssm_discrete_dim,
        rssm_hidden_dim=args.rssm_hidden_dim,
        policy_hidden_dim=args.policy_hidden_dim,
        policy_depth=args.network_depth,
        key=subkey,
    )
    optim = optax.adam(args.learning_rate)
    opt_state = optim.init(eqx.filter(model, eqx.is_array))
    env_steps, train_steps = 0, 0

    key, subkey = jr.split(key)
    obs = env.reset(subkey)
    post = rssm.init_post_state(model.rssm)
    for ep in range(args.num_episodes):
        key, subkey = jr.split(key)
        traj, obs, post = rollout_episode(
            obs, post, env, model, subkey, args.num_episode_steps, **kwargs
        )
        traj = utils.compute_returns(traj, steps=args.return_steps)
        env_steps += traj.obs.shape[0]
        buffer.add(traj, jnp.mean(traj.weight))
        if ep >= args.num_warmup_episodes:
            for _ in range(args.num_train_epochs):
                key, k1, k2 = jr.split(key, 3)
                batch = buffer.sample(k1, batch_size=args.batch_size, steps=args.seq_size)
                model, opt_state, aux = train_step(
                    model, batch, optim, opt_state, k2
                )
                train_steps += 1
                log_train_metrics(aux, train_steps)

            log_episode_metrics(env, traj, env_steps, ep)
            post = rssm.init_post_state(model.rssm)

        eqx.tree_serialise_leaves("data/model.eqx", model)

        with open("data/buffer.pkl", "wb") as f:
            pickle.dump(buffer, f)
    wandb.finish()


if __name__ == "__main__":
    main()
