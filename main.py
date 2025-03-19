import wandb
import argparse
import numpy as np
from functools import partial

import optax
import equinox as eqx


from jax import Array
from jax import numpy as jnp
from jax import random as jr
from jax import vmap, lax

import mcts


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--network_depth", type=int, default=2)
    p.add_argument("--network_width", type=int, default=64)
    p.add_argument("--num_episodes", type=int, default=20)
    p.add_argument("--num_warmup_episodes", type=int, default=4)
    p.add_argument("--num_episode_steps", type=int, default=500)
    p.add_argument("--num_train_epochs", type=int, default=50)
    p.add_argument("--num_simulations", type=int, default=500)
    p.add_argument("--max_depth", type=int, default=20)
    p.add_argument("--gumbel_scale", type=float, default=1.0)
    p.add_argument("--learning_rate", type=float, default=3e-4)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--steps", type=int, default=10)
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
            "episode/policy_entropy": mcts.entropy(traj.action_probs),
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
def rollout_episode(obs, env, model, rng_key, num_steps, **kwargs):
    def scan_step(carry, _):
        obs, key = carry
        key, k1, k2 = jr.split(key, 3)
        action, probs, value = mcts.plan_fn(obs, model, env, k1, **kwargs)
        next_obs, reward, done = env.step(obs, action, k2)
        return (next_obs, key), (obs, action, reward, done, value, probs)

    (final_obs, _), data = lax.scan(scan_step, (obs, rng_key), None, length=num_steps)
    obs, action, reward, done, value, probs = data
    return (
        mcts.Transition(
            obs,
            action,
            reward,
            done,
            value,
            probs,
            jnp.zeros_like(reward),
            jnp.ones_like(reward),
        ),
        final_obs,
    )


@eqx.filter_jit
def train_step(model, env, batch, optim, opt_state, rng_key):
    (loss, aux), grads = eqx.filter_value_and_grad(mcts.loss_fn, has_aux=True)(
        model, batch, env, rng_key
    )
    updates, opt_state = optim.update(grads, opt_state, model)
    model = eqx.apply_updates(model, updates)
    return model, opt_state, aux


def main():
    args = parse_args()
    wandb.init(project="muzero-project", name=args.name)
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
    model = mcts.Policy(
        int(np.prod(env.observation_shape)),
        env.action_shape,
        args.network_width,
        args.network_depth,
        subkey,
    )

    optim = optax.adam(args.learning_rate)
    opt_state = optim.init(eqx.filter(model, eqx.is_array))

    env_steps = 0
    train_steps = 0
    key, subkey = jr.split(key)
    obs = env.reset(subkey)

    for ep in range(args.num_episodes):
        key, subkey = jr.split(key)
        traj, obs = rollout_episode(
            obs, env, model, subkey, args.num_episode_steps, **kwargs
        )
        traj = mcts.compute_returns(traj, steps=args.steps)
        env_steps = env_steps + traj.obs.shape[0]
        buffer.add(traj, jnp.mean(traj.weight))

        if ep >= args.num_warmup_episodes:
            for _ in range(args.num_train_epochs):
                key, k1, k2 = jr.split(key, 3)
                batch = buffer.sample(k1, batch_size=args.batch_size, steps=args.steps)
                model, opt_state, aux = train_step(
                    model, env, batch, optim, opt_state, k2
                )
                train_steps = train_steps + 1
                log_train_metrics(aux, train_steps)

            log_episode_metrics(env, traj, env_steps, ep)

    wandb.finish()


if __name__ == "__main__":
    main()
