import argparse
import pickle
import wandb
import numpy as np
import ale_py
import gymnasium as gym
import optax
import equinox as eqx
from jax import numpy as jnp
from jax import random as jr
from jax import tree_util as jtu

gym.register_envs(ale_py) 

import mcts
from mcts import models, rssm, losses, utils, buffers, wrappers


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--env_name", type=str, default="ALE/Pong-v5")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--total_steps", type=int, default=100_000)
    p.add_argument("--warmup_steps", type=int, default=1_000)
    p.add_argument("--train_every", type=int, default=500)
    p.add_argument("--num_train_epochs", type=int, default=30)
    p.add_argument("--warmup_num_simulations", type=int, default=1)
    p.add_argument("--warmup_max_depth", type=int, default=1)
    p.add_argument("--num_simulations", type=int, default=100)
    p.add_argument("--max_depth", type=int, default=20)
    p.add_argument("--max_episode_steps", type=int, default=500)
    p.add_argument("--seq_size", type=int, default=30)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--temperature_final", type=float, default=0.2)
    p.add_argument("--learning_rate", type=float, default=3e-4)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--rssm_embed_dim", type=int, default=8)
    p.add_argument("--rssm_state_dim", type=int, default=16)
    p.add_argument("--rssm_num_discrete", type=int, default=8)
    p.add_argument("--rssm_discrete_dim", type=int, default=8)
    p.add_argument("--rssm_hidden_dim", type=int, default=32)
    p.add_argument("--policy_hidden_dim", type=int, default=64)
    p.add_argument("--policy_depth", type=int, default=2)
    p.add_argument("--value_hidden_dim", type=int, default=64)
    p.add_argument("--value_depth", type=int, default=2)
    p.add_argument("--reward_hidden_dim", type=int, default=64)
    p.add_argument("--reward_depth", type=int, default=2)
    p.add_argument("--support_size", type=int, default=5)
    p.add_argument("--discount", type=float, default=0.99)
    p.add_argument("--return_steps", type=int, default=10)
    p.add_argument("--name", type=str, default="pong")
    return p.parse_args()


def update_dynamic_params(step, args):
    progress = (step + 1) / args.total_steps
    return {
        "temperature": args.temperature
        + progress * (args.temperature_final - args.temperature)
    }


def log_episode_metrics(frames, transition, env_steps, episode_idx):
    frames_stacked = jnp.transpose(jnp.stack(frames), (0, 3, 1, 2))
    wandb.log(
        {
            "eval/video": wandb.Video(np.array(frames_stacked), fps=30),
            "eval/episode_idx": episode_idx,
            "eval/episode_reward": float(jnp.sum(transition.reward)),
            "eval/env_steps": env_steps,
            "eval/policy_entropy": utils.entropy(transition.action_probs),
        }
    )


def log_train_metrics(aux, train_steps):
    wandb.log(
        {
            **{f"train/{k}": float(v) for k, v in aux.items()},
            "train/train_steps": train_steps,
        }
    )


def rollout_step(
    rng_key, env, model, post, obs, temperature, max_depth, num_simulations
):
    _batch = lambda x: jtu.tree_map(lambda y: y[None], x)
    action, probs, value = models.action_fn(
        rng_key, model, _batch(post), temperature, max_depth, num_simulations
    )
    post_new = models.compute_posterior(model, post, obs, action[0], rng_key)
    next_obs, reward, terminated, truncated, _ = env.step(action[0].item())
    done = terminated or truncated
    out = {
        "obs": obs,
        "action": action,
        "reward": reward,
        "done": done,
        "value": value,
        "probs": probs,
    }
    return out, jnp.array(next_obs, dtype=jnp.float32), post_new


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
    wandb.init(project="muzero-pong", name=args.name)
    wandb.config.update(vars(args))

    env = gym.make(args.env_name, render_mode="rgb_array")
    env = wrappers.OCWrapper(env)
    # env = gym.wrappers.NormalizeObservation(env)
    key = jr.PRNGKey(args.seed)

    obs_dim = int(env.observation_space.shape[0])
    action_dim = int(env.action_space.n)
    key, subkey = jr.split(key)
    model = models.init_model(
        obs_dim=obs_dim,
        action_dim=action_dim,
        rssm_embed_dim=args.rssm_embed_dim,
        rssm_state_dim=args.rssm_state_dim,
        rssm_num_discrete=args.rssm_num_discrete,
        rssm_discrete_dim=args.rssm_discrete_dim,
        rssm_hidden_dim=args.rssm_hidden_dim,
        policy_hidden_dim=args.policy_hidden_dim,
        policy_depth=args.policy_depth,
        value_hidden_dim=args.value_hidden_dim,
        value_depth=args.value_depth,
        reward_hidden_dim=args.reward_hidden_dim,
        reward_depth=args.reward_depth,
        support_size=args.support_size,
        discount=args.discount,
        key=subkey,
    )

    optim = optax.chain(optax.clip_by_global_norm(1.0), optax.adam(args.learning_rate))
    opt_state = optim.init(eqx.filter(model, eqx.is_array))

    buffer = buffers.Buffer()
    post = rssm.init_post_state(model.rssm)

    key, subkey = jr.split(key)
    obs, _ = env.reset(seed=int(subkey[0]))
    obs = jnp.array(obs, dtype=jnp.float32)

    frames = []
    env_steps, train_step_count, episode_steps, episode_idx = 0, 0, 0, 0
    transition = buffers.Transition()

    for step in range(args.total_steps):
        if episode_steps == 0:
            frames.clear()

        dyn_params = update_dynamic_params(step, args)
        num_simulations = (
            args.warmup_num_simulations
            if step < args.warmup_steps
            else args.num_simulations
        )
        max_depth = (
            args.warmup_max_depth if step < args.warmup_steps else args.max_depth
        )

        frames.append(env.render())
        key, subkey = jr.split(key)
        out, next_obs, post = rollout_step(
            subkey,
            env,
            model,
            post,
            obs,
            dyn_params["temperature"],
            max_depth,
            num_simulations,
        )

        transition = transition.append(out)

        obs = next_obs
        env_steps += 1
        episode_steps += 1

        if out["done"] or (episode_steps >= args.max_episode_steps):
            log_episode_metrics(frames, transition, env_steps, episode_idx)

            transition = utils.compute_returns(
                transition, steps=args.return_steps, gamma=args.discount
            )
            buffer.add(transition, jnp.mean(transition.weight))

            transition = buffers.Transition()
            episode_steps = 0
            episode_idx += 1
            key, subkey = jr.split(key)
            obs, _ = env.reset(seed=int(subkey[0]))
            obs = jnp.array(obs, dtype=jnp.float32)
            post = rssm.init_post_state(model.rssm)

        if step >= args.warmup_steps and (step + 1) % args.train_every == 0:
            for _ in range(args.num_train_epochs):
                key, k1, k2 = jr.split(key, 3)
                batch = buffer.sample(
                    k1, batch_size=args.batch_size, steps=args.seq_size
                )
                model, opt_state, aux = train_step(model, batch, optim, opt_state, k2)
                train_step_count += 1
                log_train_metrics(aux, train_step_count)

    eqx.tree_serialise_leaves(f"data/{args.name}.eqx", model)
    with open("data/buffer.pkl", "wb") as f:
        pickle.dump(buffer, f)

    wandb.finish()
    env.close()


if __name__ == "__main__":
    main()
