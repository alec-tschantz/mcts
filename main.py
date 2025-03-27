import pickle
import wandb
import argparse
import numpy as np

import gymnax
import optax
import equinox as eqx
from jax import numpy as jnp
from jax import random as jr
from jax import tree_util as jtu
from jax import vmap, lax

import mcts
from mcts import models, rssm, losses, utils, buffers


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--env_name", type=str, default="CartPole-v1")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--num_episodes", type=int, default=30)
    p.add_argument("--num_warmup_episodes", type=int, default=4)
    p.add_argument("--num_episode_steps", type=int, default=500)
    p.add_argument("--num_train_epochs", type=int, default=100)
    p.add_argument("--num_simulations", type=int, default=100)
    p.add_argument("--max_depth", type=int, default=20)
    p.add_argument("--max_depth_end", type=int, default=20)
    p.add_argument("--return_steps", type=int, default=10)
    p.add_argument("--return_steps_end", type=int, default=10)
    p.add_argument("--seq_size", type=int, default=32)
    p.add_argument("--seq_size_end", type=int, default=32)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--temperature_end", type=float, default=0.2)
    p.add_argument("--learning_rate", type=float, default=3e-4)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--rssm_embed_dim", type=int, default=16)
    p.add_argument("--rssm_state_dim", type=int, default=8)
    p.add_argument("--rssm_num_discrete", type=int, default=4)
    p.add_argument("--rssm_discrete_dim", type=int, default=4)
    p.add_argument("--rssm_hidden_dim", type=int, default=64)
    p.add_argument("--policy_hidden_dim", type=int, default=64)
    p.add_argument("--policy_depth", type=int, default=2)
    p.add_argument("--support_size", type=int, default=10)
    p.add_argument("--name", type=str, default="cartpole")
    return p.parse_args()


def update_dynamic_params(dyn, episode):
    progress = (episode + 1) / dyn["total_episodes"]
    return {
        "max_depth": int(
            dyn["max_depth"] + progress * (dyn["max_depth_end"] - dyn["max_depth"])
        ),
        "return_steps": int(
            dyn["return_steps"]
            + progress * (dyn["return_steps_end"] - dyn["return_steps"])
        ),
        "seq_size": int(
            dyn["seq_size"] + progress * (dyn["seq_size_end"] - dyn["seq_size"])
        ),
        "temperature": dyn["temperature"]
        + progress * (dyn["temperature_end"] - dyn["temperature"]),
    }


def log_episode_metrics(env, env_params, traj, env_steps, episode_idx):
    # frames = jnp.stack([env.render(s, env_params) for s in traj.obs], axis=0)
    # frames = (frames * 255).clip(0, 255).astype(jnp.uint8)
    # frames = jnp.transpose(frames, (0, 3, 1, 2))
    wandb.log(
        {
            # "video": wandb.Video(np.array(frames), fps=30),
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
def rollout_episode(
    rng_key,
    env_state,
    obs,
    post,
    model,
    env,
    env_params,
    num_steps,
    temperature,
    max_depth,
    num_simulations,
):

    _batch = lambda x: jtu.tree_map(lambda y: y[None], x)

    def scan_fn(carry, _):
        rng, env_state, obs, post = carry
        rng, rng_act, rng_step, rng_post = jr.split(rng, 4)
        action, probs, value = models.action_fn(
            rng_act, model, _batch(post), temperature, max_depth, num_simulations
        )
        new_post = models.compute_posterior(model, post, obs, action[0], rng_post)
        next_obs, next_env_state, rew, done, _ = env.step(
            rng_step, env_state, action[0], env_params
        )
        
        # TODO:
        next_obs, next_env_state = lax.cond(
            done,
            lambda _:  env.reset(rng_step, env_params),
            lambda _: (next_obs, next_env_state),
            operand=None,
        )        
        # new_post = lax.select(done, rssm.init_post_state(model.rssm), new_post)
        
        return (rng, next_env_state, next_obs, new_post), (
            obs,
            action.squeeze(),
            rew,
            done,
            value.squeeze(),
            probs.squeeze(),
        )

    init_carry = (rng_key, env_state, obs, post)
    (final_rng, final_env_state, final_obs, final_post), data = lax.scan(
        scan_fn, init_carry, None, length=num_steps
    )
    obs_seq, action_seq, reward_seq, done_seq, value_seq, probs_seq = data

    transition = buffers.Transition(
        obs=obs_seq,
        action=action_seq,
        reward=reward_seq,
        done=done_seq,
        value=value_seq,
        action_probs=probs_seq,
        returns=jnp.zeros_like(reward_seq),
        weight=jnp.ones_like(reward_seq),
    )
    return transition, final_env_state, final_obs, final_post


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

    env, env_params = gymnax.make(args.env_name)
    key, subkey = jr.split(key)
    obs, env_state = env.reset(subkey, env_params)
    obs_dim = int(np.prod(obs.shape))
    action_dim = env.action_space(env_params).n

    buffer = buffers.Buffer()
    dyn_params = {
        "max_depth": args.max_depth,
        "max_depth_end": args.max_depth_end,
        "return_steps": args.return_steps,
        "return_steps_end": args.return_steps_end,
        "seq_size": args.seq_size,
        "seq_size_end": args.seq_size_end,
        "temperature": args.temperature,
        "temperature_end": args.temperature_end,
        "total_episodes": args.num_episodes,
    }
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
        support_size=args.support_size,
        key=subkey,
        discount=0.99,
    )
    optim = optax.chain(
        optax.clip_by_global_norm(1.0),
        optax.adam(args.learning_rate),
    )
    opt_state = optim.init(eqx.filter(model, eqx.is_array))
    env_steps, train_steps = 0, 0
    post = rssm.init_post_state(model.rssm)

    for ep in range(args.num_episodes):
        updated = update_dynamic_params(dyn_params, ep)
        key, subkey = jr.split(key)

        traj, env_state, obs, post = rollout_episode(
            rng_key=subkey,
            env_state=env_state,
            obs=obs,
            post=post,
            model=model,
            env=env,
            env_params=env_params,
            num_steps=args.num_episode_steps,
            temperature=updated["temperature"],
            max_depth=updated["max_depth"],
            num_simulations=args.num_simulations,
        )
        traj = utils.compute_returns(
            traj, steps=updated["return_steps"], gamma=0.99, alpha=0.5
        )
        buffer.add(traj, jnp.mean(traj.weight))
        env_steps = env_steps + traj.obs.shape[0]

        if ep >= args.num_warmup_episodes:
            for _ in range(args.num_train_epochs):
                key, k1, k2 = jr.split(key, 3)
                batch = buffer.sample(
                    k1,
                    batch_size=args.batch_size,
                    steps=updated["seq_size"],
                    weighted=True,
                )
                model, opt_state, aux = train_step(model, batch, optim, opt_state, k2)
                train_steps += 1
                log_train_metrics(aux, train_steps)
            log_episode_metrics(env, env_params, traj, env_steps, ep)

        eqx.tree_serialise_leaves(f"data/{args.name}.eqx", model)
        with open("data/buffer.pkl", "wb") as f:
            pickle.dump(buffer, f)

    wandb.finish()


if __name__ == "__main__":
    main()
