import argparse
import wandb
import jax
import jax.numpy as jnp
import jax.random as jr
import jax.nn as nn
import optax
import equinox as eqx
import numpy as np
import mctx
import mcts
from jax import lax, vmap
from functools import partial


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--network_depth", type=int, default=2)
    p.add_argument("--network_width", type=int, default=64)
    p.add_argument("--support_size", type=int, default=2)
    p.add_argument(
        "--value_type", type=str, default="discrete", choices=["discrete", "continuous"]
    )
    p.add_argument("--num_episodes", type=int, default=16)
    p.add_argument("--num_warmup_episodes", type=int, default=4)
    p.add_argument("--num_episode_steps", type=int, default=500)
    p.add_argument("--num_train_epochs", type=int, default=50)
    p.add_argument("--num_simulations", type=int, default=500)
    p.add_argument("--max_depth", type=int, default=20)
    p.add_argument("--gumbel_scale", type=float, default=1.0)
    p.add_argument("--run_name", type=str, default="muzero-run")
    return p.parse_args()


class Policy(eqx.Module):
    value_head: eqx.nn.MLP
    policy_head: eqx.nn.MLP
    value_type: str = "discrete"

    def __init__(
        self, input_shape, num_actions, depth, width, support_size, value_type, key
    ):
        in_features = input_shape[0] * input_shape[1]
        v_out = 2 * support_size + 1 if value_type == "discrete" else 1
        k1, k2 = jr.split(key)
        self.value_head = eqx.nn.MLP(in_features, v_out, width, depth, key=k1)
        self.policy_head = eqx.nn.MLP(in_features, num_actions, width, depth, key=k2)
        self.value_type = value_type

    def __call__(self, obs):
        flat = obs.reshape(-1)
        v = self.value_head(flat)
        p = self.policy_head(flat)
        return v, p


def env_step_batched(env, state, action, rng_key):
    return vmap(env.step)(state, action.astype(jnp.int32), rng_key)


def root_fn(env, obs, model, support_size, rng_key):
    v, p = vmap(model)(obs)
    if model.value_type == "discrete":
        v_probs = nn.softmax(v)
        val = mcts.from_discrete(v_probs, support_size)
    else:
        val = v.squeeze(-1)
    return mctx.RootFnOutput(prior_logits=p, value=val, embedding=obs)


def recurrent_fn(params, rng_key, action, embedding):
    env, model, support_size = params
    b = embedding.shape[0]
    ns, r, d = env_step_batched(env, embedding, action, jr.split(rng_key, b))
    v_logits, p_logits = vmap(model)(ns)
    if model.value_type == "discrete":
        vp = nn.softmax(v_logits)
        vs = mcts.from_discrete(vp, support_size)
    else:
        vs = v_logits.squeeze(-1)
    disc = jnp.where(d, 0.0, 1.0)
    return (
        mctx.RecurrentFnOutput(
            reward=r, discount=disc, prior_logits=p_logits, value=vs
        ),
        ns,
    )


def act(state, model, env, support_size, rng_key, sims, maxd, gscale):
    s = state[None]
    root = root_fn(env, s, model, support_size, rng_key)
    params = (env, model, support_size)
    out = mctx.gumbel_muzero_policy(
        params=params,
        rng_key=rng_key,
        root=root,
        recurrent_fn=recurrent_fn,
        num_simulations=sims,
        max_depth=maxd,
        gumbel_scale=gscale,
    )
    return out.action[0], out.action_weights[0], root.value[0]


def scan_step(env, model, support_size, rng_key, carry, sims, maxd, gscale):
    st, key = carry
    k1, k2 = jr.split(key)
    a, pa, val = act(st, model, env, support_size, k1, sims, maxd, gscale)
    k2, k3 = jr.split(k2)
    ns, r, d = env.step(st, a, k2)
    return (ns, k3), (st, a, r, d, val, pa)


@eqx.filter_jit
def rollout_episode(st, env, model, support_size, rng_key, steps, sims, maxd, gscale):
    def body(c, _):
        return scan_step(env, model, support_size, rng_key, c, sims, maxd, gscale)

    (fstate, _), data = lax.scan(body, (st, rng_key), None, length=steps)
    s, a, r, d, v, pa = data
    return (
        mcts.Transition(s, a, r, d, v, pa, jnp.zeros_like(r), jnp.ones_like(r)),
        fstate,
    )


def loss_fn(model, batch, env, support_size, rng_key):
    b, l = batch.action.shape
    init_obs = batch.obs[:, 0]
    if model.value_type == "discrete":
        rt = mcts.to_discrete(batch.returns, support_size)

    def loop(c, t):
        tl, vl, pl, obs, rng = c
        v_logits, p_logits = vmap(model)(obs)
        tv = rt[:, t] if model.value_type == "discrete" else batch.returns[:, t, None]
        tp = batch.action_probs[:, t]
        al = batch.action[:, t]
        if model.value_type == "discrete":
            lv = jnp.mean(optax.softmax_cross_entropy(v_logits, tv))
        else:
            lv = jnp.mean((v_logits.squeeze(-1) - tv.squeeze(-1)) ** 2)
        lpi = jnp.mean(optax.softmax_cross_entropy(p_logits, tp))
        rng, sk = jr.split(rng)
        ks = jr.split(sk, b)
        nx, rw, dn = vmap(env.step)(obs, al, ks)
        return (tl + lv + lpi, vl + lv, pl + lpi, nx, rng), None

    (tl, vl, pl, _, _), _ = lax.scan(
        loop, (0.0, 0.0, 0.0, init_obs, rng_key), jnp.arange(l)
    )
    l2 = 0.5 * sum(
        jnp.sum(jnp.square(p))
        for p in jax.tree_util.tree_leaves(eqx.filter(model, eqx.is_array))
    )
    fin = tl / l + 1e-4 * l2
    return fin, {
        "total_loss": fin,
        "value_loss": vl / l,
        "policy_loss": pl / l,
        "l2_loss": 1e-4 * l2,
    }


@eqx.filter_jit
def train_step(model, batch, env, optim, opt_state, support_size, rng_key):
    (ls, aux), grads = eqx.filter_value_and_grad(loss_fn, has_aux=True)(
        model, batch, env, support_size, rng_key
    )
    up, opt_state = optim.update(grads, opt_state, model)
    model = eqx.apply_updates(model, up)
    return model, opt_state, aux


def main():
    args = parse_args()
    wandb.init(project="muzero-project", name=args.run_name)
    wandb.config.update(vars(args))
    key = jr.PRNGKey(args.seed)
    env = mcts.Pong()
    st = env.reset(key)
    buf = mcts.Buffer()
    if args.value_type == "discrete":
        sup = args.support_size
    else:
        sup = 1
    mk, key = jr.split(key)
    model = Policy(
        env.observation_shape,
        env.action_shape,
        args.network_depth,
        args.network_width,
        args.support_size,
        args.value_type,
        mk,
    )
    opt = optax.adam(3e-4)
    ost = opt.init(eqx.filter(model, eqx.is_array))
    env_steps = 0
    for _ in range(args.num_warmup_episodes):
        key, sk = jr.split(key)
        traj, st = rollout_episode(
            st,
            env,
            model,
            sup,
            sk,
            args.num_episode_steps,
            args.num_simulations,
            args.max_depth,
            args.gumbel_scale,
        )

        traj = mcts.compute_returns(traj)
        env_steps += traj.reward.shape[0]
        buf.add(traj, jnp.mean(traj.weight))

    for ep in range(args.num_episodes):
        key, sk = jr.split(key)
        traj, st = rollout_episode(
            st,
            env,
            model,
            sup,
            sk,
            args.num_episode_steps,
            args.num_simulations,
            args.max_depth,
            args.gumbel_scale,
        )
        traj = mcts.compute_returns(traj)
        env_steps += traj.reward.shape[0]
        buf.add(traj, jnp.mean(traj.weight))
        ep_rew = float(jnp.sum(traj.reward))
        wandb.log({"episode_reward": ep_rew, "env_steps": env_steps}, commit=False)
        pol_ent = -jnp.mean(
            jnp.sum(traj.action_probs * jnp.log(traj.action_probs + 1e-12), axis=-1)
        )
        wandb.log({"policy_entropy": float(pol_ent)}, commit=False)
        frames = jnp.stack([env.render(s) for s in traj.obs], axis=0)
        frames = (frames * 255).clip(0, 255).astype(jnp.uint8)
        frames = jnp.transpose(frames, (0, 3, 1, 2))
        wandb.log({"episode_video": wandb.Video(np.array(frames), fps=20)})
        for _ in range(args.num_train_epochs):
            key, sk = jr.split(key)
            batch = buf.sample(sk, batch_size=32, steps=10)
            key, sk = jr.split(key)
            model, ost, aux = train_step(model, batch, env, opt, ost, sup, sk)
            wandb.log({**{k: float(v) for k, v in aux.items()}, "env_steps": env_steps})
    wandb.finish()


if __name__ == "__main__":
    main()
