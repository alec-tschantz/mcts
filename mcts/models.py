import mctx
import optax
import equinox as eqx

from jax import Array
from jax import numpy as jnp
from jax import random as jr
from jax import nn, vmap

from . import policy
from . import rssm


class Model(eqx.Module):
    policy: policy.Policy
    rssm: rssm.RSSM
    reward_head: eqx.nn.MLP

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        rssm_embed_dim: int,
        rssm_state_dim: int,
        rssm_num_discrete: int,
        rssm_discrete_dim: int,
        rssm_hidden_dim: int,
        policy_hidden_dim: int,
        policy_depth: int,
        key: jr.PRNGKey,
    ):
        key_policy, key_rssm, key_reward = jr.split(key, 3)
        feature_dim = rssm_state_dim + (rssm_num_discrete * rssm_discrete_dim)
        self.policy = policy.init_policy(
            feature_dim=feature_dim,
            action_dim=action_dim,
            width=policy_hidden_dim,
            depth=policy_depth,
            key=key_policy,
        )
        self.rssm = rssm.init_model(
            obs_dim=obs_dim,
            action_dim=action_dim,
            embed_dim=rssm_embed_dim,
            state_dim=rssm_state_dim,
            num_discrete=rssm_num_discrete,
            discrete_dim=rssm_discrete_dim,
            hidden_dim=rssm_hidden_dim,
            key=key_rssm,
        )
        self.reward_head = eqx.nn.MLP(
            in_size=feature_dim,
            out_size=1,
            width_size=policy_hidden_dim,
            depth=policy_depth,
            key=key_reward,
        )


def root_fn(key: jr.PRNGKey, model: Model, post: rssm.State) -> mctx.RootFnOutput:
    flat_sample = vmap(lambda x: x.flatten())(post.sample)
    features = jnp.concatenate([flat_sample, post.state], axis=-1)
    value, policy_logits = vmap(policy.forward, in_axes=(None, 0))(
        model.policy, features
    )
    return mctx.RootFnOutput(
        prior_logits=policy_logits,
        value=value.squeeze(-1),
        embedding=post,
    )


def recurrent_fn(
    model: Model,
    key: jr.PRNGKey,
    action: Array,
    embedding: rssm.State,
) -> tuple[mctx.RecurrentFnOutput, rssm.State]:
    B = action.shape[0]
    action = nn.one_hot(action, model.rssm.action_dim)
    prior = vmap(lambda emb, act, k: rssm.forward_prior(model.rssm.prior, emb, act, k))(
        embedding, action, jr.split(key, B)
    )
    flat_sample = vmap(lambda x: x.flatten())(prior.sample)
    features = jnp.concatenate([flat_sample, prior.state], axis=-1)
    value, policy_logits = vmap(policy.forward, in_axes=(None, 0))(
        model.policy, features
    )
    rewards = vmap(model.reward_head)(features)
    discounts = jnp.ones((B,))
    return (
        mctx.RecurrentFnOutput(
            reward=rewards.squeeze(-1),
            discount=discounts,
            prior_logits=policy_logits,
            value=value.squeeze(-1),
        ),
        prior,
    )


def action_fn(
    key: jr.PRNGKey,
    model: Model,
    post: rssm.State,
    max_depth: int = 20,
    gumbel_scale: float = 1.0,
    num_simulations: int = 500,
):
    root = root_fn(key, model, post)
    out = mctx.gumbel_muzero_policy(
        params=model,
        rng_key=key,
        root=root,
        recurrent_fn=recurrent_fn,
        num_simulations=num_simulations,
        max_depth=max_depth,
        gumbel_scale=gumbel_scale,
    )
    return out.action, out.action_weights, root.value


def compute_posterior(
    model: Model, post: rssm.State, obs: Array, action: int, key: jr.PRNGKey
):
    keys = jr.split(key, 2)
    action = nn.one_hot(action, model.rssm.action_dim)
    enc_obs = rssm.forward_encoder(model.rssm.encoder, obs)
    prior = rssm.forward_prior(model.rssm.prior, post, action, keys[0])
    return rssm.forward_posterior(model.rssm.posterior, enc_obs, prior, keys[1])
