import mctx
import optax
import equinox as eqx

from jax import Array
from jax import numpy as jnp
from jax import random as jr
from jax import nn, vmap

from . import policy, rssm, utils


class Model(eqx.Module):
    policy: policy.Policy
    rssm: rssm.RSSM
    reward_head: eqx.nn.MLP
    reward_dim: int
    reward_offset: int

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
        value_dim: int,
        key: jr.PRNGKey,
        reward_dim: int = 3,
        reward_offset: int = 1,
    ):
        key_policy, key_rssm, key_reward = jr.split(key, 3)
        feature_dim = rssm_state_dim + (rssm_num_discrete * rssm_discrete_dim)
        self.policy = policy.init_policy(
            feature_dim=feature_dim,
            action_dim=action_dim,
            width=policy_hidden_dim,
            depth=policy_depth,
            key=key_policy,
            value_dim=value_dim,
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
            out_size=reward_dim,
            width_size=rssm_hidden_dim,
            depth=2,
            key=key_reward,
        )
        self.reward_dim = reward_dim
        self.reward_offset = reward_offset


def root_fn(key: jr.PRNGKey, model: Model, post: rssm.State) -> mctx.RootFnOutput:
    B = post.sample.shape[0]
    flat_sample = vmap(lambda x: x.flatten())(post.sample)
    features = jnp.concatenate([flat_sample, post.state], axis=-1)
    value_logits, policy_logits = vmap(policy.forward, in_axes=(None, 0))(
        model.policy, features
    )
    val_probs = nn.softmax(value_logits, axis=-1)
    val_bin = jnp.argmax(val_probs, axis=-1)
    continuous_val = vmap(utils.map_class_to_value, in_axes=(0, None, None, None))(
        val_bin,
        model.policy.value_dim,
        model.policy.value_min,
        model.policy.value_max,
    )
    # return mctx.RootFnOutput(
    #     prior_logits=policy_logits,
    #     value=continuous_val,
    #     embedding=post,
    # )

    return mctx.RootFnOutput(
        prior_logits=jnp.ones((B, model.rssm.action_dim,)) / 3,
        value=jnp.zeros((B,)),
        embedding=post,
    )


def recurrent_fn(
    model: Model,
    key: jr.PRNGKey,
    action: Array,
    embedding: rssm.State,
) -> tuple[mctx.RecurrentFnOutput, rssm.State]:
    B = action.shape[0]
    action_1hot = nn.one_hot(action, model.rssm.action_dim)

    prior = vmap(lambda emb, act, k: rssm.forward_prior(model.rssm.prior, emb, act, k))(
        embedding, action_1hot, jr.split(key, B)
    )

    flat_sample = vmap(lambda x: x.flatten())(prior.sample)
    features = jnp.concatenate([flat_sample, prior.state], axis=-1)

    value_logits, policy_logits = vmap(policy.forward, in_axes=(None, 0))(
        model.policy, features
    )

    val_probs = nn.softmax(value_logits, axis=-1)
    val_bin = jnp.argmax(val_probs, axis=-1)
    continuous_val = vmap(utils.map_class_to_value, in_axes=(0, None, None, None))(
        val_bin,
        model.policy.value_dim,
        model.policy.value_min,
        model.policy.value_max,
    )

    rew_logits = vmap(model.reward_head)(features)
    rew_probs = nn.softmax(rew_logits, axis=-1)
    rew_argmax = jnp.argmax(rew_probs, axis=-1)
    predicted_reward = rew_argmax - model.reward_offset

    discount = jnp.ones((B,)) * 0.999

    # return (
    #     mctx.RecurrentFnOutput(
    #         reward=predicted_reward.astype(jnp.float32),
    #         discount=discount,
    #         prior_logits=policy_logits,
    #         value=continuous_val,
    #     ),
    #     prior,
    # )
    return (
        mctx.RecurrentFnOutput(
            reward=predicted_reward.astype(jnp.float32),
            discount=discount,
            prior_logits=jnp.ones((B, model.rssm.action_dim,)) / 3,
            value=jnp.zeros((B,)),
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
