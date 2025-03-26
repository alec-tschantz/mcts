import optax
import equinox as eqx

from jax import Array
from jax import numpy as jnp
from jax import random as jr
from jax import nn, lax, vmap, tree_util as jtu

from mcts.buffer import Transition
from mcts.rssm import forward_prior, forward_posterior, forward_decoder, forward_encoder
from mcts.rssm import init_post_state
from mcts.rssm import RSSM, State
from mcts import policy
from mcts.models import Model

from . import utils


def l2_loss(model: eqx.Module) -> jnp.ndarray:
    return 0.5 * sum(
        jnp.sum(jnp.square(p)) for p in jtu.tree_leaves(eqx.filter(model, eqx.is_array))
    )


def kl_loss(
    prior_logits: Array, post_logits: Array, free_nats: float = 0.1, alpha: float = 0.8
) -> Array:
    kl_lhs = optax.losses.kl_divergence_with_log_targets(
        lax.stop_gradient(post_logits), prior_logits
    ).sum(axis=-1)
    kl_rhs = optax.losses.kl_divergence_with_log_targets(
        post_logits, lax.stop_gradient(prior_logits)
    ).sum(axis=-1)
    kl_lhs, kl_rhs = jnp.mean(kl_lhs), jnp.mean(kl_rhs)
    if free_nats > 0.0:
        kl_lhs = jnp.maximum(kl_lhs, free_nats)
        kl_rhs = jnp.maximum(kl_rhs, free_nats)
    return (alpha * kl_lhs) + ((1 - alpha) * kl_rhs)


def mse_loss(pred: Array, target: Array) -> Array:
    return jnp.mean(jnp.sum((pred - target) ** 2, axis=-1))


@eqx.filter_jit
def loss_fn(model: Model, batch: Transition, key: jr.PRNGKey):
    # batch_obs = batch.obs[..., :2]
    batch_obs = batch.obs
    B, T, K, D = batch_obs.shape
    batch_obs = batch_obs.reshape(B, T, K * D)
    obs_emb = vmap(
        lambda obs_t: vmap(lambda o: forward_encoder(model.rssm.encoder, o))(obs_t)
    )(batch_obs)
    init_post = init_post_state(model.rssm, batch_shape=(B,))

    def scan_step(carry, t):
        post_prev, rng, sum_mse, sum_kl, sum_v, sum_pi, sum_r = carry
        rng, k1, k2, k3 = jr.split(rng, 4)
        action_t = batch.action[:, t]
        action_1hot = nn.one_hot(action_t, model.rssm.action_dim)
        prior_t = vmap(forward_prior, in_axes=(None, 0, 0, 0))(
            model.rssm.prior, post_prev, action_1hot, jr.split(k1, B)
        )
        post_t = vmap(forward_posterior, in_axes=(None, 0, 0, 0))(
            model.rssm.posterior, obs_emb[:, t], prior_t, jr.split(k2, B)
        )
        rec_obs_t = vmap(forward_decoder, in_axes=(None, 0))(model.rssm.decoder, post_t)

        real_obs_t = batch_obs[:, t]
        mse_t = jnp.mean(jnp.sum((rec_obs_t - real_obs_t) ** 2, axis=-1))
        kl_t = jnp.mean(vmap(kl_loss)(prior_t.logits, post_t.logits))

        sample_flat_t = post_t.sample.reshape(B, -1)
        policy_inp_t = jnp.concatenate([sample_flat_t, post_t.state], axis=-1)
        vmap_policy = vmap(policy.forward, in_axes=(None, 0))
        val_logits_t, pol_logits_t = vmap_policy(model.policy, policy_inp_t)

        val_idx_t = vmap(
            lambda r: utils.map_value_to_class(
                r,
                model.policy.value_dim,
                model.policy.value_min,
                model.policy.value_max,
            )
        )(batch.returns[:, t])
        val_1hot_t = nn.one_hot(val_idx_t, model.policy.value_dim)
        value_ce_t = jnp.mean(optax.softmax_cross_entropy(val_logits_t, val_1hot_t))
        # val_t = batch.returns[:, t]
        # value_ce_t = jnp.mean(jnp.sum((val_logits_t - val_t) ** 2, axis=-1))
        # value_ce_t = 0.0

        pi_loss_t = jnp.mean(
            optax.softmax_cross_entropy(pol_logits_t, batch.action_probs[:, t])
        )
        prior_features = jnp.concatenate(
            [prior_t.sample.reshape(B, -1), prior_t.state], axis=-1
        )

        rew_logits_t = vmap(model.reward_head)(prior_features)
        reward_t = batch.reward[:, t]
        reward_idx_t = utils.map_reward_to_class(
            reward_t, model.reward_offset, model.reward_dim
        )
        reward_1hot_t = nn.one_hot(reward_idx_t, model.reward_dim)
        ce_t = jnp.mean(optax.softmax_cross_entropy(rew_logits_t, reward_1hot_t))
        return (
            post_t,
            rng,
            sum_mse + mse_t,
            sum_kl + kl_t,
            sum_v + value_ce_t,
            sum_pi + pi_loss_t,
            sum_r + ce_t,
        ), None

    init_carry = (init_post, key, 0.0, 0.0, 0.0, 0.0, 0.0)
    (final_post, _, sum_mse, sum_kl, sum_v, sum_pi, sum_r), _ = lax.scan(
        scan_step, init_carry, jnp.arange(T)
    )
    mse_mean = sum_mse / T
    kl_mean = sum_kl / T
    v_mean = sum_v / T
    pi_mean = sum_pi / T
    r_mean = sum_r / T
    rssm_loss = mse_mean + kl_mean
    policy_loss = v_mean + pi_mean
    reward_loss = r_mean
    l2_penalty = 1e-4 * l2_loss(model)
    total_loss = rssm_loss + policy_loss + reward_loss + l2_penalty
    metrics = {
        "total_loss": total_loss,
        "rssm_loss": rssm_loss,
        "mse_loss": mse_mean,
        "kl_loss": kl_mean,
        "policy_loss": policy_loss,
        "value_loss": v_mean,
        "action_loss": pi_mean,
        "reward_loss": reward_loss,
        "l2_loss": l2_penalty,
    }
    return total_loss, metrics
