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


def l2_loss(model: eqx.Module) -> jnp.ndarray:
    """
    Simple L2 penalty on all parameter tensors.
    """
    return 0.5 * sum(
        jnp.sum(jnp.square(p)) for p in jtu.tree_leaves(eqx.filter(model, eqx.is_array))
    )


def kl_loss(
    prior_logits: Array, post_logits: Array, free_nats: float = 0.0, alpha: float = 0.8
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


def mse_loss(out_seq: Array, obs_seq: Array) -> Array:
    return jnp.mean(jnp.sum((out_seq - obs_seq) ** 2, axis=-1))


@eqx.filter_jit
def loss_fn(model: Model, batch: Transition, key: jr.PRNGKey):
    """
    A combined loss for MuZero-style training using an RSSM for latent dynamics.
      - We decode predicted observations and compare to real observations (MSE).
      - We enforce consistency between prior and posterior (KL).
      - We compute a value loss vs. the returns in the buffer.
      - We compute a policy cross-entropy vs. the action probabilities in the buffer.
      - We add a small L2 weight penalty.

    Args:
      model: A Model(Module) with both `.policy` and `.rssm`.
      batch: A Transition object of shape [batch_size, timesteps, ...].
      key:   JAX PRNGKey.

    Returns:
      total_loss: A scalar.
      metrics:    A dict of partial losses and the total.
    """

    batch_obs = batch.obs[..., :2]
    B, T, K, D = batch_obs.shape
    batch_obs = batch_obs.reshape(B, T, K * D)

    # Encode all observations in advance
    # shape: obs_emb is [B, T, rssm_embed_dim]
    obs_emb = vmap(
        lambda obs_t: vmap(lambda o: forward_encoder(model.rssm.encoder, o))(obs_t)
    )(batch_obs)

    # We initialize the posterior state to zeros for each sequence
    # shape: (B,) so we have B separate initial states
    init_post = init_post_state(model.rssm, batch_shape=(B,))

    def scan_step(carry, t):
        """
        carry = (previous_posterior_state, rng_key,
                 sum_mse, sum_kl, sum_value_loss, sum_policy_loss)
        We unroll one step in time for each sequence in the batch.
        """
        (post_prev, rng, sum_mse, sum_kl, sum_v, sum_pi) = carry

        # Split RNG for prior/posterior sampling
        rng, k1, k2 = jr.split(rng, 3)

        # shape (B,); the integer actions from the buffer
        action_t = batch.action[:, t]
        # Convert to one-hot: shape (B, action_dim)
        action_1hot = nn.one_hot(action_t, model.rssm.action_dim)

        # 1) Forward prior
        #    prior_t: shape [B], each an RSSM State
        prior_t = vmap(forward_prior, in_axes=(None, 0, 0, 0))(
            model.rssm.prior, post_prev, action_1hot, jr.split(k1, B)
        )
        # 2) Forward posterior (use real observation embedding)
        #    post_t: shape [B], each an RSSM State
        post_t = vmap(forward_posterior, in_axes=(None, 0, 0, 0))(
            model.rssm.posterior, obs_emb[:, t], prior_t, jr.split(k2, B)
        )

        # 3) Decode the predicted observation from the posterior
        #    rec_obs_t: shape (B, obs_dim)
        rec_obs_t = vmap(forward_decoder, in_axes=(None, 0))(model.rssm.decoder, post_t)
        real_obs_t = batch_obs[:, t]  # shape (B, obs_dim)
        mse_t = jnp.mean(jnp.sum((rec_obs_t - real_obs_t) ** 2, axis=-1))

        # 4) KL between prior_t and post_t
        #    kl_loss returns shape [B], so take mean across batch
        kl_t = jnp.mean(vmap(kl_loss)(prior_t.logits, post_t.logits))

        # 5) Policy/Value heads
        #    Flatten the discrete sample, then concat with state
        sample_flat_t = post_t.sample.reshape(B, -1)
        policy_inp_t = jnp.concatenate([sample_flat_t, post_t.state], axis=-1)
        val_logits_t, pol_logits_t = vmap(policy.forward, in_axes=(None, 0))(
            model.policy, policy_inp_t
        )

        # 6) Value loss vs. returns
        #    returns: shape (B, T), so here we use batch.returns[:, t]
    
        v_loss_t = jnp.mean((val_logits_t - batch.returns[:, t]) ** 2)

        # 7) Policy cross-entropy vs. stored action-probs
        pi_loss_t = jnp.mean(
            optax.softmax_cross_entropy(pol_logits_t, batch.action_probs[:, t])
        )

        # Accumulate
        new_sum_mse = sum_mse + mse_t
        new_sum_kl = sum_kl + kl_t
        new_sum_v = sum_v + v_loss_t
        new_sum_pi = sum_pi + pi_loss_t

        return (post_t, rng, new_sum_mse, new_sum_kl, new_sum_v, new_sum_pi), None

    init_carry = (init_post, key, 0.0, 0.0, 0.0, 0.0)
    (final_post, _, sum_mse, sum_kl, sum_v, sum_pi), _ = lax.scan(
        scan_step, init_carry, jnp.arange(T)
    )

    # Averages across time
    mse_mean = sum_mse / T
    kl_mean = sum_kl / T
    v_mean = sum_v / T
    pi_mean = sum_pi / T

    # Combine RSSM reconstruction losses and policy losses
    rssm_loss = mse_mean + kl_mean
    policy_loss = v_mean + pi_mean

    # Optional L2 weight penalty
    l2_penalty = 1e-4 * l2_loss(model)

    total_loss = rssm_loss + policy_loss + l2_penalty
    metrics = {
        "total_loss": total_loss,
        "rssm_loss": rssm_loss,
        "mse_loss": mse_mean,
        "kl_loss": kl_mean,
        "policy_loss": policy_loss,
        "value_loss": v_mean,
        "action_loss": pi_mean,
        "l2_loss": l2_penalty,
    }
    return total_loss, metrics
