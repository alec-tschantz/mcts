import optax
import equinox as eqx
from typing import NamedTuple, Tuple
from jax import Array, numpy as jnp, random as jr, nn, lax, vmap


class Prior(eqx.Module):
    rnn_cell: eqx.nn.GRUCell
    fc_input: eqx.nn.Linear
    fc_state: eqx.nn.Linear
    fc_logits: eqx.nn.Linear
    norm_input: eqx.nn.RMSNorm
    norm_state: eqx.nn.RMSNorm
    num_discrete: int
    discrete_dim: int


class Posterior(eqx.Module):
    fc_input: eqx.nn.Linear
    fc_logits: eqx.nn.Linear
    norm_input: eqx.nn.RMSNorm
    num_discrete: int
    discrete_dim: int


class Encoder(eqx.Module):
    fc1: eqx.nn.Linear
    fc2: eqx.nn.Linear
    norm1: eqx.nn.RMSNorm
    norm2: eqx.nn.RMSNorm


class Decoder(eqx.Module):
    fc1: eqx.nn.Linear
    fc2: eqx.nn.Linear
    norm1: eqx.nn.RMSNorm
    norm2: eqx.nn.RMSNorm


class Model(eqx.Module):
    prior: Prior
    posterior: Posterior
    encoder: Encoder
    decoder: Decoder
    logit_dim: int
    state_dim: int


class State(NamedTuple):
    logits: Array
    sample: Array
    state: Array


@eqx.filter_jit
def loss_fn(params, obs_seq, action_seq, key):
    _forward = lambda o, a, k: forward_model(params, o, a, k)
    subkeys = jr.split(key, obs_seq.shape[0])
    pred_seq, post, prior = vmap(_forward)(obs_seq, action_seq, subkeys)
    _mse_loss = mse_loss(obs_seq, pred_seq)
    _kl_loss = kl_loss(post.logits, prior.logits)
    return _mse_loss + _kl_loss, {"mse_loss": _mse_loss, "kl_loss": _kl_loss}


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


def forward_prior(
    prior: Prior, prev_post: State, action: Array, key: jr.PRNGKey
) -> State:
    prev_sample = prev_post.sample.reshape(-1)
    feat = jnp.concatenate([action, prev_sample], axis=-1)
    hidden = prior.norm_input(nn.silu(prior.fc_input(feat)))
    state = prior.rnn_cell(hidden, prev_post.state)
    hidden = prior.norm_state(nn.silu(prior.fc_state(state)))
    logits = prior.fc_logits(hidden).reshape(prior.num_discrete, prior.discrete_dim)
    logits, sample = sample_logits(logits, key)
    return State(logits, sample, state)


def forward_posterior(
    post: Posterior, obs_emb: Array, prior_state: State, key: jr.PRNGKey
) -> State:
    inp = jnp.concatenate([obs_emb, prior_state.state], axis=-1)
    hidden = post.norm_input(nn.silu(post.fc_input(inp)))
    logits = post.fc_logits(hidden).reshape(post.num_discrete, post.discrete_dim)
    logits, sample = sample_logits(logits, key)
    return State(logits, sample, prior_state.state)


def forward_encoder(encoder: Encoder, obs: Array) -> Array:
    hidden = encoder.norm1(nn.silu(encoder.fc1(obs)))
    out = encoder.fc2(hidden)
    out = encoder.norm2(out)
    return out


def forward_decoder(decoder: Decoder, post: State) -> Array:
    inp = jnp.concatenate([post.sample.reshape(-1), post.state], axis=-1)
    hidden = decoder.norm1(nn.silu(decoder.fc1(inp)))
    out = decoder.fc2(hidden)
    out = decoder.norm2(out)
    return out


def forward_model(
    model: Model, obs_seq: Array, action_seq: Array, key: jr.PRNGKey
) -> Tuple[Array, State, State]:
    obs_emb_seq = vmap(lambda o: forward_encoder(model.encoder, o))(obs_seq)
    init_post = init_post_state(model)
    post_seq, prior_seq = rollout(
        model.prior, model.posterior, obs_emb_seq, init_post, action_seq, key
    )
    out_seq = vmap(lambda s: forward_decoder(model.decoder, s))(post_seq)
    return out_seq, post_seq, prior_seq

@eqx.filter_jit
def rollout(
    prior: Prior,
    post: Posterior,
    obs_emb_seq: Array,
    init_post: State,
    action_seq: Array,
    key: jr.PRNGKey,
) -> Tuple[Array, Array]:
    def step(prev_post, step_data):
        k_, ob_, act_ = step_data
        keys = jr.split(k_, 2)
        prior_ = forward_prior(prior, prev_post, act_, keys[0])
        post_ = forward_posterior(post, ob_, prior_, keys[1])
        return post_, (post_, prior_)

    keys = jr.split(key, action_seq.shape[0])
    final_post, (post_seq, prior_seq) = lax.scan(
        step, init_post, (keys, obs_emb_seq, action_seq)
    )
    return post_seq, prior_seq

@eqx.filter_jit
def rollout_prior(
    prior: Prior, init_post: State, action_seq: Array, key: jr.PRNGKey
) -> Array:
    def step(prev_s, step_data):
        k_, act_ = step_data
        new_s = forward_prior(prior, prev_s, act_, k_)
        return new_s, new_s

    keys = jr.split(key, action_seq.shape[0])
    _, states = lax.scan(step, init_post, (keys, action_seq))
    return states


def sample_logits(
    logits: Array, key: jr.PRNGKey, unimix: float = 0.01
) -> Tuple[Array, Array]:
    probs = nn.softmax(logits, axis=-1)
    uniform = jnp.ones_like(probs) / probs.shape[-1]
    probs = (1.0 - unimix) * probs + unimix * uniform
    dist_logits = jnp.log(probs + 1e-8)
    sample = jr.categorical(key, dist_logits, axis=-1)
    onehot = nn.one_hot(sample, probs.shape[-1])
    st_sample = onehot + (probs - lax.stop_gradient(probs))
    return dist_logits, st_sample


def init_post_state(model: Model, batch_shape: tuple = ()) -> State:
    post = model.posterior
    return State(
        jnp.zeros(batch_shape + (post.num_discrete, post.discrete_dim)),
        jnp.zeros(batch_shape + (post.num_discrete, post.discrete_dim)),
        jnp.zeros(batch_shape + (model.state_dim,)),
    )


def init_model(
    obs_dim: int,
    action_dim: int,
    embed_dim: int,
    state_dim: int,
    num_discrete: int,
    discrete_dim: int,
    hidden_dim: int,
    key: jr.PRNGKey,
) -> Model:

    k1, k2, k3, k4 = jr.split(key, 4)
    return Model(
        prior=init_prior(
            action_dim, num_discrete, discrete_dim, state_dim, hidden_dim, k1
        ),
        posterior=init_posterior(
            embed_dim, num_discrete, discrete_dim, state_dim, hidden_dim, k2
        ),
        encoder=init_encoder(obs_dim, embed_dim, hidden_dim, k3),
        decoder=init_decoder(
            num_discrete, discrete_dim, state_dim, obs_dim, hidden_dim, k4
        ),
        logit_dim=num_discrete * discrete_dim,
        state_dim=state_dim,
    )


def init_prior(
    action_dim: int,
    num_discrete: int,
    discrete_dim: int,
    state_dim: int,
    hidden_dim: int,
    key: jr.PRNGKey,
) -> Prior:
    logit_dim = num_discrete * discrete_dim
    k1, k2, k3, k4, k5, k6 = jr.split(key, 6)
    return Prior(
        fc_input=eqx.nn.Linear(
            in_features=action_dim + logit_dim, out_features=hidden_dim, key=k1
        ),
        norm_input=eqx.nn.RMSNorm(shape=hidden_dim),
        rnn_cell=eqx.nn.GRUCell(input_size=hidden_dim, hidden_size=state_dim, key=k3),
        fc_state=eqx.nn.Linear(in_features=state_dim, out_features=hidden_dim, key=k4),
        norm_state=eqx.nn.RMSNorm(shape=hidden_dim),
        fc_logits=eqx.nn.Linear(in_features=hidden_dim, out_features=logit_dim, key=k6),
        num_discrete=num_discrete,
        discrete_dim=discrete_dim,
    )


def init_posterior(
    embed_dim: int,
    num_discrete: int,
    discrete_dim: int,
    state_dim: int,
    hidden_dim: int,
    key: jr.PRNGKey,
) -> Posterior:
    logit_dim = num_discrete * discrete_dim
    k1, k2, k3 = jr.split(key, 3)
    return Posterior(
        fc_input=eqx.nn.Linear(
            in_features=state_dim + embed_dim, out_features=hidden_dim, key=k1
        ),
        norm_input=eqx.nn.RMSNorm(shape=hidden_dim),
        fc_logits=eqx.nn.Linear(in_features=hidden_dim, out_features=logit_dim, key=k3),
        num_discrete=num_discrete,
        discrete_dim=discrete_dim,
    )


def init_encoder(
    obs_dim: int,
    embed_dim: int,
    hidden_dim: int,
    key: jr.PRNGKey,
) -> Encoder:
    k1, k2, k3, k4 = jr.split(key, 4)
    return Encoder(
        fc1=eqx.nn.Linear(in_features=obs_dim, out_features=hidden_dim, key=k1),
        norm1=eqx.nn.RMSNorm(shape=hidden_dim),
        fc2=eqx.nn.Linear(in_features=hidden_dim, out_features=embed_dim, key=k3),
        norm2=eqx.nn.RMSNorm(shape=embed_dim),
    )


def init_decoder(
    num_discrete: int,
    discrete_dim: int,
    state_dim: int,
    obs_dim: int,
    hidden_dim: int,
    key: jr.PRNGKey,
) -> Decoder:
    logit_dim = num_discrete * discrete_dim
    k1, k2, k3, k4 = jr.split(key, 4)
    return Decoder(
        fc1=eqx.nn.Linear(
            in_features=state_dim + logit_dim, out_features=hidden_dim, key=k1
        ),
        norm1=eqx.nn.RMSNorm(shape=hidden_dim),
        fc2=eqx.nn.Linear(in_features=hidden_dim, out_features=obs_dim, key=k3),
        norm2=eqx.nn.RMSNorm(shape=obs_dim),
    )
