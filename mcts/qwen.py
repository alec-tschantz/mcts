from typing import Optional, List

import jax
import equinox as eqx

from jax import Array, numpy as jnp, random as jr, lax, nn


class Embedding(eqx.Module):
    weight: Array


class Linear(eqx.Module):
    weight: Array
    bias: Optional[Array]


class RotaryEmbedding(eqx.Module):
    dim: int
    theta: float


class RMSNorm(eqx.Module):
    weight: Array
    eps: float


class Attention(eqx.Module):
    q_proj: Linear
    k_proj: Linear
    v_proj: Linear
    o_proj: Linear
    num_heads: int
    head_dim: int
    num_key_value_heads: int
    attn_dropout: eqx.nn.Dropout


class Dense(eqx.Module):
    gate_proj: Linear
    up_proj: Linear
    down_proj: Linear


class DecoderLayer(eqx.Module):
    self_attn: Attention
    mlp: Dense
    input_layernorm: RMSNorm
    post_attention_layernorm: RMSNorm
    residual_dropout: eqx.nn.Dropout


class QwenModel(eqx.Module):
    input_proj: Linear
    layers: List[DecoderLayer]
    norm: RMSNorm
    rotary_emb: RotaryEmbedding
    output_proj: Linear

    embed_dropout_in: eqx.nn.Dropout
    embed_dropout_out: eqx.nn.Dropout

    obs_dim: int
    action_dim: int
    reward_dim: int


def loss_fn(model, prev_obs, prev_actions, next_obs, next_rewards, key):
    B, T, K, D = prev_obs.shape
    prev_obs = prev_obs.reshape((B, T, K * D))

    output = forward(model, prev_obs, prev_actions, key=key, inference=False)
    pred_next_obs, pred_next_rewards = (
        output[..., : -model.reward_dim],
        output[..., -model.reward_dim :],
    )
    pred_next_obs = pred_next_obs.reshape((B, T, K, D))

    obs_loss = jnp.mean((pred_next_obs - next_obs) ** 2)
    reward_loss = jnp.mean((pred_next_rewards - next_rewards) ** 2)
    return obs_loss + reward_loss, {"obs_loss": obs_loss, "reward_loss": reward_loss}


def forward_linear(l: Linear, x: Array) -> Array:
    y = jnp.dot(x, l.weight.T)
    return y + l.bias if l.bias is not None else y


def forward_rotary_embedding(
    r: RotaryEmbedding, hidden: Array, position_ids: Array
) -> tuple[Array, Array]:
    b, s, _ = hidden.shape
    inv_freq = 1.0 / (r.theta ** (jnp.arange(0, r.dim, 2) / r.dim))
    freqs = position_ids.reshape(b, s, 1) * inv_freq[None, None, :]
    emb = jnp.concatenate((freqs, freqs), axis=-1)
    return jnp.cos(emb), jnp.sin(emb)


def forward_rms_norm(r: RMSNorm, hidden: Array) -> Array:
    variance = jnp.mean(hidden**2, axis=-1, keepdims=True)
    x = hidden * lax.rsqrt(variance + r.eps)
    return r.weight * x


def forward_attention(
    a: Attention,
    hidden: Array,
    cos: Array,
    sin: Array,
    attention_mask: Optional[Array],
    *,
    key: jr.PRNGKey,
    inference: bool,
) -> Array:

    b, seqlen, _ = hidden.shape

    def rotate_half(u: Array) -> Array:
        u1, u2 = jnp.split(u, 2, axis=-1)
        return jnp.concatenate((-u2, u1), axis=-1)

    def apply_rotary_pos_emb(
        q: Array, k: Array, c: Array, s: Array
    ) -> tuple[Array, Array]:
        c = jnp.expand_dims(c, axis=1)
        s = jnp.expand_dims(s, axis=1)
        q_ = (q * c) + (rotate_half(q) * s)
        k_ = (k * c) + (rotate_half(k) * s)
        return q_, k_

    q = forward_linear(a.q_proj, hidden)
    k = forward_linear(a.k_proj, hidden)
    v = forward_linear(a.v_proj, hidden)

    q = q.reshape(b, seqlen, a.num_heads, a.head_dim).transpose(0, 2, 1, 3)
    k = k.reshape(b, seqlen, a.num_key_value_heads, a.head_dim).transpose(0, 2, 1, 3)
    v = v.reshape(b, seqlen, a.num_key_value_heads, a.head_dim).transpose(0, 2, 1, 3)

    q, k = apply_rotary_pos_emb(q, k, cos, sin)

    if a.num_key_value_heads != a.num_heads:
        factor = a.num_heads // a.num_key_value_heads
        k = jnp.repeat(k, repeats=factor, axis=1)
        v = jnp.repeat(v, repeats=factor, axis=1)

    scores = jnp.einsum("bhqd,bhkd->bhqk", q, k) / jnp.sqrt(a.head_dim)

    causal_mask = jnp.tril(jnp.ones((seqlen, seqlen)))
    causal_mask = causal_mask[None, None, :, :]
    if attention_mask is not None:
        attention_mask = jnp.expand_dims(attention_mask, axis=(1, 2))
        mask = jnp.minimum(causal_mask, attention_mask)
    else:
        mask = causal_mask

    scores = jnp.where(mask == 0, float("-inf"), scores)

    probs = nn.softmax(scores, axis=-1)
    attn_key, _ = jr.split(key, 2)
    probs = a.attn_dropout(probs, key=attn_key, inference=inference)

    out = jnp.einsum("bhqk,bhkd->bhqd", probs, v)
    out = out.transpose(0, 2, 1, 3).reshape(b, seqlen, -1)

    return forward_linear(a.o_proj, out)


def forward_mlp(m: Dense, x: Array) -> Array:
    gx = forward_linear(m.gate_proj, x)
    ux = forward_linear(m.up_proj, x)
    return forward_linear(m.down_proj, nn.silu(gx) * ux)


def forward_decoder(
    d: DecoderLayer,
    hidden: Array,
    cos: Array,
    sin: Array,
    attention_mask: Optional[Array],
    *,
    key: jr.PRNGKey,
    inference: bool,
) -> Array:
    residual = hidden
    hidden = forward_rms_norm(d.input_layernorm, hidden)

    attn_key, residual_key, subkey = jr.split(key, 3)
    hidden = forward_attention(
        d.self_attn, hidden, cos, sin, attention_mask, key=attn_key, inference=inference
    )
    hidden = residual + hidden

    residual = hidden
    hidden = forward_rms_norm(d.post_attention_layernorm, hidden)
    hidden = forward_mlp(d.mlp, hidden)
    hidden = d.residual_dropout(hidden, key=residual_key, inference=inference)
    hidden = residual + hidden
    return hidden


def forward(
    model: QwenModel,
    observation: Array,
    action: Array,
    attention_mask: Optional[Array] = None,
    position_ids: Optional[Array] = None,
    *,
    key: jr.PRNGKey,
    inference: bool = False,
) -> Array:
    x = jnp.concatenate([observation, action], axis=-1)
    b, s, _ = x.shape
    if position_ids is None:
        position_ids = jnp.tile(jnp.arange(s)[None, :], (b, 1))

    key_in, key_layers, key_out = jr.split(key, 3)

    hidden = forward_linear(model.input_proj, x)
    hidden = model.embed_dropout_in(hidden, key=key_in, inference=inference)

    cos, sin = forward_rotary_embedding(model.rotary_emb, hidden, position_ids)

    layer_key_seq = jr.split(key_layers, len(model.layers))
    for layer, layer_key in zip(model.layers, layer_key_seq):
        hidden = forward_decoder(
            layer,
            hidden,
            cos,
            sin,
            attention_mask,
            key=layer_key,
            inference=inference,
        )

    hidden = forward_rms_norm(model.norm, hidden)

    output = forward_linear(model.output_proj, hidden)
    output = model.embed_dropout_out(output, key=key_out, inference=inference)
    return output


def init(
    key,
    obs_dim=6,
    action_dim=6,
    reward_dim=3,
    hidden_size=256,
    num_layers=4,
    num_heads=4,
    num_key_value_heads=4,
    rope_theta=5000.0,
    rms_norm_eps=1e-5,
    dropout=0.05,
) -> QwenModel:
    keys = jr.split(key, 3 + num_layers)

    input_dim = obs_dim + action_dim
    output_dim = obs_dim + reward_dim

    inp_proj = init_linear(keys[0], input_dim, hidden_size)
    final_norm = init_rms_norm(hidden_size, rms_norm_eps)
    head_dim = hidden_size // num_heads
    rot_emb = init_rotary_embedding(head_dim, rope_theta)

    layers = [
        init_decoder_layer(
            keys[i + 1],
            hidden_size,
            num_heads,
            num_key_value_heads,
            rms_norm_eps,
            dropout,
        )
        for i in range(num_layers)
    ]

    out_proj = init_linear(keys[-1], hidden_size, output_dim, bias=True)

    embed_dropout_in = eqx.nn.Dropout(p=dropout, inference=False)
    embed_dropout_out = eqx.nn.Dropout(p=dropout, inference=False)

    return QwenModel(
        obs_dim=obs_dim,
        action_dim=action_dim,
        reward_dim=reward_dim,
        input_proj=inp_proj,
        layers=layers,
        norm=final_norm,
        rotary_emb=rot_emb,
        output_proj=out_proj,
        embed_dropout_in=embed_dropout_in,
        embed_dropout_out=embed_dropout_out,
    )


def init_linear(
    key: jr.PRNGKey, in_dim: int, out_dim: int, bias: bool = True
) -> Linear:
    k1, k2 = jr.split(key)
    weight = jr.normal(k1, (out_dim, in_dim)) * jnp.sqrt(2.0 / (in_dim + out_dim))
    b = jr.normal(k2, (out_dim,)) * 0.01 if bias else None
    return Linear(weight=weight, bias=b)


def init_rms_norm(hidden_dim: int, eps: float) -> RMSNorm:
    weight = jnp.ones((hidden_dim,))
    return RMSNorm(weight=weight, eps=eps)


def init_rotary_embedding(head_dim: int, rope_theta: float) -> RotaryEmbedding:
    return RotaryEmbedding(dim=head_dim, theta=rope_theta)


def init_attention(
    key: jr.PRNGKey,
    hidden_size: int,
    num_heads: int,
    num_key_value_heads: int,
    dropout: float,
) -> Attention:
    head_dim = hidden_size // num_heads
    k1, k2, k3, k4 = jr.split(key, 4)
    q_proj = init_linear(k1, hidden_size, hidden_size)
    k_proj = init_linear(k2, hidden_size, hidden_size)
    v_proj = init_linear(k3, hidden_size, hidden_size)
    o_proj = init_linear(k4, hidden_size, hidden_size, bias=False)

    attn_dropout = eqx.nn.Dropout(p=dropout, inference=False)

    return Attention(
        q_proj=q_proj,
        k_proj=k_proj,
        v_proj=v_proj,
        o_proj=o_proj,
        num_heads=num_heads,
        head_dim=head_dim,
        num_key_value_heads=num_key_value_heads,
        attn_dropout=attn_dropout,
    )


def init_dense(key: jr.PRNGKey, hidden_size: int) -> Dense:
    k1, k2, k3 = jr.split(key, 3)
    gate_proj = init_linear(k1, hidden_size, hidden_size * 2, bias=False)
    up_proj = init_linear(k2, hidden_size, hidden_size * 2, bias=False)
    down_proj = init_linear(k3, hidden_size * 2, hidden_size, bias=False)
    return Dense(gate_proj=gate_proj, up_proj=up_proj, down_proj=down_proj)


def init_decoder_layer(
    key: jr.PRNGKey,
    hidden_size: int,
    num_heads: int,
    num_key_value_heads: int,
    rms_norm_eps: float,
    dropout: float,
) -> DecoderLayer:
    k1, k2, k3, k4 = jr.split(key, 4)
    attn = init_attention(k1, hidden_size, num_heads, num_key_value_heads, dropout)
    mlp = init_dense(k2, hidden_size)
    in_ln = init_rms_norm(hidden_size, rms_norm_eps)
    post_ln = init_rms_norm(hidden_size, rms_norm_eps)
    residual_dropout = eqx.nn.Dropout(p=dropout, inference=False)

    return DecoderLayer(
        self_attn=attn,
        mlp=mlp,
        input_layernorm=in_ln,
        post_attention_layernorm=post_ln,
        residual_dropout=residual_dropout,
    )
