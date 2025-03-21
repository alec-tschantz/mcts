import equinox as eqx
import jax
import jax.numpy as jnp
import jax.nn as nn
import jax.random as jr
from jax import lax, vmap
import jax.tree_util as jtu
import matplotlib.pyplot as plt


def kaiming_init(key, shape):
    fan_in = shape[1]
    std = jnp.sqrt(2.0 / fan_in)
    return jr.normal(key, shape) * std


class Linear(eqx.Module):
    weight: jnp.ndarray
    bias: jnp.ndarray

    def __init__(self, in_dim, out_dim, key):
        k_w, k_b = jr.split(key)
        self.weight = kaiming_init(k_w, (out_dim, in_dim))
        self.bias = jnp.zeros(out_dim)

    def __call__(self, x):
        return jnp.dot(self.weight, x) + self.bias


class GRUCell(eqx.Module):
    W_r: jnp.ndarray
    W_z: jnp.ndarray
    W_h: jnp.ndarray

    def __init__(self, input_dim, hidden_dim, key):
        k1, k2, k3 = jr.split(key, 3)
        self.W_r = kaiming_init(k1, (hidden_dim + input_dim, hidden_dim))
        self.W_z = kaiming_init(k2, (hidden_dim + input_dim, hidden_dim))
        self.W_h = kaiming_init(k3, (hidden_dim + input_dim, hidden_dim))

    def __call__(self, h, x):
        concat = jnp.concatenate([h, x], axis=-1)
        r = nn.sigmoid(jnp.dot(concat, self.W_r))
        z = nn.sigmoid(jnp.dot(concat, self.W_z))
        concat_r = jnp.concatenate([r * h, x], axis=-1)
        h_tilde = nn.tanh(jnp.dot(concat_r, self.W_h))
        new_h = (1 - z) * h + z * h_tilde
        return new_h


class RSLDS(eqx.Module):
    gru: GRUCell
    object_heads: list[Linear]
    action_dim: int
    obs_shape: tuple[int, int]
    hidden_dim: int
    num_modes: int
    weight_decay: float
    temp: float
    vel_mult: float

    def __init__(
        self,
        obs_shape,
        action_dim,
        key,
        hidden_dim=256,
        num_modes=7,
        weight_decay=1e-4,
        temp=1.0,
        vel_mult=10.0,
    ):
        self.obs_shape = obs_shape
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        self.num_modes = num_modes
        self.weight_decay = weight_decay
        self.temp = temp
        self.vel_mult = vel_mult

        num_objects = obs_shape[0]
        obs_dim = obs_shape[0] * obs_shape[1]
        switch_input_dim = num_objects * num_modes

        k1, k2, *head_keys = jr.split(key, 2 + num_objects)
        self.gru = GRUCell(obs_dim + action_dim + switch_input_dim, hidden_dim, k1)
        self.object_heads = [Linear(hidden_dim, num_modes, k) for k in head_keys]

    def __call__(self, obs, action, carry=None, key=None):
        num_objects = self.obs_shape[0]

        obs_flat = obs.reshape(-1)
        a_onehot = nn.one_hot(action, self.action_dim)

        if carry is None:
            hidden = jnp.zeros(self.hidden_dim)
            prev_switch = jnp.full((num_objects, self.num_modes), 1.0 / self.num_modes)
        else:
            hidden, prev_switch = carry

        prev_switch_flat = prev_switch.reshape(-1)
        inputs = jnp.concatenate([obs_flat, a_onehot, prev_switch_flat], axis=-1)

        new_hidden = self.gru(hidden, inputs)

        logits = jnp.stack([head(new_hidden) for head in self.object_heads])

        if key is None:
            key = jr.PRNGKey(0)
        gumbel_noise = jr.gumbel(key, logits.shape)
        gumbel_logits = (logits + gumbel_noise) / self.temp
        probs = nn.softmax(gumbel_logits, axis=-1)

        velocity_table = jnp.array(
            [
                [-0.05, 0.05],
                [-0.05, -0.05],
                [0.05, 0.05],
                [0.05, -0.05],
                [0.0, -0.1],
                [0.0, 0.1],
                [0.0, 0.0],
            ]
        )
        velocity_table = jnp.stack([velocity_table, velocity_table, velocity_table])
        vel = jnp.einsum("ok,okd->od", probs, velocity_table)

        pos = obs[..., 0:2]
        new_pos = pos + vel
        next_obs = jnp.concatenate([new_pos, vel], axis=-1)

        new_carry = (new_hidden, probs)
        return next_obs, new_carry, logits


def l2_loss(model: eqx.Module) -> jnp.ndarray:
    return sum(
        jnp.sum(jnp.square(p)) for p in jtu.tree_leaves(eqx.filter(model, eqx.is_array))
    )


def loss_fn(model, batch, key):
    B, T = batch.action.shape
    param_norm = l2_loss(model)
    reg_loss = model.weight_decay * param_norm

    def step(carry, t):
        total_mse, total_vel, hidden, prev_switch, rng = carry
        obs_t = batch.obs[:, t]
        action_t = batch.action[:, t]
        next_obs = batch.obs[:, t + 1]
        rng, step_key = jr.split(rng)

        def apply_model(o, a, h, p, k):
            carry_in = (h, p)
            pred, (new_h, new_p), _ = model(o, a, carry=carry_in, key=k)
            return pred, new_h, new_p

        step_keys = jr.split(step_key, B)
        preds, new_hidden, new_prev_switch = vmap(apply_model)(
            obs_t, action_t, hidden, prev_switch, step_keys
        )

        mse = jnp.mean((preds[..., 0:2] - next_obs[..., 0:2]) ** 2)
        vel_loss = jnp.mean((preds[..., 2:] - next_obs[..., 2:]) ** 2)

        return (
            total_mse + mse,
            total_vel + vel_loss,
            new_hidden,
            new_prev_switch,
            rng,
        ), (mse, vel_loss)

    hidden_init = jnp.zeros((B, model.hidden_dim))
    prev_switch_init = jnp.full(
        (B, model.obs_shape[0], model.num_modes), 1.0 / model.num_modes
    )

    init = (0.0, 0.0, hidden_init, prev_switch_init, key)
    (total_mse, total_vel, _, _, _), _ = lax.scan(step, init, jnp.arange(T - 1))

    avg_mse = total_mse / T
    avg_vel = (total_vel / T) * model.vel_mult
    total_loss = avg_mse + avg_vel + reg_loss

    aux = {
        "train_loss": total_loss,
        "mse_pos_loss": avg_mse,
        "mse_vel_loss": avg_vel,
        "reg_loss": reg_loss,
        "weight_norm": param_norm,
    }
    return total_loss, aux


@eqx.filter_jit
def rollout_fn(key, model, batch, warmup_steps=5):
    B, T = batch.action.shape
    O, D = model.obs_shape

    hidden = jnp.zeros((B, model.hidden_dim))
    prev_switch = jnp.full((B, O, model.num_modes), 1.0 / model.num_modes)

    pred_obs = []
    switch_states = []

    for t in range(T):
        obs_t = batch.obs[:, t] if t < warmup_steps else obs_t
        action_t = batch.action[:, t]

        key, subkey = jr.split(key)
        step_keys = jr.split(subkey, B)

        def step_fn(o, a, h, p, k):
            carry_in = (h, p)
            pred, (new_h, new_p), logits = model(o, a, carry=carry_in, key=k)
            return pred, new_h, new_p, logits

        preds, hidden, prev_switch, logits = vmap(step_fn)(
            obs_t, action_t, hidden, prev_switch, step_keys
        )

        pred_obs.append(preds)
        switch_states.append(jnp.argmax(logits, axis=-1))
        obs_t = preds

    pred_obs_seq = jnp.stack(pred_obs, axis=1)
    switch_seq = jnp.stack(switch_states, axis=1)
    return pred_obs_seq, switch_seq


def figure_fn(key, model, batch):
    true_obs = batch.obs
    pred_obs, switch_seq = rollout_fn(key, model, batch)

    B, T, O, D = true_obs.shape
    b = 0
    cmap = plt.cm.get_cmap("tab10", model.num_modes)

    fig, axes = plt.subplots(O, D, figsize=(D * 3, O * 3), constrained_layout=True)

    for o in range(O):
        for d in range(D):
            ax = axes[o, d]
            ax.plot(
                true_obs[b, :, o, d], color="black", alpha=0.5, linewidth=2, zorder=1
            )
            scatter = ax.scatter(
                range(T),
                pred_obs[b, :, o, d],
                c=switch_seq[b, :, o],
                cmap=cmap,
                s=50,
                edgecolors="none",
                zorder=2,
            )
            ax.grid(True, linestyle=":", alpha=0.3)
            ax.spines["top"].set_linewidth(1.5)
            ax.spines["right"].set_linewidth(1.5)
            ax.spines["bottom"].set_linewidth(1.5)
            ax.spines["left"].set_linewidth(1.5)
            ax.tick_params(width=1.3)

    return {"rollout_switching": fig}
