import equinox as eqx

from jax import random, numpy as jnp, lax


class Pong(eqx.Module):
    paddle_height: float
    paddle_width: float
    ball_size: float
    player_speed: float
    enemy_speed: float
    ball_speed: float

    def __init__(
        self,
        paddle_height=0.2,
        paddle_width=0.05,
        ball_size=0.05,
        player_speed=0.1,
        enemy_speed=0.1,
        ball_speed=0.05,
    ):
        self.paddle_height = paddle_height
        self.paddle_width = paddle_width
        self.ball_size = ball_size
        self.player_speed = player_speed
        self.enemy_speed = enemy_speed
        self.ball_speed = ball_speed

    def reset(self, rng_key):
        return self._reset(rng_key, 0.0, 0.0)

    def step(self, state, action, rng_key):
        enemy, player, ball = state[0], state[1], state[2]

        player_move = action - 1
        player_delta = player_move * self.player_speed
        player_y = jnp.clip(
            player[1] + player_delta,
            -1.0 + self.paddle_height / 2,
            1.0 - self.paddle_height / 2,
        )
        player_vel = jnp.array([0.0, player_delta])
        player = jnp.array([0.9, player_y, player_vel[0], player_vel[1]])

        delta = ball[1] - enemy[1]
        deadzone = 0.05
        enemy_move = lax.select(
            jnp.abs(delta) > deadzone, jnp.sign(delta) * self.enemy_speed, 0.0
        )
        enemy_y = jnp.clip(
            enemy[1] + enemy_move,
            -1.0 + self.paddle_height / 2,
            1.0 - self.paddle_height / 2,
        )
        enemy_vel = jnp.array([0.0, enemy_move])
        enemy = jnp.array([-0.9, enemy_y, enemy_vel[0], enemy_vel[1]])

        ball_pos = ball[:2]
        ball_vel = ball[2:]
        ball_next = ball_pos + ball_vel

        vel_y = jnp.where(jnp.abs(ball_next[1]) > 1.0, -ball_vel[1], ball_vel[1])
        pos_y = jnp.clip(ball_next[1], -1.0, 1.0)
        ball_next = jnp.array([ball_next[0], pos_y])
        ball_vel = jnp.array([ball_vel[0], vel_y])

        def collide_paddle(ball_p, ball_v, paddle_x, paddle_y):
            within_y = jnp.abs(ball_p[1] - paddle_y) <= (self.paddle_height / 2)
            overlap_x = jnp.abs(ball_p[0] - paddle_x) <= self.paddle_width
            collide = jnp.logical_and(within_y, overlap_x)
            new_vx = jnp.where(collide, -ball_v[0], ball_v[0])
            return ball_p, jnp.array([new_vx, ball_v[1]])

        ball_next, ball_vel = collide_paddle(ball_next, ball_vel, -0.9, enemy_y)
        ball_next, ball_vel = collide_paddle(ball_next, ball_vel, 0.9, player_y)

        ball = jnp.concatenate([ball_next, ball_vel])
        new_state = jnp.stack([enemy, player, ball])

        reward = jnp.where(
            ball_next[0] > 1.0,
            -1.0,
            jnp.where(ball_next[0] < -1.0, 1.0, 0.0),
        )

        reset_event = jnp.logical_or(ball_next[0] > 1.0, ball_next[0] < -1.0)

        def reset_logic():
            return self._reset_ball(state, rng_key)

        new_state = lax.cond(
            reset_event,
            reset_logic,
            lambda: new_state,
        )

        done = False
        return new_state, reward, done

    def render(self, state, width=150, height=210):
        x = jnp.linspace(-1, 1, width)
        y = jnp.linspace(-1, 1, height)
        xx, yy = jnp.meshgrid(x, y[::-1])
        xx = xx[..., None]
        yy = yy[..., None]

        def draw_obj(obj_center, obj_w, obj_h):
            in_x = jnp.abs(xx - obj_center[0]) <= (obj_w / 2)
            in_y = jnp.abs(yy - obj_center[1]) <= (obj_h / 2)
            return jnp.logical_and(in_x, in_y)

        ball = state[2]
        enemy = state[0]
        player = state[1]

        ball_mask = draw_obj(ball[:2], self.ball_size, self.ball_size)
        enemy_mask = draw_obj(enemy[:2], self.paddle_width, self.paddle_height)
        player_mask = draw_obj(player[:2], self.paddle_width, self.paddle_height)

        img = jnp.zeros((height, width, 3))
        img = jnp.where(
            jnp.broadcast_to(ball_mask, img.shape), jnp.array([1, 1, 1]), img
        )
        img = jnp.where(
            jnp.broadcast_to(enemy_mask, img.shape), jnp.array([1, 1, 1]), img
        )
        img = jnp.where(
            jnp.broadcast_to(player_mask, img.shape), jnp.array([1, 1, 1]), img
        )
        return img.clip(0, 1)

    def sample_action(self, rng_key):
        return random.randint(rng_key, (), minval=0, maxval=3)

    def _reset_ball(self, state, rng_key):
        return self._reset(rng_key, state[0, 1], state[1, 1])

    def _reset(self, rng_key, enemy_y, player_y):
        enemy = jnp.array([-0.9, enemy_y, 0.0, 0.0])
        player = jnp.array([0.9, player_y, 0.0, 0.0])
        ball_pos = jnp.array([0.0, 0.0])

        dirs = jnp.array([[-1, 1], [-1, -1], [1, 1], [1, -1]], dtype=jnp.float32)
        # dirs = jnp.array([[-1, 1]], dtype=jnp.float32)
        direction = random.randint(rng_key, (), 0, 4)
        ball_vel = dirs[direction] * self.ball_speed
        ball = jnp.concatenate([ball_pos, ball_vel])

        return jnp.stack([enemy, player, ball])

    @property
    def observation_shape(self):
        return (3, 4)

    @property
    def action_shape(self):
        return 3
